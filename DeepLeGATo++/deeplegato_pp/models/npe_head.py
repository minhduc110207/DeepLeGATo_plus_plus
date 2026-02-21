"""
Neural Posterior Estimation (NPE) Head for DeepLeGATo++

This module implements Neural Posterior Estimation using Normalizing Flows
to produce full posterior distributions over Sérsic parameters with uncertainty
quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from typing import Optional, Tuple, Dict, Any, List
import math

try:
    import nflows
    from nflows import distributions, flows, transforms
    from nflows.nn import nets
    NFLOWS_AVAILABLE = True
except ImportError:
    NFLOWS_AVAILABLE = False
    print("Warning: nflows not installed. Using simplified flow implementation.")


class NPEHead(nn.Module):
    """
    Neural Posterior Estimation head using Normalizing Flows.
    
    Given features from the Swin backbone, this module learns to approximate
    the posterior distribution p(θ|x) where θ are Sérsic parameters and x is
    the galaxy image.
    
    Features:
    - Masked Autoregressive Flow (MAF) for flexible posterior approximation
    - Conditional flow based on image features
    - Posterior sampling for uncertainty quantification
    - Support for parameter transformations (bounded → unbounded)
    
    Args:
        feature_dim: Dimension of input features from backbone
        num_params: Number of output parameters (default: 7 for full Sérsic)
        hidden_dim: Hidden dimension for flow networks
        num_flow_layers: Number of flow transformation layers
        num_transforms: Number of transforms per layer
        num_blocks: Number of residual blocks in each transform
        dropout: Dropout rate
        param_bounds: Dictionary of (min, max) bounds for each parameter
    """
    
    # Default parameter bounds for Sérsic fitting
    DEFAULT_BOUNDS = {
        "magnitude": (15.0, 28.0),
        "effective_radius": (0.1, 10.0),
        "sersic_index": (0.3, 8.0),
        "axis_ratio": (0.1, 1.0),
        "position_angle": (0.0, 180.0),
        "center_x": (-5.0, 5.0),
        "center_y": (-5.0, 5.0),
    }
    
    PARAM_NAMES = [
        "magnitude", "effective_radius", "sersic_index",
        "axis_ratio", "position_angle", "center_x", "center_y"
    ]
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_params: int = 7,
        hidden_dim: int = 256,
        num_flow_layers: int = 8,
        num_transforms: int = 5,
        num_blocks: int = 2,
        dropout: float = 0.1,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_params = num_params
        self.hidden_dim = hidden_dim
        self.num_flow_layers = num_flow_layers
        
        self.param_bounds = param_bounds or self.DEFAULT_BOUNDS
        self._register_bounds()
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        if NFLOWS_AVAILABLE:
            self.flow = self._build_nflows_model(
                num_params, hidden_dim, num_flow_layers, num_transforms, num_blocks
            )
        else:
            self.flow = self._build_simple_flow(
                num_params, hidden_dim, num_flow_layers
            )
        
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_params),
        )
        
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_params),
        )
        
    def _register_bounds(self):
        """Register parameter bounds as buffers."""
        bounds_list = [self.param_bounds.get(name, (-10, 10)) 
                       for name in self.PARAM_NAMES[:self.num_params]]
        
        lower = torch.tensor([b[0] for b in bounds_list])
        upper = torch.tensor([b[1] for b in bounds_list])
        
        self.register_buffer("param_lower", lower)
        self.register_buffer("param_upper", upper)
        
    def _build_nflows_model(
        self,
        num_params: int,
        hidden_dim: int,
        num_layers: int,
        num_transforms: int,
        num_blocks: int,
    ):
        """Build normalizing flow using nflows library."""
        base_distribution = distributions.StandardNormal(shape=[num_params])
        
        transform_list = []
        for _ in range(num_layers):
            transform_list.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=num_params,
                    hidden_features=hidden_dim,
                    context_features=hidden_dim,
                    num_blocks=num_blocks,
                    use_residual_blocks=True,
                    activation=F.gelu,
                    dropout_probability=0.1,
                )
            )
            transform_list.append(
                transforms.ReversePermutation(features=num_params)
            )
        
        transform = transforms.CompositeTransform(transform_list)
        
        flow = flows.Flow(transform, base_distribution)
        
        return flow
    
    def _build_simple_flow(
        self,
        num_params: int,
        hidden_dim: int,
        num_layers: int,
    ):
        return SimpleFlow(num_params, hidden_dim, num_layers)
    
    def forward(
        self,
        features: torch.Tensor,
        num_samples: int = 1,
        return_log_prob: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute posterior samples and statistics.
        
        Args:
            features: Backbone features of shape (B, feature_dim)
            num_samples: Number of posterior samples to draw
            return_log_prob: Whether to return log probabilities
            
        Returns:
            Dictionary containing:
                - 'samples': Posterior samples (B, num_samples, num_params)
                - 'mean': Posterior mean (B, num_params)
                - 'std': Posterior standard deviation (B, num_params)
                - 'log_prob': Log probability (if requested)
        """
        batch_size = features.shape[0]
        
        context = self.feature_embedding(features)
        
        mean_pred = self.mean_head(context)
        logvar_pred = self.logvar_head(context)
        std_pred = torch.exp(0.5 * logvar_pred)
        
        mean_bounded = self._to_bounded(mean_pred)
        
        if NFLOWS_AVAILABLE and num_samples > 1:
            context_expanded = context.unsqueeze(1).expand(-1, num_samples, -1)
            context_flat = context_expanded.reshape(-1, self.hidden_dim)
            
            samples_unbounded = self.flow.sample(1, context=context_flat)
            
            if samples_unbounded.dim() == 3:
                samples_unbounded = samples_unbounded.squeeze(0)
            
            samples_unbounded = samples_unbounded.reshape(batch_size, num_samples, self.num_params)
            samples = self._to_bounded(samples_unbounded)
            
            sample_mean = samples.mean(dim=1)
            sample_std = samples.std(dim=1)
        else:
            dist = Normal(mean_pred, std_pred + 1e-6)
            samples_unbounded = dist.rsample((num_samples,)).permute(1, 0, 2)
            samples = self._to_bounded(samples_unbounded)
            sample_mean = mean_bounded
            sample_std = std_pred * (self.param_upper - self.param_lower) / 4
        
        result = {
            "samples": samples,
            "mean": sample_mean,
            "std": sample_std,
            "mean_unbounded": mean_pred,
            "logvar": logvar_pred,
        }
        
        if return_log_prob and NFLOWS_AVAILABLE:
            log_prob = self.flow.log_prob(
                samples_unbounded.reshape(-1, self.num_params),
                context=context.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.hidden_dim)
            )
            result["log_prob"] = log_prob.reshape(batch_size, num_samples)
        
        return result
    
    def _to_bounded(self, x: torch.Tensor) -> torch.Tensor:
        """Transform unbounded values to bounded parameter space using sigmoid."""
        normalized = torch.sigmoid(x)
        return self.param_lower + normalized * (self.param_upper - self.param_lower)
    
    def _to_unbounded(self, x: torch.Tensor) -> torch.Tensor:
        """Transform bounded parameters to unbounded space using logit."""
        normalized = (x - self.param_lower) / (self.param_upper - self.param_lower)
        normalized = torch.clamp(normalized, 1e-6, 1 - 1e-6)
        return torch.log(normalized / (1 - normalized))
    
    def compute_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NPE training loss.
        
        Args:
            features: Backbone features (B, feature_dim)
            targets: Ground truth parameters (B, num_params)
            
        Returns:
            Dictionary containing loss components
        """
        context = self.feature_embedding(features)
        
        mean_pred = self.mean_head(context)
        logvar_pred = self.logvar_head(context)
        
        targets_unbounded = self._to_unbounded(targets)
        
        nll_loss = 0.5 * (logvar_pred + (targets_unbounded - mean_pred) ** 2 / torch.exp(logvar_pred))
        nll_loss = nll_loss.mean()
        
        if NFLOWS_AVAILABLE:
            flow_log_prob = self.flow.log_prob(targets_unbounded, context=context)
            flow_loss = -flow_log_prob.mean()
        else:
            flow_loss = torch.tensor(0.0, device=features.device)
        
        total_loss = nll_loss + flow_loss
        
        return {
            "loss": total_loss,
            "nll_loss": nll_loss,
            "flow_loss": flow_loss,
        }
    
    def get_credible_interval(
        self,
        features: torch.Tensor,
        num_samples: int = 1000,
        credible_level: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute credible intervals from posterior samples.
        
        Args:
            features: Backbone features
            num_samples: Number of samples for estimation
            credible_level: Credible interval level (e.g., 0.95 for 95%)
            
        Returns:
            lower: Lower bound of credible interval (B, num_params)
            upper: Upper bound of credible interval (B, num_params)
        """
        result = self.forward(features, num_samples=num_samples)
        samples = result["samples"]  # (B, num_samples, num_params)
        
        alpha = (1 - credible_level) / 2
        lower = torch.quantile(samples, alpha, dim=1)
        upper = torch.quantile(samples, 1 - alpha, dim=1)
        
        return lower, upper
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], feature_dim: int) -> "NPEHead":
        """Create NPE head from configuration dictionary."""
        npe_config = config.get("model", {}).get("npe", {})
        
        return cls(
            feature_dim=feature_dim,
            num_params=config.get("model", {}).get("num_params", 7),
            hidden_dim=npe_config.get("hidden_dim", 256),
            num_flow_layers=npe_config.get("num_flow_layers", 8),
            num_transforms=npe_config.get("num_transforms", 5),
            num_blocks=npe_config.get("num_blocks", 2),
            dropout=npe_config.get("dropout", 0.1),
        )


class SimpleFlow(nn.Module):
    """
    Simple normalizing flow for when nflows is not available.
    Uses affine coupling layers.
    """
    
    def __init__(self, num_params: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        self.num_params = num_params
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            split_dim = num_params // 2 if i % 2 == 0 else num_params - num_params // 2
            
            self.layers.append(
                AffineCouplingLayer(
                    input_dim=num_params,
                    hidden_dim=hidden_dim,
                    context_dim=hidden_dim,
                    split_dim=split_dim,
                )
            )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through flow."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        for layer in self.layers:
            x, ld = layer(x, context)
            log_det = log_det + ld
        
        return x, log_det
    
    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse pass (sampling)."""
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        return z
    
    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Sample from the flow."""
        batch_size = context.shape[0] // num_samples if context.dim() == 2 else context.shape[0]
        z = torch.randn(context.shape[0], self.num_params, device=context.device)
        return self.inverse(z, context)
    
    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        z, log_det = self.forward(x, context)
        log_prob_z = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_prob_z + log_det


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int,
        split_dim: int,
    ):
        super().__init__()
        
        self.split_dim = split_dim
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(split_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, (input_dim - split_dim) * 2),
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        params = self.net(torch.cat([x1, context], dim=-1))
        log_scale, translation = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2
        
        y2 = x2 * torch.exp(log_scale) + translation
        y = torch.cat([x1, y2], dim=-1)
        
        log_det = log_scale.sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse pass."""
        y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]
        
        params = self.net(torch.cat([y1, context], dim=-1))
        log_scale, translation = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2
        
        x2 = (y2 - translation) * torch.exp(-log_scale)
        x = torch.cat([y1, x2], dim=-1)
        
        return x


if __name__ == "__main__":
    print("Testing NPEHead...")
    
    npe = NPEHead(
        feature_dim=768,
        num_params=7,
        hidden_dim=256,
        num_flow_layers=4,
    )
    
    features = torch.randn(4, 768)
    result = npe(features, num_samples=100)
    
    print(f"Input shape: {features.shape}")
    print(f"Samples shape: {result['samples'].shape}")
    print(f"Mean shape: {result['mean'].shape}")
    print(f"Std shape: {result['std'].shape}")
    
    targets = torch.rand(4, 7) * 10 + 15
    targets[:, 3] = torch.rand(4) * 0.9 + 0.1
    loss_dict = npe.compute_loss(features, targets)
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    lower, upper = npe.get_credible_interval(features, num_samples=500)
    print(f"Credible interval shapes: {lower.shape}, {upper.shape}")
    
    print("✓ NPEHead test passed!")
