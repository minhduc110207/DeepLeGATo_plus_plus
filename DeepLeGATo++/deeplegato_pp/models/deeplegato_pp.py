"""
DeepLeGATo++ Main Model

Combines Swin Transformer V2 backbone with Neural Posterior Estimation head
for galaxy surface brightness profile fitting with uncertainty quantification.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
import yaml

from .swin_backbone import SwinBackbone
from .npe_head import NPEHead


class DeepLeGAToPP(nn.Module):
    """
    DeepLeGATo++ - Next-Generation Galaxy Profile Fitting Model.
    
    Architecture:
        Galaxy Image → Swin Transformer V2 → NPE Head → Posterior Distribution
    
    Features:
    - Multi-scale feature extraction with Swin Transformer V2
    - Neural Posterior Estimation for full uncertainty quantification
    - Support for single and multi-band images
    - PSF-aware processing (optional)
    - Gradient checkpointing for memory efficiency
    
    Args:
        backbone_config: Configuration for Swin backbone
        npe_config: Configuration for NPE head
        in_channels: Number of input channels (1 for grayscale galaxy images)
        num_params: Number of Sérsic parameters to estimate
        img_size: Input image size
        use_psf: Whether to use PSF as additional input
    """
    
    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        in_channels: int = 1,
        num_params: int = 7,
        img_size: int = 128,
        hidden_dim: int = 256,
        num_flow_layers: int = 8,
        gradient_checkpointing: bool = True,
        use_psf: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_params = num_params
        self.use_psf = use_psf
        self.img_size = img_size
        
        effective_in_channels = in_channels + 1 if use_psf else in_channels
        
        self.backbone = SwinBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            in_channels=effective_in_channels,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        
        feature_dim = self.backbone.get_feature_dim()
        
        self.npe_head = NPEHead(
            feature_dim=feature_dim,
            num_params=num_params,
            hidden_dim=hidden_dim,
            num_flow_layers=num_flow_layers,
        )
        
        self.use_reconstruction = False
        
    def forward(
        self,
        x: torch.Tensor,
        psf: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DeepLeGATo++.
        
        Args:
            x: Galaxy images of shape (B, C, H, W)
            psf: Optional PSF images of shape (B, 1, H, W)
            num_samples: Number of posterior samples to draw
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'samples': Posterior samples (B, num_samples, num_params)
                - 'mean': Posterior mean estimates (B, num_params)
                - 'std': Posterior standard deviation (B, num_params)
                - 'features': Backbone features (if return_features=True)
        """
        if self.use_psf and psf is not None:
            x = torch.cat([x, psf], dim=1)
        elif self.use_psf and psf is None:
            psf_placeholder = torch.zeros_like(x[:, :1])
            x = torch.cat([x, psf_placeholder], dim=1)
        
        if return_features:
            features, intermediate = self.backbone(x, return_features=True)
        else:
            features = self.backbone(x)
        
        result = self.npe_head(features, num_samples=num_samples)
        
        if return_features:
            result['features'] = features
            result['intermediate_features'] = intermediate
        
        return result
    
    def predict(
        self,
        x: torch.Tensor,
        psf: Optional[torch.Tensor] = None,
        num_samples: int = 100,
        return_credible_intervals: bool = True,
        credible_level: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Galaxy images
            psf: Optional PSF images
            num_samples: Number of samples for uncertainty estimation
            return_credible_intervals: Whether to compute credible intervals
            credible_level: Level for credible intervals (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x, psf, num_samples=num_samples)
            
            if return_credible_intervals:
                samples = result['samples']
                alpha = (1 - credible_level) / 2
                result['lower'] = torch.quantile(samples, alpha, dim=1)
                result['upper'] = torch.quantile(samples, 1 - alpha, dim=1)
                result['median'] = torch.quantile(samples, 0.5, dim=1)
        
        return result
    
    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        psf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x: Galaxy images (B, C, H, W)
            targets: Ground truth parameters (B, num_params)
            psf: Optional PSF images
            
        Returns:
            Dictionary with loss components
        """
        if self.use_psf and psf is not None:
            x = torch.cat([x, psf], dim=1)
        
        features = self.backbone(x)
        
        loss_dict = self.npe_head.compute_loss(features, targets)
        
        return loss_dict
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """
        Save model weights and configuration.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), save_path / "model.pt")
        
        config = {
            "backbone_name": self.backbone.model_name,
            "in_channels": self.in_channels,
            "num_params": self.num_params,
            "img_size": self.img_size,
            "use_psf": self.use_psf,
            "feature_dim": self.backbone.get_feature_dim(),
            "hidden_dim": self.npe_head.hidden_dim,
            "num_flow_layers": self.npe_head.num_flow_layers,
        }
        
        with open(save_path / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        load_path: Union[str, Path],
        device: str = "cuda",
    ) -> "DeepLeGAToPP":
        """
        Load a pretrained model.
        
        Args:
            load_path: Path to saved model directory
            device: Device to load model to
            
        Returns:
            Loaded model
        """
        load_path = Path(load_path)
        
        with open(load_path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        model = cls(
            backbone_name=config["backbone_name"],
            pretrained=False,  # Don't load ImageNet weights
            in_channels=config["in_channels"],
            num_params=config["num_params"],
            img_size=config["img_size"],
            hidden_dim=config["hidden_dim"],
            num_flow_layers=config["num_flow_layers"],
            use_psf=config["use_psf"],
        )
        
        state_dict = torch.load(load_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        print(f"Model loaded from {load_path}")
        return model
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepLeGAToPP":
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration dictionary (from YAML file)
            
        Returns:
            Initialized model
        """
        model_config = config.get("model", {})
        backbone_config = model_config.get("backbone", {})
        npe_config = model_config.get("npe", {})
        data_config = config.get("data", {})
        
        return cls(
            backbone_name=backbone_config.get("type", "swinv2_tiny_window8_256"),
            pretrained=backbone_config.get("pretrained", True),
            in_channels=1,
            num_params=model_config.get("num_params", 7),
            img_size=data_config.get("image_size", 128),
            hidden_dim=npe_config.get("hidden_dim", 256),
            num_flow_layers=npe_config.get("num_flow_layers", 8),
            gradient_checkpointing=backbone_config.get("gradient_checkpointing", True),
            drop_path_rate=backbone_config.get("drop_path_rate", 0.1),
        )
    
    def get_param_names(self) -> List[str]:
        """Get names of output parameters."""
        return self.npe_head.PARAM_NAMES[:self.num_params]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        npe_params = sum(p.numel() for p in self.npe_head.parameters())
        total_params = backbone_params + npe_params
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "backbone": backbone_params,
            "npe_head": npe_params,
            "total": total_params,
            "trainable": trainable,
        }


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print("Testing DeepLeGAToPP...")
    
    model = DeepLeGAToPP(
        backbone_name="swinv2_tiny_window8_256",
        pretrained=False,
        in_channels=1,
        num_params=7,
        img_size=128,
        gradient_checkpointing=True,
    )
    
    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
    print(f"  Backbone: {param_counts['backbone']:,}")
    print(f"  NPE Head: {param_counts['npe_head']:,}")
    
    x = torch.randn(4, 1, 128, 128)
    result = model(x, num_samples=50)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Samples shape: {result['samples'].shape}")
    print(f"Mean shape: {result['mean'].shape}")
    print(f"Std shape: {result['std'].shape}")
    
    predictions = model.predict(x, num_samples=100)
    print(f"\nPrediction keys: {list(predictions.keys())}")
    print(f"95% CI lower shape: {predictions['lower'].shape}")
    print(f"95% CI upper shape: {predictions['upper'].shape}")
    
    targets = torch.rand(4, 7)
    targets[:, 0] = targets[:, 0] * 10 + 18
    targets[:, 1] = targets[:, 1] * 5 + 0.5
    targets[:, 2] = targets[:, 2] * 4 + 1
    targets[:, 3] = targets[:, 3] * 0.8 + 0.2
    targets[:, 4] = targets[:, 4] * 180
    targets[:, 5] = targets[:, 5] * 4 - 2
    targets[:, 6] = targets[:, 6] * 4 - 2
    
    loss_dict = model.compute_loss(x, targets)
    print(f"\nLoss: {loss_dict['loss'].item():.4f}")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded_model = DeepLeGAToPP.from_pretrained(tmpdir, device="cpu")
        print(f"\n✓ Save/load test passed!")
    
    print("\n✓ DeepLeGAToPP test passed!")
