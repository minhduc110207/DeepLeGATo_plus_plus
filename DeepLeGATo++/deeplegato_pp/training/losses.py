"""
Loss Functions for DeepLeGATo++

Implements loss functions for Neural Posterior Estimation and
optional reconstruction losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class NPELoss(nn.Module):
    """
    Combined loss for Neural Posterior Estimation.
    
    Components:
    - Negative log-likelihood (Gaussian approximation)
    - Flow loss (if using normalizing flows)
    - Optional reconstruction loss
    
    Args:
        nll_weight: Weight for NLL loss
        flow_weight: Weight for flow loss
        recon_weight: Weight for reconstruction loss
    """
    
    def __init__(
        self,
        nll_weight: float = 1.0,
        flow_weight: float = 1.0,
        recon_weight: float = 0.0,
    ):
        super().__init__()
        
        self.nll_weight = nll_weight
        self.flow_weight = flow_weight
        self.recon_weight = recon_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        reconstruction: Optional[torch.Tensor] = None,
        input_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions containing 'mean_unbounded', 'logvar', etc.
            targets: Ground truth parameters (unbounded space)
            reconstruction: Optional reconstructed images
            input_images: Optional input images for reconstruction loss
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        mean = predictions["mean_unbounded"]
        logvar = predictions["logvar"]
        
        nll = 0.5 * (logvar + (targets - mean) ** 2 / torch.exp(logvar))
        nll = nll.mean()
        losses["nll_loss"] = nll * self.nll_weight
        
        if "log_prob" in predictions:
            flow_loss = -predictions["log_prob"].mean()
            losses["flow_loss"] = flow_loss * self.flow_weight
        else:
            losses["flow_loss"] = torch.tensor(0.0, device=mean.device)
        
        if self.recon_weight > 0 and reconstruction is not None and input_images is not None:
            recon_loss = F.mse_loss(reconstruction, input_images)
            losses["recon_loss"] = recon_loss * self.recon_weight
        else:
            losses["recon_loss"] = torch.tensor(0.0, device=mean.device)
        
        losses["loss"] = losses["nll_loss"] + losses["flow_loss"] + losses["recon_loss"]
        
        return losses


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for self-supervised pre-training.
    
    Can be used for:
    - Autoencoder-style pre-training
    - Profile matching validation
    
    Args:
        loss_type: Type of loss ('mse', 'l1', 'huber')
        reduction: Reduction method
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstruction: Reconstructed image
            target: Target image
            mask: Optional mask for weighted loss
            
        Returns:
            Loss value
        """
        if mask is not None:
            reconstruction = reconstruction * mask
            target = target * mask
        
        return self.loss_fn(reconstruction, target)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed regularization for Sérsic parameters.
    
    Adds soft constraints based on physical priors:
    - Luminosity-size relation
    - Sérsic index distributions
    - Axis ratio constraints
    """
    
    def __init__(
        self,
        relation_weight: float = 0.1,
        prior_weight: float = 0.1,
    ):
        super().__init__()
        
        self.relation_weight = relation_weight
        self.prior_weight = prior_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed regularization.
        
        Args:
            predictions: Model predictions with 'mean' (bounded parameters)
            
        Returns:
            Dictionary of regularization losses
        """
        mean = predictions["mean"]
        
        losses = {}
        
        mag = mean[:, 0]
        re = mean[:, 1]
        
        expected_log_re = -0.3 * (mag - 20) + 0.5
        actual_log_re = torch.log(re + 0.1)
        
        relation_loss = F.mse_loss(actual_log_re, expected_log_re)
        losses["relation_loss"] = relation_loss * self.relation_weight
        
        n = mean[:, 2]
        n_prior_loss = F.relu(n - 6) ** 2 + F.relu(0.5 - n) ** 2
        losses["n_prior_loss"] = n_prior_loss.mean() * self.prior_weight
        
        losses["physics_loss"] = losses["relation_loss"] + losses["n_prior_loss"]
        
        return losses


class UncertaintyCalibrationLoss(nn.Module):
    """
    Loss for calibrating uncertainty estimates.
    
    Ensures that the predicted uncertainties are well-calibrated:
    - 95% of true values should fall within 95% CI
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        credible_level: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute calibration loss.
        
        Args:
            predictions: Model predictions with samples
            targets: True parameter values
            credible_level: Target credible level
            
        Returns:
            Calibration loss
        """
        samples = predictions["samples"]
        
        alpha = (1 - credible_level) / 2
        lower = torch.quantile(samples, alpha, dim=1)
        upper = torch.quantile(samples, 1 - alpha, dim=1)
        
        within = ((targets >= lower) & (targets <= upper)).float()
        coverage = within.mean(dim=0)
        
        target_coverage = torch.tensor(credible_level, device=coverage.device)
        calibration_loss = (coverage - target_coverage) ** 2
        
        return calibration_loss.mean()


if __name__ == "__main__":
    print("Testing loss functions...")
    
    batch_size = 8
    num_params = 7
    num_samples = 100
    
    predictions = {
        "mean_unbounded": torch.randn(batch_size, num_params),
        "logvar": torch.randn(batch_size, num_params),
        "mean": torch.rand(batch_size, num_params) * 10,
        "samples": torch.randn(batch_size, num_samples, num_params),
    }
    
    targets = torch.randn(batch_size, num_params)
    
    npe_loss = NPELoss()
    losses = npe_loss(predictions, targets)
    print(f"NPE Loss: {losses['loss'].item():.4f}")
    print(f"  NLL: {losses['nll_loss'].item():.4f}")
    print(f"  Flow: {losses['flow_loss'].item():.4f}")
    
    physics_loss = PhysicsInformedLoss()
    physics_losses = physics_loss(predictions)
    print(f"Physics Loss: {physics_losses['physics_loss'].item():.4f}")
    
    cal_loss = UncertaintyCalibrationLoss()
    cal = cal_loss(predictions, predictions["mean"])
    print(f"Calibration Loss: {cal.item():.4f}")
    
    print("\n✓ Loss function tests passed!")
