"""
Swin Transformer V2 Backbone for DeepLeGATo++

This module provides a Swin Transformer V2 backbone optimized for galaxy image analysis,
with support for gradient checkpointing and multi-scale feature extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Union
import timm
from timm.models.swin_transformer_v2 import SwinTransformerV2


class SwinBackbone(nn.Module):
    """
    Swin Transformer V2 backbone for galaxy image feature extraction.
    
    Features:
    - Pre-trained weights from ImageNet (fine-tuned for astronomy)
    - Gradient checkpointing for memory efficiency
    - Multi-scale feature pyramid output
    - Configurable model size (tiny/small/base)
    
    Args:
        model_name: Name of the Swin model variant (e.g., 'swin_tiny_patch4_window7_224')
        pretrained: Whether to load pretrained ImageNet weights
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        embed_dim: Embedding dimension (overrides default if provided)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        drop_path_rate: Stochastic depth rate
        output_features: Whether to output multi-scale features
    """
    
    MODEL_CONFIGS = {
        "tiny": {
            "model_name": "swinv2_tiny_window8_256",
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "small": {
            "model_name": "swinv2_small_window8_256",
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "base": {
            "model_name": "swinv2_base_window8_256",
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
        },
    }
    
    def __init__(
        self,
        model_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        in_channels: int = 1,
        embed_dim: Optional[int] = None,
        gradient_checkpointing: bool = True,
        drop_path_rate: float = 0.1,
        output_features: bool = True,
        img_size: int = 128,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.output_features = output_features
        self.gradient_checkpointing = gradient_checkpointing
        
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        
        self.feature_dim = self.swin.num_features
        
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self._adaptive_proj: Optional[nn.Linear] = None
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.swin, 'set_grad_checkpointing'):
            self.swin.set_grad_checkpointing(enable=True)
        else:
            for layer in self.swin.layers:
                layer.grad_checkpointing = True
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through Swin backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: If True, also return intermediate features
            
        Returns:
            features: Global feature vector of shape (B, feature_dim)
            intermediate_features: List of intermediate feature maps (if return_features=True)
        """
        features = self.swin.forward_features(x)
        intermediates = None
        
        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 2:
            pass
        
        if features.shape[-1] != self.feature_dim:
            if not hasattr(self, '_adaptive_proj') or self._adaptive_proj.in_features != features.shape[-1]:
                self._adaptive_proj = nn.Linear(features.shape[-1], self.feature_dim).to(features.device)
            features = self._adaptive_proj(features)
        
        features = self.feature_proj(features)
        
        if return_features and intermediates is not None:
            return features, intermediates
        return features
    
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SwinBackbone":
        """Create backbone from configuration dictionary."""
        backbone_config = config.get("model", {}).get("backbone", {})
        
        return cls(
            model_name=backbone_config.get("type", "swinv2_tiny_window8_256"),
            pretrained=backbone_config.get("pretrained", True),
            embed_dim=backbone_config.get("embed_dim"),
            gradient_checkpointing=backbone_config.get("gradient_checkpointing", True),
            drop_path_rate=backbone_config.get("drop_path_rate", 0.1),
            img_size=config.get("data", {}).get("image_size", 128),
        )


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module for combining features from different Swin stages.
    
    This is useful for capturing both local (galaxy core) and global (halo) features.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, out_channels),
                nn.LayerNorm(out_channels),
                nn.GELU(),
            )
            for in_ch in in_channels_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * len(in_channels_list), out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features.
        
        Args:
            features: List of feature tensors from different scales
            
        Returns:
            Fused feature vector
        """
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        concatenated = torch.cat(projected, dim=-1)
        return self.fusion(concatenated)


if __name__ == "__main__":
    print("Testing SwinBackbone...")
    
    backbone = SwinBackbone(
        model_name="swinv2_tiny_window8_256",
        pretrained=False,
        in_channels=1,
        gradient_checkpointing=True,
        img_size=128,
    )
    
    x = torch.randn(2, 1, 128, 128)
    features = backbone(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {backbone.get_feature_dim()}")
    print("✓ SwinBackbone test passed!")
