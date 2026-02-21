"""
DeepLeGATo++ Predictor

High-level inference interface for galaxy profile fitting.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

from ..models import DeepLeGAToPP


class Predictor:
    """
    High-level predictor for galaxy profile fitting.
    
    Features:
    - Single and batch prediction
    - Automatic preprocessing
    - Uncertainty quantification
    - Result visualization
    
    Args:
        model: DeepLeGAToPP model
        device: Device for inference
    """
    
    PARAM_NAMES = [
        "magnitude", "effective_radius", "sersic_index",
        "axis_ratio", "position_angle", "center_x", "center_y"
    ]
    
    PARAM_UNITS = [
        "mag", "arcsec", "", "", "deg", "px", "px"
    ]
    
    def __init__(
        self,
        model: DeepLeGAToPP,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "Predictor":
        """
        Load predictor from pretrained model.
        
        Args:
            model_path: Path to saved model
            device: Device for inference
            
        Returns:
            Predictor instance
        """
        model = DeepLeGAToPP.from_pretrained(model_path, device=device)
        return cls(model, device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[torch.Tensor, np.ndarray],
        psf: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_samples: int = 1000,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict Sérsic parameters for a galaxy image.
        
        Args:
            image: Galaxy image (H, W) or (C, H, W) or (B, C, H, W)
            psf: Optional PSF image
            num_samples: Number of posterior samples
            return_samples: Whether to return full posterior samples
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        image = self._preprocess(image)
        if psf is not None:
            psf = self._preprocess(psf)
        
        result = self.model.predict(
            image,
            psf=psf,
            num_samples=num_samples,
            return_credible_intervals=True,
        )
        
        output = self._format_output(result, return_samples)
        
        return output
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: Union[torch.Tensor, np.ndarray, List],
        psfs: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        num_samples: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Predict Sérsic parameters for a batch of galaxy images.
        
        Args:
            images: Batch of galaxy images
            psfs: Optional batch of PSFs
            num_samples: Number of posterior samples per image
            batch_size: Processing batch size
            
        Returns:
            Dictionary with batched predictions
        """
        if isinstance(images, list):
            images = torch.stack([self._preprocess(img).squeeze(0) for img in images])
        else:
            images = self._preprocess(images)
        
        if psfs is not None:
            if isinstance(psfs, list):
                psfs = torch.stack([self._preprocess(p).squeeze(0) for p in psfs])
            else:
                psfs = self._preprocess(psfs)
        
        all_results = []
        n_images = images.shape[0]
        
        for i in range(0, n_images, batch_size):
            batch_images = images[i:i+batch_size]
            batch_psfs = psfs[i:i+batch_size] if psfs is not None else None
            
            result = self.model.predict(
                batch_images,
                psf=batch_psfs,
                num_samples=num_samples,
            )
            all_results.append(result)
        
        combined = {}
        for key in all_results[0].keys():
            combined[key] = torch.cat([r[key] for r in all_results], dim=0)
        
        return self._format_output(combined, return_samples=False)
    
    def _preprocess(
        self,
        image: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()
        
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def _format_output(
        self,
        result: Dict[str, torch.Tensor],
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """Format model output for user-friendly interface."""
        output = {}
        
        mean = result["mean"].cpu().numpy()
        std = result["std"].cpu().numpy()
        
        params = {}
        for i, (name, unit) in enumerate(zip(self.PARAM_NAMES, self.PARAM_UNITS)):
            params[name] = {
                "value": float(mean[0, i]) if mean.shape[0] == 1 else mean[:, i],
                "std": float(std[0, i]) if std.shape[0] == 1 else std[:, i],
                "unit": unit,
            }
            
            if "lower" in result and "upper" in result:
                lower = result["lower"].cpu().numpy()
                upper = result["upper"].cpu().numpy()
                params[name]["ci_lower"] = float(lower[0, i]) if lower.shape[0] == 1 else lower[:, i]
                params[name]["ci_upper"] = float(upper[0, i]) if upper.shape[0] == 1 else upper[:, i]
        
        output["params"] = params
        output["mean"] = mean
        output["std"] = std
        
        if "lower" in result:
            output["ci_lower"] = result["lower"].cpu().numpy()
            output["ci_upper"] = result["upper"].cpu().numpy()
        
        if return_samples and "samples" in result:
            output["samples"] = result["samples"].cpu().numpy()
        
        return output
    
    def print_results(self, result: Dict[str, Any]):
        """Print formatted prediction results."""
        print("\n" + "=" * 60)
        print("DeepLeGATo++ Prediction Results")
        print("=" * 60)
        
        params = result["params"]
        
        for name, data in params.items():
            value = data["value"]
            std = data["std"]
            unit = data["unit"]
            
            if "ci_lower" in data:
                ci_lower = data["ci_lower"]
                ci_upper = data["ci_upper"]
                print(f"{name:20s}: {value:8.3f} ± {std:.3f} {unit:6s}  "
                      f"[95% CI: {ci_lower:.3f} - {ci_upper:.3f}]")
            else:
                print(f"{name:20s}: {value:8.3f} ± {std:.3f} {unit}")
        
        print("=" * 60)
    
    def plot_posterior(
        self,
        result: Dict[str, Any],
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None,
    ):
        """
        Plot posterior distributions for all parameters.
        
        Args:
            result: Prediction result (must include samples)
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        if "samples" not in result:
            print("No samples available. Re-run prediction with return_samples=True")
            return
        
        samples = result["samples"]
        n_params = samples.shape[-1]
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(min(n_params, 7)):
            ax = axes[i]
            param_samples = samples[0, :, i] if samples.ndim == 3 else samples[:, i]
            
            sns.histplot(param_samples, kde=True, ax=ax, color="steelblue")
            ax.axvline(result["mean"][0, i], color="red", linestyle="--", label="Mean")
            
            if "ci_lower" in result:
                ax.axvline(result["ci_lower"][0, i], color="orange", linestyle=":", label="95% CI")
                ax.axvline(result["ci_upper"][0, i], color="orange", linestyle=":")
            
            ax.set_xlabel(f"{self.PARAM_NAMES[i]} ({self.PARAM_UNITS[i]})")
            ax.set_ylabel("Density")
            
            if i == 0:
                ax.legend(fontsize=8)
        
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Testing Predictor...")
    
    model = DeepLeGAToPP(
        backbone_name="swinv2_tiny_window8_256",
        pretrained=False,
        img_size=128,
    )
    
    predictor = Predictor(model, device="cpu")
    
    test_image = torch.randn(128, 128)
    result = predictor.predict(test_image, num_samples=100, return_samples=True)
    
    predictor.print_results(result)
    
    print("\n✓ Predictor test passed!")
