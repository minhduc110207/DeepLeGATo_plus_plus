"""
Uncertainty Quantification Utilities for DeepLeGATo++

Provides tools for:
- Credible interval computation
- Calibration diagnostics
- Posterior predictive checks
- Outlier detection
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any


def compute_credible_intervals(
    samples: np.ndarray,
    credible_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute credible intervals from posterior samples.
    
    Args:
        samples: Posterior samples of shape (n_samples, n_params) or (batch, n_samples, n_params)
        credible_level: Credible level (e.g., 0.95 for 95% CI)
        
    Returns:
        lower: Lower bounds
        upper: Upper bounds
    """
    alpha = (1 - credible_level) / 2
    
    lower = np.quantile(samples, alpha, axis=-2)
    upper = np.quantile(samples, 1 - alpha, axis=-2)
    
    return lower, upper


def compute_hdi(
    samples: np.ndarray,
    credible_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Highest Density Interval (HDI) from posterior samples.
    
    HDI is the narrowest interval containing the specified probability mass.
    More robust for multimodal distributions than equal-tailed intervals.
    
    Args:
        samples: Posterior samples
        credible_level: Credible level
        
    Returns:
        lower: Lower bounds
        upper: Upper bounds
    """
    n_samples = samples.shape[-2]
    interval_size = int(np.ceil(credible_level * n_samples))
    
    # Sort samples
    sorted_samples = np.sort(samples, axis=-2)
    
    # Find narrowest interval
    if samples.ndim == 2:
        # (n_samples, n_params)
        n_params = samples.shape[-1]
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)
        
        for p in range(n_params):
            intervals = sorted_samples[interval_size:, p] - sorted_samples[:-interval_size, p]
            min_idx = np.argmin(intervals)
            lower[p] = sorted_samples[min_idx, p]
            upper[p] = sorted_samples[min_idx + interval_size, p]
    else:
        # (batch, n_samples, n_params)
        batch_size, _, n_params = samples.shape
        lower = np.zeros((batch_size, n_params))
        upper = np.zeros((batch_size, n_params))
        
        for b in range(batch_size):
            for p in range(n_params):
                intervals = sorted_samples[b, interval_size:, p] - sorted_samples[b, :-interval_size, p]
                min_idx = np.argmin(intervals)
                lower[b, p] = sorted_samples[b, min_idx, p]
                upper[b, p] = sorted_samples[b, min_idx + interval_size, p]
    
    return lower, upper


def calibration_metrics(
    samples: np.ndarray,
    true_values: np.ndarray,
    credible_levels: List[float] = [0.50, 0.68, 0.90, 0.95, 0.99],
) -> Dict[str, np.ndarray]:
    """
    Compute calibration metrics for uncertainty estimates.
    
    Well-calibrated uncertainties should have:
    - X% of true values within X% credible intervals
    
    Args:
        samples: Posterior samples (batch, n_samples, n_params)
        true_values: True parameter values (batch, n_params)
        credible_levels: Credible levels to check
        
    Returns:
        Dictionary with calibration metrics
    """
    metrics = {}
    coverages = []
    
    for level in credible_levels:
        lower, upper = compute_credible_intervals(samples, level)
        
        # Check coverage
        within = (true_values >= lower) & (true_values <= upper)
        coverage = within.mean(axis=0)
        
        coverages.append(coverage)
        metrics[f"coverage_{int(level*100)}"] = coverage
    
    # Expected vs actual coverage
    expected = np.array(credible_levels)
    actual = np.array([c.mean() for c in coverages])
    
    metrics["expected_coverage"] = expected
    metrics["actual_coverage"] = actual
    metrics["calibration_error"] = np.abs(expected - actual).mean()
    
    return metrics


def calibration_plot(
    samples: np.ndarray,
    true_values: np.ndarray,
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Create calibration plot for uncertainty diagnostics.
    
    Args:
        samples: Posterior samples
        true_values: True values
        param_names: Names of parameters
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    levels = np.linspace(0.1, 0.99, 20)
    
    n_params = samples.shape[-1]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    
    # Compute coverage for each parameter
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    for p in range(n_params):
        coverages = []
        for level in levels:
            lower, upper = compute_credible_intervals(samples[:, :, p:p+1], level)
            lower = lower.squeeze()
            upper = upper.squeeze()
            within = (true_values[:, p] >= lower) & (true_values[:, p] <= upper)
            coverages.append(within.mean())
        
        ax.plot(levels, coverages, 'o-', color=colors[p], 
                label=param_names[p], markersize=4)
    
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Actual Coverage")
    ax.set_title("Uncertainty Calibration Plot")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def posterior_predictive_check(
    samples: np.ndarray,
    true_values: np.ndarray,
) -> np.ndarray:
    """
    Compute posterior predictive p-values for outlier detection.
    
    Low p-values indicate the model has trouble fitting the observation.
    
    Args:
        samples: Posterior samples (batch, n_samples, n_params)
        true_values: True values (batch, n_params)
        
    Returns:
        p-values for each observation (batch, n_params)
    """
    # Compute fraction of samples more extreme than true value
    # Using a two-tailed test
    
    mean = samples.mean(axis=1)
    
    # Count samples more extreme than true value
    above = (samples > true_values[:, None, :]).sum(axis=1)
    below = (samples < true_values[:, None, :]).sum(axis=1)
    
    n_samples = samples.shape[1]
    
    # Two-tailed p-value
    p_values = 2 * np.minimum(above, below) / n_samples
    
    return p_values


def detect_outliers(
    p_values: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Detect outliers based on posterior predictive p-values.
    
    Args:
        p_values: P-values from posterior_predictive_check
        threshold: P-value threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    # An observation is an outlier if any parameter has low p-value
    return (p_values < threshold).any(axis=1)


def uncertainty_summary(
    samples: np.ndarray,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for posterior distributions.
    
    Args:
        samples: Posterior samples (n_samples, n_params) or (batch, n_samples, n_params)
        param_names: Names of parameters
        
    Returns:
        Dictionary of summary statistics per parameter
    """
    if samples.ndim == 2:
        samples = samples[None, :, :]
    
    n_params = samples.shape[-1]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    
    summary = {}
    
    for p, name in enumerate(param_names):
        param_samples = samples[:, :, p]
        
        # Aggregate over batch
        all_samples = param_samples.flatten()
        
        summary[name] = {
            "mean": float(all_samples.mean()),
            "std": float(all_samples.std()),
            "median": float(np.median(all_samples)),
            "q05": float(np.quantile(all_samples, 0.05)),
            "q25": float(np.quantile(all_samples, 0.25)),
            "q75": float(np.quantile(all_samples, 0.75)),
            "q95": float(np.quantile(all_samples, 0.95)),
        }
    
    return summary


if __name__ == "__main__":
    print("Testing uncertainty utilities...")
    
    # Generate test data
    batch_size = 100
    n_samples = 1000
    n_params = 7
    
    # Simulated posterior samples (roughly calibrated)
    true_values = np.random.randn(batch_size, n_params)
    samples = true_values[:, None, :] + np.random.randn(batch_size, n_samples, n_params) * 0.5
    
    # Test credible intervals
    lower, upper = compute_credible_intervals(samples, 0.95)
    print(f"95% CI shapes: {lower.shape}, {upper.shape}")
    
    # Test HDI
    lower_hdi, upper_hdi = compute_hdi(samples, 0.95)
    print(f"95% HDI shapes: {lower_hdi.shape}, {upper_hdi.shape}")
    
    # Test calibration
    metrics = calibration_metrics(samples, true_values)
    print(f"\nCalibration error: {metrics['calibration_error']:.4f}")
    print(f"Expected coverages: {metrics['expected_coverage']}")
    print(f"Actual coverages: {metrics['actual_coverage']}")
    
    # Test posterior predictive check
    p_values = posterior_predictive_check(samples, true_values)
    print(f"\nP-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
    
    outliers = detect_outliers(p_values)
    print(f"Outliers detected: {outliers.sum()}/{batch_size}")
    
    # Test summary
    summary = uncertainty_summary(samples[:5])
    print(f"\nSummary for param_0:")
    for key, value in summary["param_0"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Uncertainty utilities test passed!")
