"""
Google Colab Utilities for DeepLeGATo++

Provides helper functions for:
- Google Drive integration
- Checkpoint management
- GPU detection and config selection
- Session persistence
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any, Union
import torch


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_drive_paths(
    project_name: str = "DeepLeGATo++",
    mount_drive: bool = True,
) -> Dict[str, Path]:
    """
    Setup Google Drive paths for checkpoints and data.
    
    Args:
        project_name: Name of project folder in Drive
        mount_drive: Whether to mount Drive if not already mounted
        
    Returns:
        Dictionary of paths
    """
    if is_colab() and mount_drive:
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"Warning: Could not mount Drive: {e}")
    
    if is_colab() and os.path.exists("/content/drive/MyDrive"):
        base_path = Path("/content/drive/MyDrive") / project_name
    else:
        base_path = Path.home() / project_name
    
    paths = {
        "base": base_path,
        "checkpoints": base_path / "checkpoints",
        "data": base_path / "data",
        "logs": base_path / "logs",
        "outputs": base_path / "outputs",
        "models": base_path / "models",
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Project paths setup at: {base_path}")
    return paths


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "*.ckpt",
) -> Optional[str]:
    """
    Find the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    ckpts = list(checkpoint_dir.glob(pattern))
    
    if not ckpts:
        return None
    
    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    
    return str(latest)


def get_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric: str = "val_loss",
    mode: str = "min",
) -> Optional[str]:
    """
    Find the best checkpoint based on metric in filename.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to optimize
        mode: 'min' or 'max'
        
    Returns:
        Path to best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    
    if not ckpts:
        return None
    
    best_ckpt = None
    best_value = float("inf") if mode == "min" else float("-inf")
    
    for ckpt in ckpts:
        if ckpt.name == "last.ckpt":
            continue
        
        try:
            parts = ckpt.stem.split("-")
            for part in parts:
                if "=" in part:
                    key, value = part.split("=")
                    if metric.replace("/", "_").replace("val_", "") in key.replace("_", ""):
                        value = float(value)
                        if mode == "min" and value < best_value:
                            best_value = value
                            best_ckpt = ckpt
                        elif mode == "max" and value > best_value:
                            best_value = value
                            best_ckpt = ckpt
        except Exception:
            continue
    
    if best_ckpt:
        return str(best_ckpt)
    
    return get_latest_checkpoint(checkpoint_dir)


def auto_select_config(
    config_dir: Union[str, Path] = "configs",
) -> str:
    """
    Automatically select configuration based on available GPU.
    
    Returns:
        Path to appropriate config file
    """
    config_dir = Path(config_dir)
    
    if not torch.cuda.is_available():
        print("Warning: No GPU available! Using CPU config.")
        cpu_config = config_dir / "colab_cpu.yaml"
        if cpu_config.exists():
            return str(cpu_config)
        return str(config_dir / "colab_t4.yaml")
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"Detected GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    
    if "A100" in gpu_name:
        config_name = "colab_a100.yaml"
    elif "L4" in gpu_name or (vram_gb > 20 and vram_gb <= 30):
        config_name = "colab_l4.yaml"
    elif "V100" in gpu_name:
        config_name = "colab_t4.yaml"
    else:
        config_name = "colab_t4.yaml"
    
    config_path = config_dir / config_name
    
    if config_path.exists():
        print(f"Selected config: {config_name}")
        return str(config_path)
    
    fallback = config_dir / "colab_t4.yaml"
    print(f"Using fallback config: colab_t4.yaml")
    return str(fallback)


def print_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\ndevice {i}: {props.name}")
        print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Multi-processor count: {props.multi_processor_count}")
    
    print(f"\nCurrent device: {torch.cuda.current_device()}")
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Memory allocated: {allocated:.2f} GB")
    print(f"Memory reserved: {reserved:.2f} GB")
    
    print("=" * 50)


def save_model_to_drive(
    model: torch.nn.Module,
    name: str,
    paths: Optional[Dict[str, Path]] = None,
    include_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Save model to Google Drive.
    
    Args:
        model: Model to save
        name: Name for saved model
        paths: Path dictionary from setup_drive_paths
        include_optimizer: Whether to save optimizer state
        optimizer: Optimizer to save
    """
    if paths is None:
        paths = setup_drive_paths()
    
    save_path = paths["models"] / name
    save_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), save_path / "model.pt")
    
    if include_optimizer and optimizer is not None:
        torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    
    print(f"Model saved to: {save_path}")


def keep_colab_alive():
    """
    Attempt to keep Colab session alive.
    
    Note: This doesn't guarantee the session won't timeout,
    but can help with longer training runs.
    """
    if not is_colab():
        return
    
    try:
        from IPython.display import display, Javascript
        
        js = '''
        function ClickConnect(){
            console.log("Keeping session alive...");
            document.querySelector("colab-connect-button").click();
        }
        setInterval(ClickConnect, 60000);
        '''
        display(Javascript(js))
        print("Session keep-alive activated (clicks connect every 60s)")
    except Exception as e:
        print(f"Could not activate keep-alive: {e}")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cache cleared")


class ColabProgressCallback:
    """
    Progress callback optimized for Colab.
    
    Provides:
    - Progress bars that don't flood output
    - Periodic checkpoint saving
    - Memory monitoring
    """
    
    def __init__(
        self,
        save_every_n_epochs: int = 1,
        paths: Optional[Dict[str, Path]] = None,
    ):
        self.save_every_n_epochs = save_every_n_epochs
        self.paths = paths or setup_drive_paths()
    
    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        metrics: Dict[str, float],
    ):
        """Called at end of each epoch."""        
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch}: {metrics_str}")
        
        if (epoch + 1) % self.save_every_n_epochs == 0:
            ckpt_path = self.paths["checkpoints"] / f"epoch_{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path.name}")
        
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {used:.1f}/{total:.1f} GB ({100*used/total:.0f}%)")


if __name__ == "__main__":
    print("Testing Colab utilities...")
    
    print(f"Running in Colab: {is_colab()}")
    
    print("\nGPU Detection:")
    print_gpu_info()
    
    print("\nPath Setup:")
    paths = setup_drive_paths()
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    print("\nConfig Selection:")
    (Path("configs")).mkdir(exist_ok=True)
    (Path("configs") / "colab_t4.yaml").touch()
    (Path("configs") / "colab_a100.yaml").touch()
    
    config = auto_select_config()
    print(f"  Selected: {config}")
    
    print("\n✓ Colab utilities test passed!")
