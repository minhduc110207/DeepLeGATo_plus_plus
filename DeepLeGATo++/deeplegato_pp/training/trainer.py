"""
PyTorch Lightning Trainer for DeepLeGATo++

Provides a full training pipeline with:
- Mixed precision training
- Gradient accumulation
- Checkpoint saving to Google Drive
- Automatic resume from checkpoints
- Weights & Biases logging
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import yaml

from ..models import DeepLeGAToPP
from ..data import SimulatedGalaxyDataset, create_dataloaders
from .losses import NPELoss
from .colab_utils import setup_drive_paths, get_latest_checkpoint, auto_select_config


class DeepLeGAToPPTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training DeepLeGATo++.
    
    Features:
    - Mixed precision training (FP16/BF16)
    - Gradient checkpointing for memory efficiency
    - Automatic learning rate scheduling
    - Validation with posterior predictive checks
    
    Args:
        config: Configuration dictionary
        model: Optional pre-initialized model
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[DeepLeGAToPP] = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(config)
        self.config = config
        
        if model is not None:
            self.model = model
        else:
            self.model = DeepLeGAToPP.from_config(config)
        
        self.loss_fn = NPELoss(
            nll_weight=1.0,
            flow_weight=1.0,
        )
        
        self.validation_step_outputs = []
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle checkpoint loading with potentially mismatched keys.
        
        This allows resuming from checkpoints that contain dynamically-created
        layers like _adaptive_proj in the backbone.
        """
        state_dict = checkpoint.get("state_dict", {})
        
        # Find keys that need special handling (like _adaptive_proj)
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        unexpected = ckpt_keys - model_keys
        adaptive_proj_keys = [k for k in unexpected if "_adaptive_proj" in k]
        
        if adaptive_proj_keys:
            print(f"Found adaptive projection keys in checkpoint: {adaptive_proj_keys}")
            for key in adaptive_proj_keys:
                if "weight" in key:
                    weight = state_dict[key]
                    import torch.nn as nn
                    in_features = weight.shape[1]
                    out_features = weight.shape[0]
                    self.model.backbone._adaptive_proj = nn.Linear(in_features, out_features)
                    print(f"Created _adaptive_proj layer: {in_features} -> {out_features}")
                    break
            
            if "optimizer_states" in checkpoint:
                print("Clearing optimizer state due to model structure change...")
                checkpoint["optimizer_states"] = []
        
    def forward(
        self, 
        x: torch.Tensor,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x, num_samples=num_samples)
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        params = batch["params"]
        
        loss_dict = self.model.compute_loss(images, params)
        
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        self.log("train/nll_loss", loss_dict["nll_loss"])
        self.log("train/flow_loss", loss_dict["flow_loss"])
        
        return loss_dict["loss"]
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images = batch["image"]
        params = batch["params"]
        
        loss_dict = self.model.compute_loss(images, params)
        
        with torch.no_grad():
            predictions = self.model.predict(
                images, 
                num_samples=self.config.get("validation", {}).get("num_posterior_samples", 100)
            )
        
        mean_pred = predictions["mean"]
        mae = torch.abs(mean_pred - params).mean(dim=0)
        
        output = {
            "val_loss": loss_dict["loss"],
            "mae": mae,
            "params": params,
            "mean_pred": mean_pred,
            "std_pred": predictions["std"],
        }
        
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        if not self.validation_step_outputs:
            return
        
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True)
        
        all_mae = torch.stack([x["mae"] for x in self.validation_step_outputs]).mean(dim=0)
        
        param_names = self.model.get_param_names()
        for i, name in enumerate(param_names):
            self.log(f"val/mae_{name}", all_mae[i])
        
        self.log("val/mae_mean", all_mae.mean())
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        training_config = self.config.get("training", {})
        opt_config = training_config.get("optimizer", {})
        sched_config = training_config.get("scheduler", {})
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get("learning_rate", 1e-4),
            weight_decay=opt_config.get("weight_decay", 0.05),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        )
        
        scheduler_type = sched_config.get("type", "CosineAnnealingWarmRestarts")
        
        if scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_config.get("T_0", 10),
                T_mult=sched_config.get("T_mult", 2),
                eta_min=sched_config.get("eta_min", 1e-6),
            )
        elif scheduler_type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_config.get("learning_rate", 1e-4) * 10,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if scheduler_type == "OneCycleLR" else "epoch",
            },
        }


def train(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    resume: bool = True,
    fast_dev_run: bool = False,
) -> DeepLeGAToPPTrainer:
    """
    Train DeepLeGATo++ model.
    
    Args:
        config_path: Path to config file (auto-detected if None)
        config: Configuration dictionary (overrides config_path)
        resume: Whether to resume from checkpoint
        fast_dev_run: Quick test run
        
    Returns:
        Trained model
    """
    if config is None:
        if config_path is None:
            config_path = auto_select_config()
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    try:
        paths = setup_drive_paths()
        checkpoint_dir = paths["checkpoints"]
    except Exception:
        # Not in Colab
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
    
    train_loader, val_loader = create_dataloaders(config)
    
    trainer_module = DeepLeGAToPPTrainer(config)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="deeplegato-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=config.get("training", {}).get("early_stopping_patience", 15),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    logging_config = config.get("logging", {})
    
    if logging_config.get("use_wandb", False):
        try:
            logger = WandbLogger(
                project=logging_config.get("project_name", "DeepLeGATo++"),
                log_model=True,
            )
        except Exception:
            logger = TensorBoardLogger("logs", name="deeplegato_pp")
    else:
        logger = TensorBoardLogger("logs", name="deeplegato_pp")
    
    training_config = config.get("training", {})
    
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator="auto",
        devices=1,
        precision=training_config.get("precision", "16-mixed"),
        accumulate_grad_batches=training_config.get("accumulation_steps", 1),
        callbacks=callbacks,
        logger=logger,
        val_check_interval=config.get("validation", {}).get("val_check_interval", 1.0),
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
    )
    
    ckpt_path = None
    if resume:
        ckpt_path = get_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
    
    trainer.fit(
        trainer_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
    
    return trainer_module


if __name__ == "__main__":
    print("Testing DeepLeGAToPPTrainer...")
    
    test_config = {
        "model": {
            "backbone": {
                "type": "swinv2_tiny_window8_256",
                "pretrained": False,
                "gradient_checkpointing": True,
            },
            "npe": {
                "hidden_dim": 128,
                "num_flow_layers": 2,
            },
            "num_params": 7,
        },
        "data": {
            "image_size": 64,
            "num_train_samples": 100,
            "num_val_samples": 50,
            "num_workers": 0,
            "generate_on_the_fly": True,
        },
        "training": {
            "batch_size": 4,
            "max_epochs": 2,
            "optimizer": {
                "learning_rate": 1e-4,
            },
            "precision": 32,
        },
    }
    
    model = train(config=test_config, resume=False, fast_dev_run=True)
    
    print("\n✓ Training test passed!")
