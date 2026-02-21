# DeepLeGATo++ — Complete Usage Guide
## From Input to Output: Galaxy Surface Brightness Profile Fitting

---

## Table of Contents

1. [What is DeepLeGATo++?](#what-is-deeplegato)
2. [The Pipeline at a Glance](#the-pipeline-at-a-glance)
3. [Input Specification](#input-specification)
4. [Output Specification](#output-specification)
5. [Setup & Installation (Google Colab)](#setup--installation-google-colab)
6. [Training the Model](#training-the-model)
7. [Running Inference](#running-inference)
8. [Understanding the Results](#understanding-the-results)
9. [Using Your Own Galaxy Images](#using-your-own-galaxy-images)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## What is DeepLeGATo++?

**DeepLeGATo++** is a deep learning model that measures the structural properties of galaxies from astronomical images. It is a modern upgrade to the original [DeepLeGATo](https://arxiv.org/abs/1711.03108) (Tuccillo et al., 2018).

Given a galaxy image, it predicts the **7 Sérsic profile parameters** that describe how the galaxy's light is distributed — and provides **full uncertainty quantification** using Neural Posterior Estimation (NPE).

### What makes it different from the original?

| Feature | Original DeepLeGATo | DeepLeGATo++ |
|---------|---------------------|--------------|
| **Backbone** | CNN (ResNet-like) | **Swin Transformer V2** |
| **Output** | Single point estimates | **Full posterior distributions** |
| **Uncertainty** | None | **Calibrated credible intervals** |
| **Parameters** | 4 (mag, Rₑ, n, q) | **7** (+PA, cx, cy) |
| **Data** | Pre-generated | **On-the-fly GPU simulation** |

---

## The Pipeline at a Glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   INPUT                    MODEL                         OUTPUT              │
│                                                                              │
│  Galaxy Image          ┌─────────────┐             7 Sérsic Parameters       │
│  (128×128 px,   ──────►│  Swin V2     │──►Features    + Uncertainties        │
│   1 channel,           │  Backbone    │   (768-d)                            │
│   grayscale)           └─────────────┘      │      ┌───────────────────┐     │
│                                             │      │  magnitude        │     │
│                                             ▼      │  effective_radius │     │
│                                       ┌──────────┐ │  sersic_index     │     │
│                                       │  NPE     │ │  axis_ratio       │     │
│                                       │  Head    ├►│  position_angle   │     │
│                                       │ (Norm.   │ │  center_x         │     │
│                                       │  Flows)  │ │  center_y         │     │
│                                       └──────────┘ └───────────────────┘     │
│                                                     Each with: value ± std   │
│                                                     + 95% credible interval  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Specification

### What goes into the model

| Property | Value |
|----------|-------|
| **Format** | PyTorch tensor or NumPy array |
| **Shape** | `(H, W)`, `(1, H, W)`, or `(B, 1, H, W)` |
| **Channels** | 1 (grayscale) |
| **Image Size** | 128×128 pixels (T4 config) or 256×256 pixels (L4/A100 config) |
| **Data Type** | `float32` |
| **Pixel Values** | Flux values (not normalized — the model handles scaling internally) |

### Input preprocessing (handled automatically by the Predictor)

The `Predictor` class automatically handles these for you:

1. **NumPy → Tensor**: Converts numpy arrays to PyTorch tensors
2. **Dimension expansion**: A raw 2D image `(H, W)` is reshaped to `(1, 1, H, W)`
3. **Device transfer**: Moves the tensor to GPU/CPU as appropriate

So you can simply pass in a 2D array:

```python
import numpy as np

# Any of these work:
image = np.random.randn(128, 128)                    # (H, W) numpy
image = torch.randn(128, 128)                         # (H, W) tensor
image = torch.randn(1, 128, 128)                      # (C, H, W) tensor
image = torch.randn(4, 1, 128, 128)                   # (B, C, H, W) batch
```

### About the training data

DeepLeGATo++ trains on **simulated galaxy images** generated on-the-fly. No pre-made dataset needed. The simulator (`SersicSimulator`) creates synthetic Sérsic profile galaxies with:

- Random parameters drawn from configurable prior ranges
- Realistic noise (Poisson shot noise + Gaussian read noise)
- Optional PSF (Point Spread Function) convolution

---

## Output Specification

### The 7 Sérsic Parameters

| # | Parameter | Symbol | Description | Prior Range | Unit |
|---|-----------|--------|-------------|-------------|------|
| 1 | **Magnitude** | mag | Total apparent brightness | 15.0 – 28.0 | mag |
| 2 | **Effective Radius** | Rₑ | Radius enclosing half the total light | 0.1 – 10.0 | arcsec |
| 3 | **Sérsic Index** | n | Shape of the light profile | 0.3 – 8.0 | — |
| 4 | **Axis Ratio** | q | How round the galaxy is (1 = circular) | 0.1 – 1.0 | — |
| 5 | **Position Angle** | PA | Orientation of the major axis | 0 – 180 | degrees |
| 6 | **Center X** | cx | Horizontal offset from image center | -5.0 – 5.0 | pixels |
| 7 | **Center Y** | cy | Vertical offset from image center | -5.0 – 5.0 | pixels |

### Understanding the Sérsic Index

The Sérsic index `n` controls the **shape** of the light profile:

- **n = 0.5** → Gaussian profile
- **n = 1** → Exponential disk (typical spiral galaxy)
- **n = 4** → de Vaucouleurs profile (typical elliptical galaxy)
- **n > 4** → Very concentrated light profile (cD galaxies)

### What the output looks like

```python
result = predictor.predict(galaxy_image, num_samples=1000, return_samples=True)
```

The result dictionary contains:

```python
{
    "params": {
        "magnitude":        {"value": 21.34, "std": 0.12, "unit": "mag",
                             "ci_lower": 21.10, "ci_upper": 21.58},
        "effective_radius": {"value": 2.51,  "std": 0.28, "unit": "arcsec",
                             "ci_lower": 1.96, "ci_upper": 3.06},
        "sersic_index":     {"value": 3.72,  "std": 0.45, "unit": "",
                             "ci_lower": 2.84, "ci_upper": 4.60},
        "axis_ratio":       {"value": 0.68,  "std": 0.05, "unit": "",
                             "ci_lower": 0.58, "ci_upper": 0.78},
        "position_angle":   {"value": 42.3,  "std": 5.2,  "unit": "deg",
                             "ci_lower": 32.1, "ci_upper": 52.5},
        "center_x":         {"value": 0.12,  "std": 0.08, "unit": "px",
                             "ci_lower": -0.04, "ci_upper": 0.28},
        "center_y":         {"value": -0.23, "std": 0.09, "unit": "px",
                             "ci_lower": -0.41, "ci_upper": -0.05},
    },
    "mean":     np.array([21.34, 2.51, 3.72, 0.68, 42.3, 0.12, -0.23]),    # (7,)
    "std":      np.array([0.12, 0.28, 0.45, 0.05, 5.2, 0.08, 0.09]),        # (7,)
    "ci_lower": np.array([21.10, 1.96, 2.84, 0.58, 32.1, -0.04, -0.41]),    # (7,)
    "ci_upper": np.array([21.58, 3.06, 4.60, 0.78, 52.5, 0.28, -0.05]),     # (7,)
    "samples":  np.array(...)   # shape (1, 1000, 7) — 1000 posterior samples
}
```

**Accessing individual parameters:**

```python
# Get the Sérsic index estimate
n = result["params"]["sersic_index"]["value"]       # 3.72
n_std = result["params"]["sersic_index"]["std"]     # 0.45
n_lower = result["params"]["sersic_index"]["ci_lower"]  # 2.84
n_upper = result["params"]["sersic_index"]["ci_upper"]  # 4.60

print(f"Sérsic index: {n:.2f} ± {n_std:.2f} (95% CI: [{n_lower:.2f}, {n_upper:.2f}])")
# Output: Sérsic index: 3.72 ± 0.45 (95% CI: [2.84, 4.60])
```

---

## Setup & Installation (Google Colab)

### Step 1: Upload project to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder: `My Drive/DeepLeGATo++/`
3. Upload the following folders into it:
   - `deeplegato_pp/` (the Python package)
   - `configs/` (YAML configuration files)
   - `notebooks/` (Colab notebooks)
   - `requirements_colab.txt`

Your Drive should look like:
```
My Drive/
└── DeepLeGATo++/
    ├── deeplegato_pp/
    │   ├── models/
    │   ├── data/
    │   ├── training/
    │   └── inference/
    ├── configs/
    │   ├── colab_t4.yaml
    │   ├── colab_l4.yaml
    │   └── colab_a100.yaml
    ├── notebooks/
    │   └── 03_Training.ipynb
    └── requirements_colab.txt
```

### Step 2: Open in Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. **File → Open notebook → Google Drive**
3. Navigate to `DeepLeGATo++/notebooks/03_Training.ipynb`

### Step 3: Enable GPU

1. **Runtime → Change runtime type**
2. Set **Hardware accelerator** to:
   - **T4 GPU** (free tier — 16GB VRAM)
   - **L4 GPU** (paid — 24GB VRAM)
   - **A100 GPU** (Pro+ — 40GB VRAM)
3. Click **Save**

### Step 4: Mount Drive & Install

Run these cells in the notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Copy project to local Colab filesystem for faster I/O
import os, sys

PROJECT_NAME = "DeepLeGATo++"
DRIVE_PATH = f"/content/drive/MyDrive/{PROJECT_NAME}"

if os.path.exists(f"{DRIVE_PATH}/deeplegato_pp"):
    !cp -r "{DRIVE_PATH}/deeplegato_pp" /content/
    !cp -r "{DRIVE_PATH}/configs" /content/
    sys.path.insert(0, '/content')
    print("✅ Project loaded!")
else:
    print("❌ Project not found on Drive!")
```

```python
# Install dependencies
!pip install -q torch>=2.1.0 pytorch-lightning>=2.1.0 timm>=0.9.12
!pip install -q nflows zuko astropy photutils
!pip install -q wandb gradio plotly seaborn einops
print("✅ Dependencies installed!")
```

```python
# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU! Go to Runtime > Change runtime type > GPU")
```

---

## Training the Model

### Quick Start (3 lines)

```python
from deeplegato_pp.training import train

# This auto-detects GPU, selects config, creates data, and starts training
trained_model = train(resume=True)
```

### Step-by-Step Training

#### 1. Load Configuration

The configuration is **automatically selected** based on your GPU:

| GPU | Config File | Backbone | Image Size | Batch Size | ~Training Time |
|-----|-------------|----------|------------|------------|----------------|
| T4 (free) | `colab_t4.yaml` | `swinv2_tiny` | 128×128 | 16 × 4 = 64 | ~8 hours |
| L4 (paid) | `colab_l4.yaml` | `swinv2_base` | 256×256 | 32 × 2 = 64 | ~5 hours |
| A100 (Pro+) | `colab_a100.yaml` | `swinv2_base` | 256×256 | 64 | ~2 hours |

```python
import yaml
from deeplegato_pp.training.colab_utils import auto_select_config

# Auto-detect GPU and pick the right config
config_path = auto_select_config("/content/configs")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Using config: {config_path}")
print(f"Backbone: {config['model']['backbone']['type']}")
print(f"Image size: {config['data']['image_size']}")
```

#### 2. Create Model

```python
from deeplegato_pp.models import DeepLeGAToPP

model = DeepLeGAToPP.from_config(config)

# Check model size
param_counts = model.count_parameters()
print(f"Total parameters: {param_counts['total']:,}")
print(f"  Backbone: {param_counts['backbone']:,}")
print(f"  NPE Head: {param_counts['npe_head']:,}")
```

Expected output (T4):
```
Total parameters: 32,600,000
  Backbone: 27,500,000
  NPE Head: 5,100,000
```

#### 3. Create Data Loaders

```python
from deeplegato_pp.data import create_dataloaders

train_loader, val_loader = create_dataloaders(config)
print(f"Training samples: {len(train_loader.dataset):,}")        # 100,000
print(f"Validation samples: {len(val_loader.dataset):,}")        # 10,000
print(f"Batches per epoch: {len(train_loader):,}")                # 6,250
```

> **Note**: Data is generated on-the-fly by `SimulatedGalaxyDataset`. Each time a sample is requested, a random Sérsic profile is created with random parameters and realistic noise. No pre-downloaded dataset required.

#### 4. Start Training

```python
from deeplegato_pp.training import train

trained_model = train(
    config=config,
    resume=True,       # Auto-resume from checkpoint if available
    fast_dev_run=False, # Set True for a quick test (1 batch only)
)
```

**What happens during training:**
- The model sees batches of simulated galaxy images
- For each image, it predicts the posterior distribution over Sérsic parameters
- The loss function (negative log-likelihood + flow loss) trains the model to produce accurate posteriors
- Checkpoints are saved to Google Drive **every epoch**
- Training automatically resumes if Colab disconnects

#### 5. Handle Colab Disconnections

If Colab disconnects mid-training:

1. Reconnect to Colab
2. Re-run Steps 4 (Mount Drive), install dependencies, and setup path
3. Run `train(resume=True)` again — **training resumes from the last checkpoint automatically**

Checkpoints are stored at:
```
/content/drive/MyDrive/DeepLeGATo++/checkpoints/last.ckpt
```

---

## Running Inference

### After Training

```python
from deeplegato_pp.inference import Predictor

# Create predictor from trained model
predictor = Predictor(trained_model.model, device='cuda')

# Get a test image from the validation set
val_batch = next(iter(val_loader))
test_image = val_batch['image'][0]       # Shape: (1, 128, 128)
true_params = val_batch['params'][0]     # Shape: (7,)

# Predict — this is the core inference call
result = predictor.predict(
    test_image,
    num_samples=1000,        # Number of posterior samples (more = better uncertainty estimates)
    return_samples=True,     # Keep the raw samples for analysis
)

# Print formatted results
predictor.print_results(result)
```

**Output:**
```
============================================================
DeepLeGATo++ Prediction Results
============================================================
magnitude           :   21.340 ± 0.120 mag     [95% CI: 21.100 - 21.580]
effective_radius    :    2.510 ± 0.280 arcsec   [95% CI: 1.960 - 3.060]
sersic_index        :    3.720 ± 0.450          [95% CI: 2.840 - 4.600]
axis_ratio          :    0.680 ± 0.050          [95% CI: 0.580 - 0.780]
position_angle      :   42.300 ± 5.200 deg      [95% CI: 32.100 - 52.500]
center_x            :    0.120 ± 0.080 px       [95% CI: -0.040 - 0.280]
center_y            :   -0.230 ± 0.090 px       [95% CI: -0.410 - -0.050]
============================================================
```

### Visualize Posterior Distributions

```python
# Plot the posterior distributions for all 7 parameters
predictor.plot_posterior(
    result,
    save_path="posterior_distributions.png"
)
```

### Loading a Previously Saved Model

```python
from deeplegato_pp.inference import Predictor

# Load from saved model directory
predictor = Predictor.from_pretrained(
    "/content/drive/MyDrive/DeepLeGATo++/models/final_model",
    device="cuda"
)

# Now use predictor.predict() as shown above
```

### Batch Prediction (Multiple Galaxies)

```python
# Predict on many images at once
images = torch.randn(100, 1, 128, 128)  # 100 galaxy images

results = predictor.predict_batch(
    images,
    num_samples=100,     # Fewer samples for speed on large batches
    batch_size=32,       # Process 32 images at a time
)
```

### Saving the Trained Model

```python
# Save model weights + config for later use
model.save_pretrained("/content/drive/MyDrive/DeepLeGATo++/models/final_model")
```

This saves:
- `model.pt` — Model weights
- `config.yaml` — Model configuration (backbone type, dimensions, etc.)

---

## Understanding the Results

### What the Uncertainties Mean

Each parameter comes with:

| Output | What it means |
|--------|---------------|
| **value** | Best estimate (posterior mean) |
| **std** | Standard deviation of the posterior — how uncertain the model is |
| **ci_lower / ci_upper** | 95% credible interval — the true value has a 95% probability of being in this range |

**Example interpretation:**

```
effective_radius: 2.51 ± 0.28 arcsec [95% CI: 1.96 - 3.06]
```

This means:
- The model's best estimate for the effective radius is **2.51 arcseconds**
- The uncertainty is **±0.28 arcseconds**
- There is a **95% probability** that the true effective radius is between **1.96 and 3.06 arcseconds**

### When uncertainties are large

Large uncertainties indicate:
- The galaxy image is noisy or faint
- The galaxy is poorly resolved (too small)
- The Sérsic model may not be a good fit (e.g., merging galaxies)
- The model is unsure — this is a feature, not a bug!

### Posterior samples

If you called `predict(..., return_samples=True)`, you get the raw posterior samples:

```python
samples = result["samples"]  # shape: (1, 1000, 7)

# Plot the joint posterior for Sérsic index vs effective radius
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.scatter(samples[0, :, 2], samples[0, :, 1], alpha=0.1, s=2)
plt.xlabel("Sérsic Index (n)")
plt.ylabel("Effective Radius (Rₑ)")
plt.title("Joint Posterior: n vs Rₑ")
plt.show()
```

---

## Using Your Own Galaxy Images

### From a FITS file

```python
from astropy.io import fits
import torch

# Load FITS file
hdu = fits.open("my_galaxy.fits")
image_data = hdu[0].data.astype("float32")  # shape: (H, W)

# Crop/resize to expected size (128×128 for T4 config)
from torchvision.transforms.functional import resize
image_tensor = torch.from_numpy(image_data).float()
image_tensor = resize(image_tensor.unsqueeze(0), [128, 128])  # (1, 128, 128)

# Predict
result = predictor.predict(image_tensor, num_samples=1000)
predictor.print_results(result)
```

### From a NumPy array

```python
import numpy as np

# Load your image (any source)
image = np.load("galaxy_cutout.npy")  # shape: (128, 128)

# The Predictor handles numpy arrays directly
result = predictor.predict(image, num_samples=1000)
```

### From a batch of files

```python
from pathlib import Path
from astropy.io import fits
import torch

galaxy_dir = Path("/path/to/galaxy/cutouts")
images = []

for fits_file in sorted(galaxy_dir.glob("*.fits")):
    data = fits.getdata(fits_file).astype("float32")
    tensor = torch.from_numpy(data).unsqueeze(0)  # (1, H, W)
    images.append(tensor)

# Stack into batch
batch = torch.stack(images)  # (N, 1, H, W)

# Predict all at once
results = predictor.predict_batch(batch, num_samples=100, batch_size=32)
```

### Important notes for real images

1. **Image size must match training config**: 128×128 for T4, 256×256 for L4/A100
2. **Single channel**: The model expects grayscale images. If you have multi-band data, use one band at a time
3. **Pixel values**: Use raw flux counts or calibrated surface brightness — avoid taking the log
4. **Centering**: The galaxy should be approximately centered in the image
5. **Background**: Some sky background is expected (the noise model accounts for it)

---

## Configuration Reference

### Config File Structure

```yaml
model:
  name: "DeepLeGATo++"

  backbone:
    type: "swinv2_tiny_window8_256"   # timm model name
    pretrained: true                   # Use ImageNet pre-trained weights
    embed_dim: 96                      # Embedding dimension
    depths: [2, 2, 6, 2]              # Transformer blocks per stage
    num_heads: [3, 6, 12, 24]         # Attention heads per stage
    window_size: 8                     # Attention window size
    gradient_checkpointing: true       # Trade compute for memory
    drop_path_rate: 0.1                # Stochastic depth

  npe:
    num_flow_layers: 8                 # Normalizing flow depth
    hidden_dim: 256                    # Flow hidden dimension
    num_transforms: 5                  # Number of MAF transforms
    num_blocks: 2                      # Blocks per transform
    dropout: 0.1

  num_params: 7                        # Output parameters

training:
  batch_size: 16                       # Per-GPU batch size
  accumulation_steps: 4                # Gradient accumulation (effective = 64)
  max_epochs: 100
  precision: "16-mixed"                # FP16 mixed precision

  optimizer:
    type: "AdamW"
    learning_rate: 1.0e-4
    weight_decay: 0.05

  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 10                            # First restart period
    T_mult: 2                          # Restart period multiplier

  checkpoint_every_n_epochs: 1
  save_to_drive: true
  resume_from_latest: true
  early_stopping_patience: 15          # Stop if no improvement

data:
  image_size: 128                      # Input resolution
  num_workers: 2
  num_train_samples: 100000
  num_val_samples: 10000
  generate_on_the_fly: true            # Simulate data during training

  priors:                              # Parameter sampling ranges
    magnitude: [15.0, 28.0]
    effective_radius: [0.1, 10.0]
    sersic_index: [0.3, 8.0]
    axis_ratio: [0.1, 1.0]
    position_angle: [0.0, 180.0]
    center_offset: [-5.0, 5.0]

  noise:
    sky_background: 0.01
    read_noise: 5.0
    gain: 2.5

logging:
  use_wandb: true
  project_name: "DeepLeGATo++"
  log_every_n_steps: 50
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| **No GPU available** | Runtime type not set | Runtime → Change runtime type → GPU |
| **Out of memory (OOM)** | Batch size too large | Reduce `batch_size` in config, keep `accumulation_steps` high |
| **`ModuleNotFoundError`** | Path not set | Run `sys.path.insert(0, '/content')` |
| **Stale code after edits** | Colab caches imports | Restart runtime (Runtime → Restart runtime) |
| **Session disconnected** | Colab timeout | Reconnect and re-run → training resumes from checkpoint |
| **`RuntimeError: shape ...`** | `timm` version issue | Add `dynamic_img_size=True` to `timm.create_model()` in `swin_backbone.py` |
| **Slow training** | No gradient accumulation | Ensure `accumulation_steps` matches config |
| **W&B login prompt** | Weights & Biases tracking | Enter your API key, or set `use_wandb: false` in config |

### Restarting from scratch

If you want to start training completely fresh (discard all checkpoints):

```python
# Delete existing checkpoints
import shutil
shutil.rmtree("/content/drive/MyDrive/DeepLeGATo++/checkpoints", ignore_errors=True)

# Train from scratch
trained_model = train(config=config, resume=False)
```

---

## File Structure

```
DeepLeGATo++/
│
├── configs/                         # Configuration files
│   ├── colab_t4.yaml                # Free tier T4 GPU (16 GB)
│   ├── colab_l4.yaml                # L4 GPU (24 GB)
│   └── colab_a100.yaml              # A100 GPU (40 GB)
│
├── deeplegato_pp/                   # Main Python package
│   │
│   ├── models/                      # Neural network modules
│   │   ├── deeplegato_pp.py         # Main model (backbone + NPE head)
│   │   ├── swin_backbone.py         # Swin Transformer V2 feature extractor
│   │   └── npe_head.py              # Normalizing Flow posterior estimator
│   │
│   ├── data/                        # Data generation & loading
│   │   ├── sersic_simulator.py      # GPU-accelerated galaxy simulator
│   │   ├── dataset.py               # PyTorch datasets
│   │   └── psf_handler.py           # Point Spread Function handling
│   │
│   ├── training/                    # Training pipeline
│   │   ├── trainer.py               # PyTorch Lightning trainer + train()
│   │   ├── losses.py                # NPE loss functions
│   │   └── colab_utils.py           # Colab/Drive helpers
│   │
│   └── inference/                   # Inference pipeline
│       ├── predictor.py             # High-level prediction API
│       └── uncertainty.py           # Uncertainty analysis utilities
│
├── notebooks/
│   └── 03_Training.ipynb            # Main Colab training notebook
│
├── requirements_colab.txt           # Python dependencies
├── README.md                        # Quick start
├── TRAINING_GUIDE.md                # Step-by-step training instructions
├── ARCHITECTURE_GUIDE.md            # Technical architecture details
└── USER_GUIDE.md                    # This file
```

---

## References

1. **Original DeepLeGATo**: Tuccillo et al. 2018, MNRAS — [arXiv:1711.03108](https://arxiv.org/abs/1711.03108)
2. **Swin Transformer V2**: Liu et al. 2022 — [arXiv:2111.09883](https://arxiv.org/abs/2111.09883)
3. **Neural Posterior Estimation**: Cranmer et al. 2020 — [arXiv:1911.01429](https://arxiv.org/abs/1911.01429)
4. **Sérsic Profile**: Sérsic 1963, 1968 — Fundamental galaxy light distribution model
5. **Normalizing Flows**: Papamakarios et al. 2021 — [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
