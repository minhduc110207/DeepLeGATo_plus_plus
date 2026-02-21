# DeepLeGATo++ Training Guide
## Complete Step-by-Step Instructions for Google Colab

---

## Prerequisites

- A Google account (for Colab and Drive access)
- The `DeepLeGATo++` project folder

---

## Step 1: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `My Drive/DeepLeGATo++/`
3. Upload the entire project contents:
   - `configs/` folder
   - `deeplegato_pp/` folder
   - `notebooks/` folder
   - `requirements_colab.txt`

---

## Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Open notebook**
3. Select **Google Drive** tab
4. Navigate to `DeepLeGATo++/notebooks/03_Training.ipynb`

---

## Step 3: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU** (free) or **A100** (Pro+)
3. Click **Save**

Verify GPU:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 5: Install Dependencies

```python
!pip install -q torch>=2.1.0 pytorch-lightning>=2.1.0 timm>=0.9.12
!pip install -q nflows zuko einops
!pip install -q astropy photutils
```

---

## Step 6: Setup Project Path

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/DeepLeGATo++')

# Verify
from deeplegato_pp.models import DeepLeGAToPP
print("✅ Import successful!")
```

---

## Step 7: Load Configuration

```python
import yaml
from deeplegato_pp.training.colab_utils import auto_select_config

config_path = auto_select_config('/content/drive/MyDrive/DeepLeGATo++/configs')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Config: {config_path}")
print(f"Backbone: {config['model']['backbone']['type']}")
# Expected: swinv2_tiny_window8_256 (T4) or swinv2_base_window8_256 (A100)
```

---

## Step 8: Create Model

```python
from deeplegato_pp.models import DeepLeGAToPP

model = DeepLeGAToPP.from_config(config)
params = model.count_parameters()
print(f"Total parameters: {params['total']:,}")
```

---

## Step 9: Create Data Loaders

```python
from deeplegato_pp.data import create_dataloaders

train_loader, val_loader = create_dataloaders(config)
print(f"Training batches: {len(train_loader)}")
```

---

## Step 10: Start Training

```python
from deeplegato_pp.training import train

# Training with auto-resume
trained_model = train(config=config, resume=True)
```

**Training Time:**
| GPU | Time |
|-----|------|
| T4 | ~8-10 hours |
| A100 | ~2-3 hours |

---

## Step 11: Handle Disconnections

If Colab disconnects:
1. Reconnect to Colab
2. Re-run Steps 4-6 (mount, install, setup path)
3. Run Step 10 again - **training resumes automatically!**

Checkpoints are saved to Drive every epoch.

---

## Step 12: Evaluate Model

```python
from deeplegato_pp.inference import Predictor

predictor = Predictor(trained_model.model, device='cuda')

# Test on sample
test_batch = next(iter(val_loader))
result = predictor.predict(test_batch['image'][0:1].cuda(), num_samples=1000)
predictor.print_results(result)
```

---

## Step 13: Save Final Model

```python
from deeplegato_pp.training.colab_utils import setup_drive_paths

paths = setup_drive_paths()
trained_model.model.save_pretrained(paths['models'] / 'final_model')
print(f"✅ Saved to: {paths['models'] / 'final_model'}")
```

---

## Quick Reference

### Minimal Training Code:
```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/DeepLeGATo++')

!pip install -q torch pytorch-lightning timm nflows einops

from deeplegato_pp.training import train
model = train(resume=True)
```

### Model Names:
- **T4 (16GB)**: `swinv2_tiny_window8_256`
- **A100 (40GB)**: `swinv2_base_window8_256`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU | Runtime → Change runtime type → GPU |
| Out of memory | Reduce `batch_size` in config |
| Session disconnected | Reconnect and re-run - training resumes |
| Module not found | Check `sys.path` includes project folder |

---

## Output Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| magnitude | Total brightness | 15-28 mag |
| effective_radius | Half-light radius | 0.1-10" |
| sersic_index | Profile shape | 0.3-8 |
| axis_ratio | Ellipticity | 0.1-1.0 |
| position_angle | Orientation | 0-180° |
| center_x, center_y | Position offset | ±5 px |
