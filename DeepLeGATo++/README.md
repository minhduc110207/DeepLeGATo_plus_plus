# DeepLeGATo++
## Next-Generation Galaxy Profile Fitting with Transformers and Neural Posterior Estimation

**Python 3.9+** | **PyTorch 2.0+** | **MIT License** | **Optimized for Google Colab**

## Overview

**DeepLeGATo++** is a significant upgrade to the original [DeepLeGATo](https://arxiv.org/abs/1711.03108) (Tuccillo et al., 2017) for galaxy surface brightness profile fitting. It leverages modern deep learning architectures optimized for Google Colab.

### Key Features

- 🔭 **Swin Transformer V2** backbone for multi-scale feature extraction
- 📊 **Neural Posterior Estimation** with full uncertainty quantification
- ☁️ **Google Colab optimized** with Drive integration and auto-resume
- ⚡ **Memory efficient** - runs on free T4 GPU (16GB VRAM)
- 🎯 **7 Sérsic parameters** with credible intervals

## Quick Start (Google Colab)

1. Upload this folder to Google Drive as `My Drive/DeepLeGATo++/`
2. Open `notebooks/03_Training.ipynb` in Google Colab
3. Run all cells - training auto-resumes on disconnect!

```python
# Mount Drive and setup
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/DeepLeGATo++')

# Install dependencies
!pip install -q torch pytorch-lightning timm nflows einops

# Start training
from deeplegato_pp.training import train
train(resume=True)
```

## Architecture

```
Galaxy Image → Swin Transformer V2 → NPE Head → Posterior Distribution
                    ↓                    ↓
              Multi-scale          Normalizing
               Features              Flows
```

## Output Parameters

| Parameter | Description | Prior Range |
|-----------|-------------|-------------|
| Magnitude | Total brightness | [15, 28] mag |
| R_eff | Effective radius | [0.1", 10"] |
| n | Sérsic index | [0.3, 8] |
| q | Axis ratio | [0.1, 1.0] |
| PA | Position angle | [0°, 180°] |
| x, y | Center offset | [±5 px] |

## GPU Requirements

| GPU | VRAM | Batch Size | Training Time |
|-----|------|------------|---------------|
| T4 (Free) | 16 GB | 16 | ~8 hours |
| A100 (Pro+) | 40 GB | 64 | ~2 hours |

## Project Structure

```
DeepLeGATo++/
├── notebooks/           # Colab notebooks
├── deeplegato_pp/       # Main package
│   ├── models/          # Swin + NPE
│   ├── data/            # Simulators & datasets
│   ├── training/        # Trainer & losses
│   └── inference/       # Predictor & UQ
├── configs/             # YAML configs
└── tests/               # Unit tests
```

## Citation

If you use DeepLeGATo++ in your research, please cite:

```bibtex
@article{deeplegato_pp_2025,
  title={DeepLeGATo++: Galaxy Profile Fitting with Transformers and Neural Posterior Estimation},
  year={2025}
}

@article{tuccillo2018deep,
  title={Deep learning for galaxy surface brightness profile fitting},
  author={Tuccillo, D. and others},
  journal={MNRAS},
  year={2018}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
