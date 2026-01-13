# Item Response Scaling Laws

[![Paper](https://img.shields.io/badge/paper-ICML%202026-blue)](./67ff391e1fb5f3e66c8a79c9/main.tex)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> **Item Response Scaling Laws: A Measurement Theory Approach to Generalizable Neural Performance Prediction**

This repository contains the code for our ICML 2026 submission applying Item Response Theory (IRT) from educational testing to model LLM scaling behavior.

## Overview

Classical neural scaling laws describe how LLM performance improves with increased compute, but they typically treat benchmarks as homogeneous and overlook individual question characteristics. We propose **Item Response Scaling Laws**, which leverage IRT to model:

1. **Pre-training Downstream Scaling**: How model performance on downstream tasks improves as a function of computational effort (FLOPs) during pre-training
2. **Test-time Scaling**: How inference performance grows with the number of independent samples at test time

By modeling per-question characteristics (difficulty, discrimination), our approach produces scaling laws that are:
- **Interpretable**: Question difficulties and model abilities have clear meanings
- **Generalizable**: Transfer across questions with different difficulty levels
- **Reliable**: More stable estimates with lower variance than aggregate metrics

### Key Contributions

- **Binary-IRT**: Apply IRT with Bernoulli loss to binary response matrices
- **Beta-IRT**: Extend IRT with Beta loss to model empirical probability responses
- **Large-scale validation**:
  - Pre-training: 25 models from 6 families, up to 359 checkpoints, 15 NLP datasets
  - Test-time: 15 models, 10 NLP datasets, up to 10,000 samples per question

---

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── utils.py                              # Core IRT implementations (calibration, CAT, losses)
│
├── 67ff391e1fb5f3e66c8a79c9/             # LaTeX manuscript (ICML 2026 submission)
│   ├── main.tex                          # Main document
│   ├── sections/                         # Paper sections (abstract, intro, method, etc.)
│   └── figures/                          # Paper figures
│
├── downstream/                           # Pre-training downstream scaling experiments
│   ├── pretrain_cat_helm_binary.py      # Binary-IRT CAT (Figure 1)
│   ├── pretrain_metric_tv_binary.py     # Total variation metrics
│   ├── pretrain_organize_and_plotnaivelaw_resmat2.py  # Naive scaling laws (Figure 2)
│   ├── pretrain_cat_resmat2.py          # Beta-IRT CAT (Figure 3)
│   ├── pretrain_metric_resmat2.py       # Beta-IRT metrics
│   ├── downstream_scaleup.py            # Generalization experiments (Figures 4, 7)
│   ├── downstream_scaleup_plot.py       # Plot generation
│   └── downstream_scaleup_plot_heatmap.py  # Heatmap visualization
│
├── monkey/                               # Test-time scaling experiments
│   ├── monkey_query/                    # Query LLMs with repeated samples (initial version)
│   ├── monkey_query_2/                  # Query LLMs (updated version)
│   ├── monkey_gather_data/              # Collect response data (initial version)
│   ├── monkey_gather_data_2/            # Collect response data (updated version)
│   └── monkey_analysis/                 # Analysis and plotting
│       ├── testtime_calibrate.py        # Calibrate item parameters
│       ├── testtime_cat.py              # Run CAT on test-time data
│       ├── testtime_aggregate_results.py  # Generate Figures 5, 8, 9, 10
│       └── monkey_analysis_denoise_data.py  # Denoising analysis (Figure 6)
│
├── ai2/                                  # Allen AI DataDecide integration
│   ├── DataDecide/                      # Full DataDecide project (25 corpora, 14 model sizes)
│   │   ├── eval/                        # Evaluation scripts for checkpoints
│   │   ├── pretraining/                 # Training configurations
│   │   ├── release/                     # Model release utilities
│   │   ├── scaling_laws/                # Scaling law analysis notebooks
│   │   ├── viz/                         # Visualization notebooks
│   │   └── README.md                    # DataDecide documentation
│   └── clean_datadecide/                # Data preprocessing pipeline
│       ├── 1_download_and_unzip.py     # Download raw data
│       ├── 2_get_long_table.py         # Convert to long format
│       ├── 3_clean_and_pivot.py        # Clean and pivot data
│       ├── 4_calibrate.py              # Calibrate IRT parameters
│       └── 5_cat.py                    # Run CAT
│
├── gather_ckpt_data/                    # Checkpoint evaluation data collection
│   ├── pretrain_helm_json2csv.py       # Convert HELM JSON outputs to CSV
│   ├── pretrain_helm_csv2matrix.py     # Aggregate into response matrices
│   └── ...
│
└── gather_helm_data/                    # HELM benchmark data collection
    ├── json2csv.py                      # Convert HELM JSON to CSV
    ├── csv2matrix.py                    # Create response matrices
    ├── irt_calibrate.py                 # Calibrate IRT on HELM data
    └── ...
```

### Note on Directory Names
- `67ff391e1fb5f3e66c8a79c9/`: Auto-generated directory name (likely from Overleaf) containing the LaTeX paper
- `monkey/`: Named after "monkey testing" or exhaustive sampling with repeated queries

---

## Installation

### Prerequisites

⚠️ **Important Requirements:**
- **Python 3.10** (Required - Python 3.11+ has dependency compatibility issues)
- **LaTeX** (Required for plot generation with matplotlib)
- **CUDA** (Optional but recommended for faster computation)

### System Dependencies

LaTeX is required for rendering plots with mathematical notation. Install before proceeding:

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra cm-super dvipng
```

#### macOS:
```bash
brew install --cask mactex
# Or for a minimal installation:
brew install basictex
```

#### Windows:
Install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/)

### Python Environment Setup

```bash
# Create conda environment with Python 3.10
conda create --name deval python=3.10 -y
conda activate deval

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

Use the provided validation script to check if everything is set up correctly:

```bash
python scripts/check_environment.py
```

This will check:
- Python version (should be 3.10.x)
- LaTeX installation
- Required Python packages
- PyTorch and CUDA availability
- HuggingFace Hub access
- Available disk space

If all checks pass, you're ready to run experiments!

### Dependencies

Key packages include:
- `torch>=2.4.0` - PyTorch for model training and IRT optimization
- `transformers>=4.30.0` - Hugging Face transformers for LLM inference
- `scikit-learn>=1.3.0` - Machine learning utilities
- `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `tueplots>=0.2.0` - Visualization
- `statsmodels>=0.14.0` - Statistical modeling
- `scipy>=1.10.0` - Scientific computing (for Spearman correlation, etc.)

Optional packages for inference acceleration (requires CUDA):
- `flash-attn>=2.5.0` - Fast attention implementation
- `vllm>=0.6.3` - Fast LLM inference

See [requirements.txt](requirements.txt) for complete list.

---

## Quick Start

### 1. Understanding the Core IRT Implementation

The main IRT functions are in [utils.py](utils.py):

- **`calibrate(probmat, ...)`**: Calibrate item difficulty parameters from probability response matrix
- **`cat_binary_1pl(ys, zs, ...)`**: Run Binary-IRT CAT (Computerized Adaptive Testing)
- **`beta_nll(...)`**: Beta negative log-likelihood loss for probability responses
- **IRT models**: Rasch (1PL), 2PL models with sigmoid link functions

### 2. Pre-training Downstream Scaling

To reproduce the pre-training scaling experiments:

```bash
cd downstream

# Binary-IRT experiments (Figure 1)
python pretrain_cat_helm_binary.py

# Naive probability scaling laws (Figure 2)
python pretrain_organize_and_plotnaivelaw_resmat2.py

# Beta-IRT experiments (Figure 3)
python pretrain_cat_resmat2.py
python pretrain_metric_resmat2.py

# Generalization experiments (Figures 4, 7)
python downstream_scaleup.py
python downstream_scaleup_plot.py
python downstream_scaleup_plot_heatmap.py
```

### 3. Test-time Scaling

To reproduce the test-time scaling experiments:

```bash
cd monkey/monkey_analysis

# Run the complete pipeline (Figures 5, 8, 9, 10)
python testtime_calibrate.py      # Step 1: Calibrate item parameters
python testtime_cat.py             # Step 2: Run CAT
python testtime_aggregate_results.py  # Step 3: Aggregate and plot

# Denoising analysis (Figure 6)
python monkey_analysis_denoise_data.py
```

---

## Reproducing Paper Figures

### Pre-training Downstream Scaling

| Figure | Description | Script(s) |
|--------|-------------|-----------|
| Figure 1 | Binary-IRT vs Accuracy curves | `downstream/pretrain_cat_helm_binary.py` |
| Figure 2 | Naive scaling behavior of p(Correct Choice) | `downstream/pretrain_organize_and_plotnaivelaw_resmat2.py` |
| Figure 3 | Beta-IRT CAT scaling curves | `downstream/pretrain_cat_resmat2.py` → `pretrain_metric_resmat2.py` |
| Figure 4, 7 | Generalization across datasets | `downstream/downstream_scaleup.py` → `downstream_scaleup_plot.py` → `downstream_scaleup_plot_heatmap.py` |
| Total Variation | Evaluation stability metrics | `downstream/pretrain_metric_tv_binary.py` |

### Test-time Scaling

| Figure | Description | Script(s) |
|--------|-------------|-----------|
| Figure 5, 8, 9, 10 | Test-time scaling laws | `monkey/monkey_analysis/testtime_calibrate.py` → `testtime_cat.py` → `testtime_aggregate_results.py` |
| Figure 6 | Denoising analysis | `monkey/monkey_analysis/monkey_analysis_denoise_data.py` |

---

## Data

### Pre-training Experiments

The experiments use response matrices from:
1. **HELM binary responses**: 13 models from 3 families (Pythia, Amber, SmolLM) with up to 359 checkpoints on 13 datasets
2. **Probability responses**: 5 model families on 12 multiple-choice datasets (from Schaeffer et al. 2024)

Pre-calibrated item difficulties are obtained from:
- HELM leaderboard (42-91 diverse LLMs)
- Open LLM Leaderboard (102 diverse LLMs)

Data is hosted on Hugging Face: `stair-lab/irsl_downstream_resmat`

### Test-time Experiments

- 15 models across 10 popular NLP benchmarks
- Up to 10,000 repeated samples per question
- Datasets include MMLU, HellaSwag, ARC, PIQA, etc.

### DataDecide Integration

The `ai2/DataDecide/` directory integrates with the DataDecide project:
- 25 pretraining corpora (Dolma, C4, FineWeb, Falcon, DCLM, etc.)
- 14 model sizes (4M to 1B parameters)
- 30k+ model checkpoints
- Evaluated on OLMES suite of 10 benchmarks

See [ai2/DataDecide/README.md](ai2/DataDecide/README.md) for more details.

---

## Key Concepts

### Item Response Theory (IRT)

IRT models the interaction between test-takers (LLMs) and questions:

**Rasch Model (1PL)**:
```
P(correct | θ, z) = σ(θ - z)
```
- `θ` (theta): Model ability (higher = more capable)
- `z`: Question difficulty (higher = more difficult)
- `σ`: Sigmoid function

**2PL Model**:
```
P(correct | θ, z, a) = σ(a(θ - z))
```
- `a`: Discrimination parameter (how well the question differentiates abilities)

### Binary-IRT vs Beta-IRT

- **Binary-IRT**: Uses Bernoulli loss on binary responses (correct/incorrect)
  - Good for: Traditional evaluation with single-pass inference
  - Limitation: Binary responses can show emergent behaviors

- **Beta-IRT**: Uses Beta loss on empirical probability responses
  - Good for: Modeling p(Correct Choice) from model logits or repeated sampling
  - Advantage: Smoother scaling curves, less emergence, better predictability

### Computerized Adaptive Testing (CAT)

CAT efficiently estimates model ability with fewer questions:
1. Select most informative question (highest Fisher information)
2. Query model and observe response
3. Update ability estimate
4. Repeat until budget exhausted

---

## Model Families

### Pre-training Experiments

| Family | Sizes | Checkpoints | Source |
|--------|-------|-------------|--------|
| Pythia | 14M - 12B | 143 | EleutherAI |
| Amber | 6.7B | 358 | LLM360 |
| SmolLM2 | 135M - 1.7B | 125-240 | HuggingFace |

### Test-time Experiments

15 models across various families (see paper for complete list)

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{truong2026irsl,
    title={Item Response Scaling Laws: A Measurement Theory Approach to Generalizable Neural Performance Prediction},
    author={[Authors TBD]},
    year={2026},
    journal={ICML},
}
```

---

## Project Status

This is research code for an ICML 2026 submission. The repository contains:
- ✅ Complete experiment scripts for reproducing all paper figures
- ✅ Core IRT implementation in utils.py
- ✅ Integration with DataDecide checkpoint suite
- ⚠️ Data files not included (download from Hugging Face)
- ⚠️ Results directory structure not tracked in git

---

## Related Work

- **DataDecide**: [allenai/DataDecide](https://huggingface.co/allenai/DataDecide-dolma1_7-1B)
- **HELM**: [HELM Benchmark](https://github.com/stanford-crfm/helm)
- **IRT for LLMs**: [Reliable, Efficient CAT](https://arxiv.org/abs/2504.09855)
- **Scaling Laws**: [Chinchilla](https://arxiv.org/abs/2203.15556), [Pythia](https://arxiv.org/abs/2304.01373)

---

## Troubleshooting

### Common Issues

#### 1. PyTorch 2.6+ `weights_only` Error

**Error:**
```
_pickle.UnpicklingError: Weights only load failed
```

**Solution:** PyTorch 2.6+ changed the default for `torch.load()`. If you see this error, either:
- Use PyTorch 2.5 or earlier: `pip install torch==2.5.0`
- Or update scripts to add `weights_only=False` to all `torch.load()` calls

#### 2. LaTeX Not Found

**Error:**
```
RuntimeError: Failed to process string with tex because latex could not be found
```

**Solution:** Install LaTeX system dependencies (see [Installation](#installation) section above)

#### 3. Python Version Issues

**Error:** Package installation fails or import errors

**Solution:** Ensure you're using Python 3.10:
```bash
python --version  # Should show 3.10.x
```

If using wrong version, recreate environment:
```bash
conda create --name deval python=3.10 -y
conda activate deval
```

#### 4. CUDA Device Errors

**Error:**
```
RuntimeError: CUDA error: invalid device ordinal
```

**Solution:** Some scripts have hard-coded CUDA devices (e.g., `cuda:7`). If you have fewer GPUs, edit the script to use `cuda:0` or `cpu`.

#### 5. HuggingFace Download Issues

**Error:** Timeouts or connection errors when downloading data

**Solution:**
```bash
# Pre-download datasets
huggingface-cli download stair-lab/irsl_downstream_resmat --repo-type dataset
huggingface-cli download stair-lab/irsl_testtime_resmat2 --repo-type dataset
```

---

## Contact

For questions about the code or paper, please open an issue in this repository.

---

## License

[License TBD - Add before public release]
