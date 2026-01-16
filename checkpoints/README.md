# Pre-trained Model Checkpoints

This directory contains pre-trained model checkpoints for MS2 quality prediction based on MS1 and instrument configurations that were used to generate consecutive MS2. The models are **not included** in this repository due to file size and should be downloaded from Zenodo.

## Available Pre-trained Models

### 1. BCE Pre-trained Models (Polarity-Specific)

These models were trained with Binary Cross-Entropy loss separated by ionization mode:

**Positive Ionization Mode**:
- **File**: `best_model_bce_positive.ckpt`
- **Zenodo DOI**: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
- **Training**: Trained on positive mode spectra only
- **Validation Recall**: ~0.5996
- **Use case**: Prior estimation, polarity-specific predictions

**Negative Ionization Mode**:
- **File**: `best_model_bce_negative.ckpt`
- **Zenodo DOI**: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
- **Training**: Trained on negative mode spectra only
- **Validation Recall**: ~0.4392
- **Use case**: Prior estimation, polarity-specific predictions

### 2. nnPU Model (Final Model)

This is the final model trained with non-negative Positive-Unlabeled (nnPU) loss using estimated class priors:

**Combined Model**:
- **File**: `best_model_nnpu.ckpt`
- **Zenodo DOI**: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
- **Training**: Both polarities, nnPU loss with π_pos=0.45, π_neg=0.29
- **Test Set 3 Recall**: 0.8986 (at threshold 0.767)
- **Use case**: Final quality predictions 

## How to Download

### Option 1: Direct Download via wget

```bash
cd checkpoints/

# Download BCE models
wget https://zenodo.org/record/18266932/files/best_model_bce_positive.ckpt
wget https://zenodo.org/record/18266932/files/best_model_bce_negative.ckpt

# Download nnPU model
wget https://zenodo.org/record/18266932/files/best_model_nnpu.ckpt
```

### Option 2: Download from Zenodo Web Interface

1. Visit the Zenodo record: [https://zenodo.org/record/18266932](https://zenodo.org/record/18266932)
2. Download the checkpoint files
3. Place them in this directory


## File Structure

After downloading, your directory should look like:

```
checkpoints/
├── README.md                        # This file (in repo)
├── best_model_bce_positive.ckpt     # Download from Zenodo
├── best_model_bce_negative.ckpt     # Download from Zenodo
└── best_model_nnpu.ckpt             # Download from Zenodo
```

