# Model Training Guide

This guide covers the complete training pipeline for the spectral quality assessment model.

## Training Overview

The training process consists of three phases:

1. **Phase 1**: BCE pre-training (polarity-specific models)
2. **Phase 2**: Prior estimation using Test Set 1
3. **Phase 3**: nnPU training with estimated priors

## Prerequisites

- Lance datasets created or downloaded from Zenodo (see `DATA_PREPROCESSING.md`, /data/README.md)
- GPU cluster with CUDA 12.8+
- PyTorch 2.0+, Lightning 2.0+
- At least 200GB RAM per node (for multi-GPU training)

## Phase 1: BCE Pre-training

### Purpose

Train separate models for positive and negative ionization modes using binary cross-entropy (BCE) loss. These models serve as:
- Initialization for nnPU training
- Source for prior estimation (c parameter)

### Training Positive Polarity Model

```bash
cd slurm_scripts/training

# Edit run_train_bce_loss_diff_polarity.sh:
#   - Set --polarity 1 (for positive mode)
#   - Adjust paths to Lance datasets
#   - Set resource allocation

sbatch run_train_bce_loss_diff_polarity.sh
```

**Key Script Parameters:**

```bash
python scripts/training/training_bce_loss_diff_polarity_one_hot.py \
    --lance_uri /path/to/lance/ \
    --lance_uri_val /path/to/lance/ \
    --log_dir ./logs \
    --batch_size 256 \
    --num_workers 8 \
    --d_model 128 \
    --n_layers 2 \
    --dropout 0.3 \
    --linear_lr 0.00001 \
    --encoder_lr 0.000005 \
    --epochs 30 \
    --instrument_embedding_dim 16 \
    --weight_decay 0.001 \
    --polarity 1  # 1 for positive, 0 for negative
```

### Training Negative Polarity Model

Repeat the same process with `--polarity 0`:

```bash
sbatch run_train_bce_loss_diff_polarity.sh  # With --polarity 0
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Transformer embedding dimension |
| `n_layers` | 2 | Number of transformer encoder layers |
| `dropout` | 0.3 | Dropout rate |
| `encoder_lr` | 5e-6 | Learning rate for spectrum encoder |
| `linear_lr` | 1e-5 | Learning rate for linear layers |
| `batch_size` | 256 | Batch size per GPU |
| `weight_decay` | 0.001 | L2 regularization |
| `instrument_embedding_dim` | 16 | Dimension for instrument feature embedding |
| `epochs` | 30 | Maximum training epochs |

### Multi-GPU Training

The scripts use PyTorch Lightning DDP (Distributed Data Parallel):

```bash
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
```

Training automatically distributes across available GPUs.

### Early Stopping

Models use early stopping based on **validation recall**:
- Monitors: `val_recall`
- Patience: 10 epochs
- Mode: maximize

The model with highest validation recall is automatically saved.



## Phase 2: Prior Estimation

### Purpose

Estimate the probability **c = P(s=1|y=1)** - the probability that a truly high-quality spectrum (y=1) is labeled (s=1).

According to Elkan & Noto (2008), we estimate c by averaging model predictions over **known positive samples**.

### Step 1: Run Predictions on Test Set 1

Using the best BCE models from Phase 1:

**Positive polarity:**
```bash
python scripts/inference/predict_lance_diff_polarity_one_hot.py \
    --checkpoint_path logs/training_bce_pos/best_model-*.ckpt \
    --lance_path /path/to/test_1_data \
    --output_csv test1_predictions_pos.csv \
    --polarity 1
```

**Negative polarity:**
```bash
python scripts/inference/predict_lance_diff_polarity_one_hot.py \
    --checkpoint_path logs/training_bce_neg/best_model-*.ckpt \
    --lance_path /path/to/test_1_data \
    --output_csv test1_predictions_neg.csv \
    --polarity 0
```

### Step 2: Calculate c for Each Polarity

```python
import pandas as pd

# Positive polarity
df_pos = pd.read_csv('test1_predictions_pos.csv')
known_positives_pos = df_pos[df_pos['label'] == 1]
c_pos = known_positives_pos['probability'].mean()
print(f"Positive polarity c: {c_pos:.4f}")

# Negative polarity
df_neg = pd.read_csv('test1_predictions_neg.csv')
known_positives_neg = df_neg[df_neg['label'] == 1]
c_neg = known_positives_neg['probability'].mean()
print(f"Negative polarity c: {c_neg:.4f}")
```

**Expected values** (the ones that we obtained):
- Positive polarity: ĉ = 0.5889
- Negative polarity: ĉ = 0.4486

### Step 3: Calculate Class Priors

Using Equation 5 from Elkan & Noto (2008): **π = s / c**

Where:
- **s** = Fraction of labeled positive samples in the training data
- **c** = Estimated label frequency from Step 2

**Calculate s (positivity rate) using the analysis script**:

```bash
# For positive polarity
python scripts/post_processing_scripts/analyze_lance_one_hot.py \
    --path data/lance_datasets/train_data \
    --polarity 1

# For negative polarity
python scripts/post_processing_scripts/analyze_lance_one_hot.py \
    --path data/lance_datasets/train_data \
    --polarity 0
```

The script outputs the "Positivity Rate" (as percentage), which is `s`.

**Calculate π (class prior)**:

Using the c values from Step 2 and the s values from the script:

- **Positive polarity**: π = s_pos / c_pos = 0.267 / 0.5889 = **0.454**
- **Negative polarity**: π = s_neg / c_neg = 0.128 / 0.4486 = **0.285**

**Expected values**:
- Positive polarity: π = 0.454
- Negative polarity: π = 0.285

## Phase 3: nnPU Training

### Purpose

Train the final model using non-negative PU (nnPU) loss with polarity-specific priors.

### nnPU Loss

The nnPU loss (Kiryo et al., 2017) is designed for PU learning:

```
L_nnPU = π · E_p[l(f(x))] + max(0, E_u[l(-f(x))] - π · E_p[l(-f(x))])
```

Where:
- π: class prior (estimated in Phase 2)
- E_p: expectation over positive samples
- E_u: expectation over unlabeled samples
- l: loss function (binary cross-entropy)
- f(x): model prediction

### Training Script

```bash
cd slurm_scripts/training

# Edit run_train_nnpu_loss.sh to set:
sbatch run_train_nnpu_loss.sh
```

**Key parameters:**

```bash
python scripts/training/training_nn_pu_loss_detach_diff_polarity.py \
    --lance_uri /path/to/train_data \
    --lance_uri_val /path/to/validation_data \
    --log_dir ./logs \
    --batch_size 256 \
    --num_workers 8 \
    --d_model 128 \
    --n_layers 2 \
    --dropout 0.3 \
    --linear_lr 0.00001 \
    --encoder_lr 0.000005 \
    --epochs 30 \
    --instrument_embedding_dim 16 \
    --weight_decay 0.001 \
    --prior_pos 0.454 \     # Set from Phase 2
    --prior_neg 0.285       # Set from Phase 2
```


Best checkpoint:
```
logs/training_nnpu/best_model-epoch=XX-val_recall=0.YYYY.ckpt
```

