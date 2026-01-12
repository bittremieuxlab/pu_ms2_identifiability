# Influence of instrument settings on quality of MS2 spectra



This repository contains the complete pipeline for training model that predicts the quality of MS2 spectra based on provided MS1 spectra and instrument configurations used to generate MS2.
## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Inference](#inference)


## 🔬 Overview

 This project addresses the challenge of identifying high-quality MS2 spectra when true negative examples (definitively poor-quality spectra) are unavailable.

### Key Features

- **Positive-Unlabeled (PU) Learning**: Trains models with only positive examples (library-matched spectra) and unlabeled data
- **Instrument Settings Integration**: Incorporates acquisition parameters as features
- **Polarity-Specific Modeling**: Separate processing for positive and negative ionization modes
- **Transformer Architecture**: Uses spectrum transformers (depthcharge library) for spectral encoding
- **Lance Dataset Format**: Efficient storage and loading for large-scale spectral data

### Methodology

1. **Data Labeling**: MS2 spectra are labeled via GNPS spectral library matching (cosine similarity > 0.7, ≥6 matching peaks)
2. **BCE Pre-training**: Separate models trained for positive and negative polarity using binary cross-entropy loss
3. **Prior Estimation**: Class prior probability (*c*) estimated using held-out positive examples (Test Set 1)
4. **nnPU Training**: Final model trained with non-negative PU loss using estimated polarity-specific priors
5. **Threshold Selection**: Decision threshold determined using Test Set 2 (5th percentile of positive probabilities)
6. **Evaluation**: Final performance assessed on Test Set 3

### Results Summary

- **Positive ionization**: Prior π = 0.454, c = 0.5889
- **Negative ionization**: Prior π = 0.285, c = 0.4486
- **Test Set 3 Recall**: 0.8986 at threshold 0.767

## 📁 Repository Structure

```
spectral_quality_assessment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
│
├── src/                               # Core model implementations
│   └── transformers/
│       ├── model_bce_loss_one_hot.py              # BCE loss model (polarity-aware)
│       └── model_nn_pu_loss_detach_diff_polarity.py  # nnPU loss model
│
├── scripts/                           # Python scripts for each pipeline step
│   ├── data_download/
│   │   └── msv_download_datasets.py   # Download datasets from MassIVE using CSV metadata
│   │
│   ├── data_preprocessing/
│   │   ├── split_library.py           # Split GNPS library by polarity
│   │   ├── process_raw.py             # Convert .raw → .mzML, run ScanHeadsman
│   │   ├── library_matching_diff_polarity.py  # GNPS library matching
│   │   ├── data_processing_pipeline.py        # Complete data processing pipeline
│   │   └── create_lance_add_one_hot.py        # Create Lance dataset
│   │
│   ├── training/
│   │   ├── training_bce_loss_diff_polarity_one_hot.py  # Train BCE models
│   │   └── training_nn_pu_loss_detach_diff_polarity.py # Train nnPU model
│   │
│   └── inference/
│       ├── predict_lance_all.py                        # Run predictions
│       └── predict_lance_diff_polarity_one_hot.py      # Polarity-specific predictions
│
├── slurm_scripts/                     # Cluster job submission scripts
│   ├── data_download/
│   │   └── msv_download.sh
│   ├── data_preprocessing/
│   │   ├── run_process_raw.sh         # Convert raw files to mzML
│   │   ├── library_matching.sh        # Run library matching
│   │   ├── run_processing_pipeline.sh # Complete processing pipeline
│   │   └── run_build_lance.sh         # Build Lance datasets
│   ├── training/
│   │   ├── run_train_bce_loss_diff_polarity.sh
│   │   └── run_train_nnpu_loss.sh
│   └── inference/
│       ├── run_predict_lance.sh
│       └── run_predict_lance_val.sh
│
├── checkpoints/                       # Pre-trained model checkpoints (download from Zenodo)
│   └── README.md                      # Download instructions
│
├── tools/                             # External tools (download separately)
│   └── README.md                      # Installation guide for ThermoRawFileParser & ScanHeadsman
│
├── data/                              # Data and metadata
│   ├── README.md                      # Data directory documentation
│   ├── metadata/                      # Dataset metadata (in repo)
│   │   ├── train_datasets.csv
│   │   ├── val_datasets.csv
│   │   ├── test_1_metadata.csv
│   │   ├── test_2_metadata.csv
│   │   └── test_3_metadata.csv
│   ├── libraries/                     # GNPS libraries (download & split)
│   │   └── README.md                  # Download & split instructions
│   ├── file_paths/                    # [User-created] Lists of local file paths
│   │   ├── file_paths_train.txt
│   │   └── file_paths_val.txt
│   ├── lance_datasets/                # [External] Training & validation Lance data (download from Zenodo)
│   ├── lance_data_test_set_1/         # [External] Test Set 1 (download from Zenodo)
│   ├── lance_data_test_set_2/         # [External] Test Set 2 (download from Zenodo)
│   └── lance_data_test_set_3/         # [External] Test Set 3 (download from Zenodo)
│
└── docs/                              # Detailed documentation
    ├── DATA_PREPROCESSING.md          # Preprocessing pipeline
    ├── TRAINING.md                    # Model training guide
    └── INFERENCE.md                   # Running predictions
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA 12.8+ (for GPU training)
- Conda or pip
- Access to a computing cluster (recommended for full pipeline)

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/spectral_quality_assessment.git
cd spectral_quality_assessment

# Create and activate conda environment
conda env create -f environment.yml
conda activate instrument_setting
```



**External Tools** (only needed for raw data processing):
- **ThermoRawFileParser** - Convert .raw to .mzML (see `tools/README.md`)
- **ScanHeadsman** - Extract MS1 spectra (see `tools/README.md`)
- **Mono runtime** - Required to run .NET executables on Linux/macOS

> **Note**: External tools are not required if using pre-processed data from Zenodo


## 📊 Data Preparation

### Step 1: Download data

Download the preprocessed Lance datasets directly from Zenodo.

**Training and Validation Data:**
- Zenodo DOI: [10.5281/zenodo.XXXXXX](https://zenodo.org/record/XXXXXX) (to do: LINK TO BE ADDED)
- File: `train_validation_lance.tar.gz` (XX GB)
- Contains ONE Lance dataset with TWO tables: `train_data` and `validation_data`
- Extract to: `data/lance_datasets/`

**Test Sets:**
- Zenodo DOI: [10.5281/zenodo.YYYYYY](https://zenodo.org/record/YYYYYY) (to do: LINK TO BE ADDED)
- Files:
  - `test_set_1_lance.tar.gz` - For prior estimation (XX GB)
  - `test_set_2_lance.tar.gz` - For threshold selection (XX GB)
  - `test_set_3_lance.tar.gz` - For final evaluation (XX GB)
- Extract to: `data/lance_data_test_set_1/`, `data/lance_data_test_set_2/`, `data/lance_data_test_set_3/`

```bash
# Download from Zenodo
cd data/

# Extract training and validation data (contains both train_data and validation_data tables)
tar -xzf train_validation_lance.tar.gz
# This creates: data/lance_datasets/ with train_data/ and validation_data/ subdirectories

# Extract test sets
tar -xzf test_set_1_lance.tar.gz
tar -xzf test_set_2_lance.tar.gz
tar -xzf test_set_3_lance.tar.gz
```

**Note**: Dataset metadata files (`training_metadata.csv`, etc.) are provided in `data/metadata/`.

### (Optional) Preparing Data from Scratch

If you want to process raw data and create the Lance datasets yourself:

#### 1. Install External Tools

Install ThermoRawFileParser and ScanHeadsman (required for .raw file processing):

```bash
# See detailed installation instructions
cat tools/README.md

# Quick install (example for ThermoRawFileParser)
cd tools
wget https://github.com/compomics/ThermoRawFileParser/releases/download/v1.4.4/ThermoRawFileParser1.4.4.zip
unzip ThermoRawFileParser1.4.4.zip -d ThermoRawFileParser/
```

For detailed instructions, see [`tools/README.md`](tools/README.md)

#### 2. Download GNPS Spectral Libraries

For library matching, download spectral libraries from GNPS:

```bash
# Create directory for libraries
mkdir -p data/libraries

# Download from GNPS (https://gnps.ucsd.edu/ProteoSAFe/libraries.jsp)
# Recommended libraries:
#   - GNPS-LIBRARY (all public spectral libraries)
#   - Separate by polarity: positive mode and negative mode

# Place downloaded files as:
#   data/libraries/spectral_db_positive.mgf
#   data/libraries/spectral_db_negative.mgf
```

**Direct download links**:
- Visit [GNPS Spectral Libraries](https://external.gnps2.org/processed_gnps_data/matchms.mgf)
- Or browse: [GNPS Library Portal](https://gnps.ucsd.edu/ProteoSAFe/libraries.jsp)
- Download "ALL_GNPS" library 
- Filter/split by ionization mode if needed

#### 3. Run Data Processing Pipeline

See detailed instructions in `docs/DATA_PREPROCESSING.md` for:
- Converting raw files to mzML and mgf
- Running library matching
- Creating Lance datasets

## 🤖 Using Pre-trained Models

**Skip training and use our pre-trained models!**

We provide pre-trained model checkpoints on Zenodo for immediate use:

### Download Pre-trained Models

```bash
cd checkpoints/

# Download all pre-trained models from Zenodo
# Zenodo DOI: [10.5281/zenodo.XXXXXX] (LINK TO BE ADDED)

# Or use wget (links to be added)
wget https://zenodo.org/record/XXXXXX/files/best_model_nnpu.ckpt
wget https://zenodo.org/record/XXXXXX/files/best_model_bce_positive.ckpt
wget https://zenodo.org/record/XXXXXX/files/best_model_bce_negative.ckpt
```

### Available Models

1. **nnPU Model** (`best_model_nnpu.ckpt`) - **Recommended for production**
   - Final model 
   - Test Set 3 Recall: 0.8986 @ threshold 0.767
   - Trained on both polarities with class priors

2. **BCE Models** (polarity-specific)
   - `best_model_bce_positive.ckpt` - For positive ionization mode
   - `best_model_bce_negative.ckpt` - For negative ionization mode
   - Used for prior estimation and polarity-specific analysis

See [`checkpoints/README.md`](checkpoints/README.md) for detailed documentation.

### Running Inference with Pre-trained Models

```bash
# Edit the checkpoint path in the script, then run:
sbatch slurm_scripts/inference/run_predict_lance.sh
```

For more details, see the [Inference](#-inference) section below.

---

## 🎯 Model Training (Optional)

**Note**: Training from scratch is optional if you're using the pre-trained models above.

If you want to train the model, you'll need training data. You have **two options**:

1. **Download pre-processed Lance datasets from Zenodo** (recommended, see [Data Preparation](#-data-preparation))
2. **Process data from scratch** (see [Preparing Data from Scratch](#optional-preparing-data-from-scratch))

### Phase 1: BCE Pre-training (Polarity-Specific)

Train separate models for positive and negative ionization modes:

```bash
cd slurm_scripts/training

# Train on positive polarity
sbatch run_train_bce_loss_diff_polarity.sh  # Set --polarity 1 in script

# Train on negative polarity  
sbatch run_train_bce_loss_diff_polarity.sh  # Set --polarity 0 in script
```

**Model Selection**: The model with highest validation recall is selected for each polarity.

### Phase 2: Prior Estimation

After BCE training, estimate class priors using Test Set 1:

```bash
# Run predictions on Test Set 1 using best BCE models
python scripts/inference/predict_lance_diff_polarity_one_hot.py \
    --checkpoint_path logs/bce_pos/best_model.ckpt \
    --lance_path /path/to/test_set_1 \
    --output_csv test1_predictions_pos.csv \
    --polarity 1

# Calculate average probability over known positives
# For positive polarity: ĉ = 0.5889 → π = 0.454
# For negative polarity: ĉ = 0.4486 → π = 0.285
```

### Phase 3: nnPU Training with Estimated Priors

```bash
cd slurm_scripts/training

# Edit run_train_nnpu_loss.sh to set:
#   --prior_pos 0.454
#   --prior_neg 0.285

sbatch run_train_nnpu_loss.sh
```

See [`docs/TRAINING.md`](docs/TRAINING.md) for detailed training procedures and hyperparameters.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_layers | 2 |
| dropout | 0.3 |
| encoder_lr | 5e-6 |
| linear_lr | 1e-5 |
| batch_size | 256 per GPU |
| weight_decay | 0.001 |
| instrument_embedding_dim | 16 |

## 🔮 Inference

### Predict on New Data

Use the pre-trained nnPU model (download from Zenodo - see [Pre-trained Models](#-using-pre-trained-models)):


Submit as a SLURM job:

```bash
# Edit checkpoint path in script if needed
sbatch slurm_scripts/inference/run_predict_lance.sh
```

Output CSV contains:
- `original_index`: Index in Lance dataset
- `probability`: Predicted quality score (0-1)
- `label`: Ground truth label (if available)
- `mzml_filepath`: Path to the source file
- `scan_number`: MS2 scan number


### Threshold Selection

Using Test Set 2, the decision threshold was set to **0.767** (5th percentile of positive sample probabilities).




