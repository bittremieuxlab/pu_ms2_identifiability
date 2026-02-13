#!/bin/bash
#
# SLURM Script for Training BCE Model (Polarity-Specific)
#

#SBATCH --job-name=spectra_transformer_lance
#SBATCH --output=logs/training_lance_%j.log
#SBATCH --error=logs/training_lance_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

#SBATCH --time=36:00:00
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module purge
module load cuda/12.8
module load miniconda/25.1.1

# 2. Ask the system where conda is located and source it
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
# NOTE: If using this script outside a cluster, ensure you have created the environment:
#       conda env create -f environment.yml
conda activate instrument_setting
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Debug: confirm environment and torch
echo "Python path: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import lance; print('Lance version:', lance.__version__)"

# Change to working directory
cd /path/to/your/working/directory
mkdir -p logs

# -----------------------
# Training Data (Lance Format)
# -----------------------
# You have two options to obtain the training and validation Lance dataset:
#
# OPTION 1: Download pre-processed Lance dataset from Zenodo (RECOMMENDED)
#   Zenodo DOI: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
#   File: lance_data_train_validation.tar.gz
#   
#   This archive contains one Lance dataset with TWO tables:
#     - train_data (training spectra)
#     - validation_data (validation spectra)
#   
#   Download and extract:
#     cd data/
#     wget https://zenodo.org/record/18266932/files/lance_data_train_validation.tar.gz
#     tar -xzf lance_data_train_validation.tar.gz
#   
#   See data/README.md for detailed download instructions
#
# OPTION 2: Create Lance dataset from scratch
#   1. Download raw data from MassIVE
#   2. Download GNPS libraries
#   3. Process and create Lance dataset with both tables
#   
#   See docs/DATA_PREPROCESSING.md for step-by-step guide

echo "========================================"
echo "Starting BCE training (polarity-specific)..."
echo "========================================"

# Run your training script with Lance database
# Note: Both arguments point to the same Lance dataset directory
#       The script will access train_data and validation_data tables within it
srun --gpu-bind=closest python scripts/training/training_bce_loss_diff_polarity_one_hot.py \
    --lance_uri data/lance_datasets \
    --lance_uri_val data/lance_datasets \
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
    --polarity 0 \



echo "Training completed at $(date)"