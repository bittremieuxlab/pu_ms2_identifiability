#!/bin/bash
#SBATCH --job-name=predict_lance
#SBATCH --output=logs/predict_lance_%j.log
#SBATCH --error=logs/predict_lance_%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

#SBATCH --time=36:00:00
#

#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module purge
module load cuda/12.8
module load miniconda/25.1.1

# Activate conda environment
# NOTE: If using this script outside a cluster, ensure you have created the environment:
#       conda env create -f environment.yml
conda activate instrument_setting

# Debug: confirm environment
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "========================================"
echo "Python path: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
echo "========================================"

# Move to project directory
cd /path/to/your/working/directory

# -----------------------
# Pre-trained Model Checkpoint
# -----------------------
# Download the pre-trained BCE model checkpoints from Zenodo:
#   Zenodo DOI: [10.5281/zenodo.XXXXXX] (to do: LINK TO BE ADDED)
#   Files: 
#     - best_model_bce_positive.ckpt (for positive polarity, --polarity 1)
#     - best_model_bce_negative.ckpt (for negative polarity, --polarity 0)
#
# Place the checkpoints at:
#   /path/to/your/working/directory/checkpoints/best_model_bce_positive.ckpt
#   /path/to/your/working/directory/checkpoints/best_model_bce_negative.ckpt
#
# Or specify your own trained model checkpoint paths below

# -----------------------
# Test Data (Lance Format)
# -----------------------
# Download the Test Set 1 Lance dataset from Zenodo:
#   Zenodo DOI: [10.5281/zenodo.YYYYYY] (to do: LINK TO BE ADDED)
#   File: test_set_1_lance.tar.gz
#
# Extract to:
#   /path/to/your/working/directory/data/lance_data_test_set_1
#
# Or use your own Lance-formatted dataset

echo "========================================"
echo "Starting inference with polarity-specific model..."
echo "========================================"

python scripts/inference/predict_lance_diff_polarity_one_hot.py \
    --checkpoint_path /path/to/your/working/directory/checkpoints/best_model_bce_negative.ckpt \
    --lance_path /path/to/your/working/directory/data/lance_data_test_set_1 \
    --output_csv results/predictions_polarity.csv \
    --polarity 0 \
    --batch_size 2048


