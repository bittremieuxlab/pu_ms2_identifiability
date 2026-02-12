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


#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module purge
module load cuda/12.8
module load miniconda/25.1.1
# 2. Ask the system where conda is located and source it
source $(conda info --base)/etc/profile.d/conda.sh
# Activate conda environment

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
# Download the pre-trained nnPU model checkpoint from Zenodo:
#   Zenodo DOI: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
#   File: best_model_nnpu.ckpt
#
# Place the checkpoint at:
#   /path/to/your/working/directory/checkpoints/best_model_nnpu.ckpt
#
# Or specify your own trained model checkpoint path below

# -----------------------
# Test Data (Lance Format)
# -----------------------
# Download the Test Set 3 Lance dataset from Zenodo:
#   Zenodo DOI: [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)
#   File: lance_data_test_set_3.tar.gz
#
# Extract to:
#   /path/to/your/working/directory/data/lance_data_test_set_3
#
# Or use your own Lance-formatted dataset

echo "========================================"
echo "Starting inference..."
echo "========================================"

python scripts/inference/predict_lance_all.py \
    --checkpoint_path /path/to/your/working/directory/checkpoints/best_model_nnpu.ckpt \
    --lance_path /path/to/your/working/directory/data/lance_data_test_set_3/test_data \
    --output_csv results/predictions.csv \
    --batch_size 2048 \
    --fetch_metadata

