#!/bin/bash

#SBATCH --job-name=msv_download
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=cpucourt

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module purge
module load miniconda/25.1.1

# 2. Ask the system where conda is located and source it
# This replaces your "source ~/miniconda3/..." line
source $(conda info --base)/etc/profile.d/conda.sh
# Activate conda environment
# NOTE: If using this script outside a cluster, ensure you have created the environment:
#       conda env create -f environment.yml
conda activate instrument_setting

# Debug
echo "Python path: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"

# Change to project directory
cd /path/to/your/working/directory

# Run the download script with srun
srun python scripts/data_download/msv_download_datasets.py \
    --new_csv data/metadata/train_datasets.csv \
    --base_dir data/data \
    --download_limit 15 \
    --max_retries 3
