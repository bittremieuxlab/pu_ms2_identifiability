#!/bin/bash
#SBATCH --job-name=convert_lance
#SBATCH --output=/path/to/your/working/directory/logs/convert_lance_%j.log
#SBATCH --error=/path/to/your/working/directory/logs/convert_lance_%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpucourt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00


#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module purge

module load miniconda/25.1.1

source $(conda info --base)/etc/profile.d/conda.sh



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

# Create logs and results directories
mkdir -p logs
mkdir -p results/lance_datasets

# Verify required packages
echo "Checking required packages..."
python -c "import lance; import pyarrow; import pyteomics; import spectrum_utils; print('✓ All packages available')" || {
    echo "ERROR: Missing required packages. Install with:"
    echo "pip install lance pyarrow pyteomics spectrum-utils"
    exit 1
}

# Run the Lance conversion script
echo "========================================"
echo "Starting Lance conversion..."
echo "========================================"

python scripts/data_preprocessing/create_lance_add_one_hot.py \
    --train_file_list data/file_paths/file_paths_train.txt \
    --val_file_list data/file_paths/file_paths_val.txt \
    --lance_uri results/lance_data_train_validation \
    --train_table train_data \
    --val_table validation_data \
    --workers $SLURM_CPUS_PER_TASK \
    --cap_training_set 300000 \
    --cap_val_set 100000 \
    --training_set_csv data/metadata/train_datasets.csv \
    --val_set_csv data/metadata/val_datasets.csv \


EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Lance conversion completed successfully!"
    echo "Database location: results/lance_datasets"

    # Show dataset stats
    python -c "
import os
import lance
for table_name in ['train_data', 'validation_data']:
    dataset_path = os.path.join('results/lance_datasets', table_name)
    if os.path.exists(dataset_path):
        ds = lance.dataset(dataset_path)
        print(f'{table_name}: {ds.count_rows():,} records')
    else:
        print(f'{table_name} dataset not found')
" 2>/dev/null
else
    echo "❌ Lance conversion failed with exit code: $EXIT_CODE"
fi
echo "Job ended at: $(date)"
echo "========================================"

exit $EXIT_CODE