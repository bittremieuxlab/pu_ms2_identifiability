#!/bin/bash
#SBATCH --job-name=spectral_matching_parallel_batch_2
#SBATCH --output=logs/spectral_matching_%j.log
#SBATCH --error=logs/spectral_matching_%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpucourt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=48:00:00

#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
# Load necessary modules
module purge
module load miniconda/25.1.1
# Initialize Conda for shell interaction
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
# NOTE: If using this script outside a cluster, ensure you have created the environment:
#       conda env create -f environment.yml
conda activate instrument_setting



# Go to your project directory
cd /path/to/your/working/directory

# NOTE: Download spectral library files from GNPS before running:
# - Visit https://external.gnps2.org/processed_gnps_data/matchms.mgf

# - Split into positive and negative mode using `scripts/data_preprocessing/split_library.py`->it will create:spectral_db_positive.mgf and spectral_db_negative.mgf

python /path/to/your/working/directory/scripts/data_preprocessing/library_matching_diff_polarity.py \
    --msv_folder /path/to/your/working/directory/data\
    --reference_mgf_positive /path/to/your/working/directory/data/libraries/spectral_db_positive.mgf \
    --reference_mgf_negative /path/to/your/working/directory/data/libraries/spectral_db_negative.mgf \
    --output_tsv /path/to/your/working/directory/results/spectral_matching_results.tsv \
    --num_cpus $SLURM_CPUS_PER_TASK
