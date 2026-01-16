#!/bin/bash
#SBATCH --job-name=ms_pipeline
#SBATCH --output=logs/ms_pipeline_%j.log
#SBATCH --error=logs/ms_pipeline_%j.err
#SBATCH --partition=cpucourt
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# -----------------------
# Load environment
# -----------------------
module purge
module load miniconda/25.1.1
# This replaces your "source ~/miniconda3/..." line
source $(conda info --base)/etc/profile.d/conda.sh
# Activate conda environment
# NOTE:  ensure you have created the environment:
#       conda env create -f environment.yml
conda activate instrument_setting

# Debug info
echo "Python path: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"
echo "Pandas version:"
python -c "import pandas; print(pandas.__version__)"

# -----------------------
# Move to project folder
# -----------------------
cd /path/to/your/working/directory

# Create logs directory if it doesn't exist
mkdir -p logs

# -----------------------
# Define arguments
# -----------------------
PARENT_DIR="/path/to/your/working/directory/data"
OUTPUT_PARENT_DIR="/path/to/your/working/directory/data"
TSV_FILE="/path/to/your/working/directory/results/spectral_matching_results.tsv"
LOG_DIR="logs/pipeline"

# -----------------------
# Process each MSV folder
# -----------------------
echo "Starting MS data processing pipeline..."
echo "Parent directory: $PARENT_DIR"
echo "Output parent directory: $OUTPUT_PARENT_DIR"
echo "TSV file: $TSV_FILE"
echo "Log directory: $LOG_DIR"
echo "----------------------------------------"

# Counter for tracking progress
total_folders=0
processed_folders=0
failed_folders=0

# Count total MSV folders
for msv_folder in "$PARENT_DIR"/MSV*; do
    if [ -d "$msv_folder" ]; then
        ((total_folders++))
    fi
done

echo "Found $total_folders MSV folders to process"
echo "----------------------------------------"

# Loop through all MSV* folders in the parent directory
for msv_folder in "$PARENT_DIR"/MSV*; do
    # Check if it's a directory
    if [ -d "$msv_folder" ]; then
        # Extract the MSV folder name
        msv_name=$(basename "$msv_folder")

        echo ""
        echo "========================================"
        echo "Processing folder: $msv_name"
        echo "Progress: $((processed_folders + 1))/$total_folders"
        echo "========================================"

        # Define input and output paths for this MSV folder
        INPUT_DIR="$msv_folder"
        OUTPUT_DIR="$OUTPUT_PARENT_DIR/$msv_name"

        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        echo "Input directory: $INPUT_DIR"
        echo "Output directory: $OUTPUT_DIR"

        # Run the Python script for this folder
        if srun python scripts/data_preprocessing/data_processing_pipeline.py \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR" \
            --tsv "$TSV_FILE" \
            --log-dir "$LOG_DIR"; then
            echo "✓ Successfully processed $msv_name"
            ((processed_folders++))
        else
            echo "✗ Failed to process $msv_name"
            ((failed_folders++))
        fi

        echo "----------------------------------------"
    fi
done

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "Total folders: $total_folders"
echo "Successfully processed: $processed_folders"
echo "Failed: $failed_folders"
echo "========================================"