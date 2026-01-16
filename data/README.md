# Data Directory

This directory should all data files needed for training, validation, and testing. Most data files are **not included** in this repository due to size and should be downloaded from Zenodo.

## Directory Structure

```
data/
├── README.md                           # This file (in repo)
├── metadata/                           # Dataset metadata (in repo)
│   ├── train_datasets.csv
│   ├── val_datasets.csv
│   ├── test_1_metadata.csv
│   ├── test_2_metadata.csv
│   └── test_3_metadata.csv
├── libraries/                          # GNPS spectral libraries
│   ├── README.md                       # ✓ In repo (download & split instructions)
│   ├── cleaned_spectra.mgf             # [External] Download from GNPS2
│   ├── spectral_db_positive.mgf        # [User-created] Split from cleaned_spectra.mgf
│   └── spectral_db_negative.mgf        # [User-created] Split from cleaned_spectra.mgf        
├── file_paths/                         # [User-created] File path lists 
│   ├── file_paths_train.txt            # User creates
│   └── file_paths_val.txt              # User creates
├── lance_dataset_train_validation/     # [External] Training & validation (download from Zenodo)
│   ├── train_data/                     # Training table
│   └── validation_data/                # Validation table
├── lance_data_test_set_1/              # [External] Test Set 1 (download from Zenodo)
├── lance_data_test_set_2/              # [External] Test Set 2 (download from Zenodo)
├── lance_data_test_set_3/              # [External] Test Set 3 (download from Zenodo)
├── new_data/                           # This will be created for Massive datasets with .raw files (optional, if processing new data)

```

## Downloading Pre-processed Data from Zenodo

**Data and model weights repository:** [10.5281/zenodo.18266932](https://doi.org/10.5281/zenodo.18266932)

### Quick Start
Run the following commands **from inside this directory** (`data/`) to download and extract all required files.


```bash
# --- 1. Download & Extract Datasets (Current Directory) ---

# Train & Validation Data
wget https://zenodo.org/record/18266932/files/lance_data_train_validation.tar.gz
tar -xzf lance_data_train_validation.tar.gz

# Test Sets
wget https://zenodo.org/record/18266932/files/lance_data_test_set_1.tar.gz
wget https://zenodo.org/record/18266932/files/lance_data_test_set_2.tar.gz
wget https://zenodo.org/record/18266932/files/lance_data_test_set_3.tar.gz

tar -xzf lance_data_test_set_1.tar.gz
tar -xzf lance_data_test_set_2.tar.gz
tar -xzf lance_data_test_set_3.tar.gz

# Cleanup data zips
rm *_lance.tar.gz





---

## 🔧 Creating Data from Scratch (Advanced)

If you want to reproduce the data processing pipeline from raw MassIVE data:

### Step 1: Download Raw Data from MassIVE

The metadata CSV files (`train_datasets.csv`, `val_datasets.csv`, etc.) contain MassIVE dataset IDs in the `dataset_id` column. Use the provided script to automatically download them:

```bash
# Edit the script to specify which datasets to download
cd /path/to/your/working/directory

# Submit download job (after configuring - see below)
sbatch slurm_scripts/data_download/msv_download.sh
```

**Before running**, edit `slurm_scripts/data_download/msv_download.sh` to configure:

```bash
srun python scripts/data_download/msv_download_datasets.py \
    --new_csv data/metadata/train_datasets.csv \  # CSV with dataset IDs to download
    --base_dir data/new_data \                    # Output directory (MSV folders created here)
    --download_limit 15 \                         # Max number of datasets to download
    --max_retries 3                               # Retry attempts per dataset (optional, default=3)
```

**Arguments**:
- `--new_csv` (required): Path to CSV file containing `dataset_id` column with MassIVE accessions
- `--base_dir` (required): Base directory where `MSV*` folders will be created
- `--download_limit` (optional, default=2): Maximum number of datasets to download (useful for testing or batch processing)
- `--max_retries` (optional, default=3): Number of retry attempts if download fails

**What this does**:
- Reads MassIVE dataset IDs from the `dataset_id` column in the specified CSV file
- Downloads raw `.raw` files from MassIVE (https://massive.ucsd.edu/) using the `ppx` library
- Organizes downloads into `data/new_data/MSV*/` folders (one folder per dataset)
- Implements automatic retry logic with exponential backoff for failed downloads
- Stops after reaching the download limit
- Prints a summary of successful and failed downloads

**Example: Download different dataset splits**

```bash
# Download training datasets
# Edit msv_download.sh to use:
#   --new_csv data/metadata/train_datasets.csv
sbatch slurm_scripts/data_download/msv_download.sh

# Download validation datasets  
# Edit msv_download.sh to use:
#   --new_csv data/metadata/val_datasets.csv
sbatch slurm_scripts/data_download/msv_download.sh

# Download test set 3
# Edit msv_download.sh to use:
#   --new_csv data/metadata/test_3_metadata.csv
sbatch slurm_scripts/data_download/msv_download.sh
```





### Step 2: Download GNPS Libraries

Follow instructions in `data/libraries/README.md` to:
1. Download GNPS library from [GNPS](https://external.gnps2.org/processed_gnps_data/matchms.mgf)
2. Split into positive and negative mode using `scripts/data_preprocessing/split_library.py`

### Step 3: Process Raw Data

Once you have raw files and GNPS libraries, follow the complete preprocessing pipeline:

```bash
# 1. Convert .raw to .mzML and extract MS1
sbatch slurm_scripts/data_preprocessing/run_process_raw.sh

# 2. Run library matching (label spectra based on GNPS matches)
sbatch slurm_scripts/data_preprocessing/library_matching.sh

# 3. Process files and prepare for Lance creation
sbatch slurm_scripts/data_preprocessing/run_processing_pipeline.sh

# 4. Create Lance datasets (after all processing is complete)
sbatch slurm_scripts/data_preprocessing/run_build_lance.sh
```

**Processing pipeline overview**:
1. **run_process_raw.sh**: Converts raw files to mzML format and extracts MS1 spectra using ThermoRawFileParser and ScanHeadsman
2. **library_matching.sh**: Matches MS2 spectra against GNPS libraries to create labels (positive = library match, unlabeled = no match)
3. **run_processing_pipeline.sh**: Processes all MSV folders, prepares data structures, and organizes files for Lance dataset creation
4. **run_build_lance.sh**: Creates the final Lance datasets (train_data and validation_data tables) from processed mzML files

See `docs/DATA_PREPROCESSING.md` for detailed step-by-step instructions.

### Metadata Files

The `metadata/` directory contains CSV files with dataset information:

- `train_datasets.csv` - Training dataset IDs and metadata
- `val_datasets.csv` - Validation dataset IDs and metadata
- `test_1_metadata.csv` - Test Set 1 dataset IDs
- `test_2_metadata.csv` - Test Set 2 dataset IDs  
- `test_3_metadata.csv` - Test Set 3 dataset IDs

**CSV Format**:
These files are **semicolon-separated (`;`)** and contain:
- `dataset_id` (required) - MassIVE accession (e.g., MSV000012345)
- `ms2` - Number of MS2 spectra
- Other metadata fields

**Important**: The download script `msv_download_datasets.py` requires:
- Semicolon (`;`) as separator
- A column named `dataset_id` containing MassIVE accessions
- The script will read unique dataset IDs from this column and download the corresponding `.raw` files

---


