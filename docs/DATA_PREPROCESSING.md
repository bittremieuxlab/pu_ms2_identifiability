# Data Preprocessing Pipeline

This document describes the complete data preprocessing pipeline from .raw MS files to Lance dataset format.

## Pipeline Overview

```
.raw files → .mzML, .mgf, .csv → Library Matching → Data Processing → Lance Dataset
```

**Quick Start** (complete workflow):

```bash
# Step 1: Convert .raw files and extract instrument settings
sbatch slurm_scripts/data_preprocessing/run_process_raw.sh

# Step 2: Run library matching to label spectra
sbatch slurm_scripts/data_preprocessing/library_matching.sh

# Step 3a: Process and prepare data for Lance creation
sbatch slurm_scripts/data_preprocessing/run_processing_pipeline.sh

# Step 3b: Create Lance datasets
sbatch slurm_scripts/data_preprocessing/run_build_lance.sh
```



## Prerequisites

### External Tools (Download Required)

- **ThermoRawFileParser** - Converts .raw files to .mzML and .mgf formats
- **ScanHeadsman** - Extracts instrument settings and scan metadata

See `tools/README.md` for download and installation instructions.

### Python Libraries

- **matchms** - For spectral library matching
- **lance** - For creating Lance datasets
- **pyteomics** - For mzML file processing
- **pandas** - For data manipulation


### Data Files

- **GNPS spectral library** - Download from GNPS2 (see `data/libraries/README.md`)
- **Raw .raw files** - Download from MassIVE (see `data/README.md`)

---

## Step 1: Convert .raw Files and Extract Settings

The `run_process_raw.sh` script automatically handles:
- ✅ .raw → .mzML conversion (using ThermoRawFileParser)
- ✅ .raw → .mgf conversion (using ThermoRawFileParser)
- ✅ .raw → .csv conversion (instrument settings extraction using ScanHeadsman)

```bash
# Process all .raw files in MSV folders
cd /path/to/your/working/directory
sbatch slurm_scripts/data_preprocessing/run_process_raw.sh
```


**Configuration** (edit the script if needed):
```bash
# Default input directory (where MSV folders are located)
MSV_PARENT_DEFAULT="/path/to/your/working/directory/new_data"

# Paths to external tools
SCANHEADSMAN="/path/to/tools/ScanHeadsman/ScanHeadsman.exe"
THERMORAWFILEPARSER="/path/to/tools/ThermoRawFileParser/ThermoRawFileParser.exe"
```

**Requirements**:
- ThermoRawFileParser and ScanHeadsman must be installed (see `tools/README.md`)
- Mono runtime (on Linux/macOS) for running .NET executables



**Output Files**:

For each `.raw` file (e.g., `sample.raw`), the script creates:
- `sample.mzML` - Spectral data in mzML format
- `sample.mgf` - Spectral data in MGF format (for library matching)
- `sample.csv` - Instrument settings and scan metadata

---

## Step 2: Library Matching for Data Labeling

The `library_matching.sh` script labels MS2 spectra by matching them against the GNPS spectral library using the matchms library.

```bash
sbatch slurm_scripts/data_preprocessing/library_matching.sh
```

**What this script does**:
- Matches MS2 spectra from .mgf files against GNPS spectral libraries (positive and negative modes)
- Uses cosine similarity to find the best match for each spectrum
- Processes spectra separately by polarity (positive/negative ionization mode)

**Matching Criteria**:
- **Cosine similarity > 0.7**
- **Minimum 6 matching peaks**
- **Precursor m/z tolerance**: 0.05 Da

**Requirements**:
- GNPS libraries must be downloaded and split (see `data/libraries/README.md`)
- `.mgf` files from Step 1 must be available

**Configuration** (edit the script if needed):
```bash
python scripts/data_preprocessing/library_matching_diff_polarity.py \
    --msv_folder /path/to/your/working/directory/new_data \
    --reference_mgf_positive data/libraries/spectral_db_positive.mgf \
    --reference_mgf_negative data/libraries/spectral_db_negative.mgf \
    --output_tsv results/spectral_matching_results.tsv \
    --num_cpus $SLURM_CPUS_PER_TASK
```

**Output**:

TSV file containing with library matching results. 

## Step 3: Process Data and Create Lance Dataset

After library matching, process all MSV folders and create the final Lance dataset.

### Step 3a: Process Files (Prepare for Lance Creation)

```bash
sbatch slurm_scripts/data_preprocessing/run_processing_pipeline.sh
```

This script processes all MSV folders, organizes the data structure, and labels samples based on GNPS library matching: samples with matches are assigned Label=1, while unmatched samples are assigned Label=0. Finally, it prepares the files for Lance dataset creation. 

### Step 3b: Create Lance Dataset

```bash
sbatch slurm_scripts/data_preprocessing/run_build_lance.sh
```

This combines mzML spectra, instrument settings, and labels into an efficient Lance format.



**Configuration** (edit `run_build_lance.sh` if needed):

```bash
python scripts/data_preprocessing/create_lance_add_one_hot.py \
    --train_file_list data/file_paths/file_paths_train.txt \
    --val_file_list data/file_paths/file_paths_val.txt \
    --lance_uri results/lance_datasets \
    --train_table train_data \
    --val_table validation_data \
    --workers 16 \
    --cap_training_set 300000 \
    --cap_val_set 100000 \
    --training_set_csv data/metadata/train_datasets.csv \
    --val_set_csv data/metadata/val_datasets.csv
```

**Key Arguments:**

- `--train_file_list` / `--val_file_list`: Text files listing mzML files and their annotations (**User must create**)
- `--lance_uri`: Output directory for Lance dataset
- `--train_table` / `--val_table`: Table names within the Lance dataset
- `--workers`: Number of parallel workers
- `--cap_training_set` / `--cap_val_set`: Maximum spectra per split (optional)

**Creating File Lists (Required Before Running)**:

The `--train_file_list` and `--val_file_list` text files must be created manually by the user. Each line should contain the path to an `.mzML` file and its corresponding `_annotated.csv` file, **comma-separated**.

**File Format**:
```
/path/to/MSV000012345/folder1/sample1.mzML,/path/to/MSV000012345/folder1/sample1_annotated.csv
/path/to/MSV000012345/folder2/sample2.mzML,/path/to/MSV000012345/folder2/sample2_annotated.csv
/path/to/MSV000067890/folder1/sample3.mzML,/path/to/MSV000067890/folder1/sample3_annotated.csv
```



**How to Create**:

1. **Identify Dataset IDs**: Check `data/metadata/train_datasets.csv` or `data/metadata/val_datasets.csv` to find which MassIVE dataset IDs (e.g., MSV000084346) belong to training or validation split.

2. **Locate Processed Files**: After running `run_processing_pipeline.sh`, your processed data will be in `data/new_data/MSV*/` directories. Each dataset will contain:
   - `.mzML` files (spectral data)
   - `_annotated.csv` files (labels library matching and instrument setting columns)

3. **Create File Lists**: For each dataset ID in your split, find all `.mzML` files and create a comma-separated pair with their corresponding `_annotated.csv` file.

4. **Save to File**: Save the list to `data/file_paths/file_paths_train.txt` (for training) or `data/file_paths/file_paths_val.txt` (for validation).



**Note**: Ensure you only include dataset IDs that belong to the correct split (training or validation) based on  metadata CSV files.

### Lance Dataset Schema

Each row represents one MS2 scan with:

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `mz_array` | float32 | variable | MS1 m/z values |
| `intensity_array` | float32 | variable | MS1 intensity values |
| `precursor_mz` | float32 | scalar | MS2 precursor m/z |
| `instrument_settings` | float32 | [22] | Normalized + one-hot encoded settings |
| `label` | float32 | scalar | 1 (positive) or 0 (unlabeled) |
| `mzml_filepath` | string | - | Source file path |
| `ms2_scan_number` | int | scalar | Scan number in mzML |
| `dataset_id` | string | - | MassIVE accession (optional) |

### Feature Encoding

**Numerical features** (standardized using training set statistics):
```python
normalized = (value - mean) / std
```

**Categorical features** (one-hot encoded):
- Polarity: [negative, positive]
- Activation Type: [CID, HCD, ETD, etc.]

Final `instrument_settings` vector concatenates:
```
[normalized_numerical_features (N), one_hot_polarity (2), one_hot_activation (K)]
```

### Data Sampling Strategy

**Training set** (42 datasets):
- Sample up to 300,000 MS2 scans per dataset
- Random sampling to balance dataset contributions

**Validation set** (9 datasets):
- Sample up to 100,000 MS2 scans per dataset

**Test sets** (7, 5, 7 datasets):
- Sample up to 100,000 MS2 scans per dataset



