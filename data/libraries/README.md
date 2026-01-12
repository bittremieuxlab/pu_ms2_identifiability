# GNPS Spectral Libraries

This directory should contain GNPS spectral library files for library matching, separated by ionization mode.

## Required Files

After downloading and splitting, you'll have:

```
data/libraries/
├── cleaned_spectra.mgf                  # Downloaded GNPS library (both polarities)
├── spectral_db_positive.mgf     # Positive ionization mode library (created by split script)
└── spectral_db_negative.mgf     # Negative ionization mode library (created by split script)
```

## Step-by-Step Guide

### Step 1: Download GNPS Library

Download the  GNPS library (contains both positive and negative spectra):

**Download from GNPS2 Library Page**

1. Visit the GNPS2 Spectral Libraries page: [https://external.gnps2.org/gnpslibrary](https://external.gnps2.org/gnpslibrary)
2. Scroll to the **"Preprocessed Data"** section at the bottom of the page
3. Find the dataset processed with the **matchms** pipeline
4. Click **"MGF Download"** to download the library file
5. Rename the downloaded file to `cleaned_spectra.mgf` and place it in `data/libraries/`

```bash
cd data/libraries/


# Verify the file
ls -lh cleaned_spectra.mgf
```




### Step 2: Split Library by Polarity

Use the provided script to automatically split the library into positive and negative mode files:

```bash
# From the project root directory
python scripts/data_preprocessing/split_library.py
```

**What this script does**:
- Reads `data/libraries/cleaned_spectra.mgf` (or your downloaded library file)
- Filters spectra by `ionmode` metadata field
- Creates two files:
  - `spectral_db_positive.mgf` - All positive mode spectra
  - `spectral_db_negative.mgf` - All negative mode spectra


