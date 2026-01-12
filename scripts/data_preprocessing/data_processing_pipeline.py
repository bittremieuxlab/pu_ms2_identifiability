#!/usr/bin/env python3
# Mass Spectrometry Data Processing Pipeline
# This script automates the processing of MS data files for model training preparation

import os
import sys
import pandas as pd
import logging
import argparse
import shutil
import re
from datetime import datetime
from typing import Set, List, Dict, Tuple, Optional, Union


# Configure logging
def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ms_pipeline_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("ms_pipeline")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Step 1: Move incomplete MSOrder files to invalid subfolder
def move_incomplete_msorder_files(root_path: str, logger: logging.Logger) -> int:
    """
    Recursively finds .csv files under root_path that do not contain '2' in the 'MSOrder' column
    and moves both the .csv and the corresponding .raw file to an 'invalid' subfolder in the same directory.
    """
    moved_count = 0
    logger.info("Starting movement of incomplete MSOrder files to invalid subfolder...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_path = os.path.join(dirpath, file)
                base_name = os.path.splitext(file)[0]
                raw_path = os.path.join(dirpath, base_name + ".raw")
                mgf_path = os.path.join(dirpath, base_name + ".mgf")
                mzml_path = os.path.join(dirpath, base_name + ".mzML")

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    logger.error(f"Could not read {csv_path}: {e}")
                    continue

                # Check if 'MSOrder' column exists and contains value 2
                if "MSOrder" in df.columns:
                    msorder_values = set(df["MSOrder"].dropna().unique())
                    if 2 not in msorder_values:
                        # Create invalid subfolder
                        invalid_dir = os.path.join(dirpath, "invalid")
                        os.makedirs(invalid_dir, exist_ok=True)
                        logger.debug(f"Created invalid subfolder: {invalid_dir}")

                        # Move CSV to invalid subfolder
                        csv_dest = os.path.join(invalid_dir, file)
                        shutil.move(csv_path, csv_dest)
                        logger.info(f"Moved CSV to invalid: {csv_path} → {csv_dest}")

                        # Move corresponding RAW file if it exists
                        if os.path.exists(raw_path):
                            raw_dest = os.path.join(invalid_dir, base_name + ".raw")
                            shutil.move(raw_path, raw_dest)
                            logger.info(f"Moved RAW to invalid: {raw_path} → {raw_dest}")
                        else:
                            logger.warning(f"RAW file not found for: {csv_path}")

                        if os.path.exists(mzml_path):
                            mzml_dest = os.path.join(invalid_dir, base_name + ".mzML")
                            shutil.move(mzml_path, mzml_dest)
                            logger.info(f"Moved MZML to invalid: {mzml_path}")

                        else:
                            logger.warning(f"MZML file not found for: {csv_path}")

                        if os.path.exists(mgf_path):
                            mgf_dest = os.path.join(invalid_dir, base_name + ".mgf")
                            shutil.move(mgf_path, mgf_dest)
                            logger.info(f"Moved MGF to invalid: {mgf_path}")
                        else:
                            logger.warning(f"MGF file not found for: {csv_path}")

                        moved_count += 1
                else:
                    # If MSOrder column is missing, treat as invalid and move
                    # Create invalid subfolder
                    invalid_dir = os.path.join(dirpath, "invalid")
                    os.makedirs(invalid_dir, exist_ok=True)
                    logger.debug(f"Created invalid subfolder: {invalid_dir}")

                    # Move CSV to invalid subfolder
                    csv_dest = os.path.join(invalid_dir, file)
                    shutil.move(csv_path, csv_dest)
                    logger.info(f"Moved CSV (missing MSOrder) to invalid: {csv_path} → {csv_dest}")

                    if os.path.exists(raw_path):
                        raw_dest = os.path.join(invalid_dir, base_name + ".raw")
                        shutil.move(raw_path, raw_dest)
                        logger.info(f"Moved RAW to invalid: {raw_path} → {raw_dest}")
                    else:
                        logger.warning(f"RAW file not found for: {csv_path}")
                    moved_count += 1

    logger.info(f"Finished moving files. Total files moved to invalid subfolders (CSV + RAW pairs): {moved_count}")
    logger.info("Invalid files are now organized in 'invalid' subfolders within their original directories")
    return moved_count


# Step 2: Check CSV files for required columns and proper MSOrder values
def check_csv_files(root_path: str, required_columns: List[str], logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """
    Recursively checks CSV files under the root_path for:
    1. MSOrder values (must contain both 1 and 2).
    2. Presence of required columns.
    """
    missing_msorder_files = []
    missing_columns_files = []
    msorder_incomplete_count = 0

    logger.info("Checking CSV files for required columns and MSOrder values...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".csv"):
                file_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    logger.error(f"Could not read {file_path}: {e}")
                    continue

                # Check if 'MSOrder' exists and has required values
                if "MSOrder" in df.columns:
                    unique_orders = set(df["MSOrder"].dropna().unique())
                    if not ({1, 2}.issubset(unique_orders)):
                        logger.warning(f"MSOrder incomplete in file: {file_path} — found: {unique_orders}")
                        missing_msorder_files.append(file_path)
                        msorder_incomplete_count += 1
                else:
                    logger.warning(f"Missing 'MSOrder' column in: {file_path}")
                    missing_msorder_files.append(file_path)

                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"{file_path} is missing columns: {missing_columns}")
                    missing_columns_files.append((file_path, missing_columns))

    logger.info(f"Number of files that do not have both MS1 and MS2 scans: {msorder_incomplete_count}")
    logger.info(f"Number of files missing required columns: {len(missing_columns_files)}")

    return missing_msorder_files, missing_columns_files


# Step 3: Standardize columns in CSV files
def standardize_columns(root_path: str, logger: logging.Logger) -> Dict[str, List[str]]:
    """
    Process CSV files to standardize column names and ensure all required columns exist.
    Handles different naming conventions for the same data fields.
    """
    # Standardized column mappings
    column_mappings = {
        "FT Resolution": "Orbitrap Resolution",
        "Orbitrap Resolution": "Orbitrap Resolution",
        "AGC Target": "AGC Target",
        "TargetAGC": "AGC Target",
        "HCD Energy [eV]": "HCD Energy V",
        "HCD Energy V": "HCD Energy V",
        "HCD Energy eV": "HCD Energy V",
        "HCD Energy": "HCD Energy",
        "Micro Scan Count": "Micro Scan Count",
        "Microscans": "Micro Scan Count",
        "NrLockMasses": "Number of Lock Masses",
        "Number of Lock Masses": "Number of Lock Masses",
        "LM Correction (ppm)": "LM m/z-Correction (ppm)",
        "LM m/z-Correction (ppm)": "LM m/z-Correction (ppm)"
    }

    # Columns that are always required
    # UPDATED: Added "Ionization" to base_columns
    base_columns = [
        "Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
        "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State", "Monoisotopic M/Z",
        "Ion Injection Time (ms)", "MS2 Isolation Width", "Conversion Parameter C",
        "LM Search Window (ppm)", "Number of LM Found", "Mild Trapping Mode",
        "Source CID eV", "SelectedMass1", "Activation1", "Energy1", "Orbitrap Resolution", "AGC Target",
        "HCD Energy V", "HCD Energy", "Micro Scan Count", "Number of Lock Masses", "LM m/z-Correction (ppm)", "label",
        "Ionization"
    ]

    # Track files with completely missing columns
    missing_columns_report = {col: [] for col in base_columns if not col.startswith("HCD Energy V(")}
    processed_count = 0
    skipped_count = 0

    logger.info("Starting column standardization process...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('.csv') and not file.endswith(('_processed.csv', '_annotated.csv')):
                csv_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(csv_path)

                    # Skip if DataFrame is empty
                    if df.empty:
                        logger.warning(f"Skipping empty CSV: {csv_path}")
                        skipped_count += 1
                        continue

                except Exception as e:
                    logger.error(f"Error reading {csv_path}: {e}")
                    skipped_count += 1
                    continue

                # Initialize with empty dummy label column if needed
                if 'label' not in df.columns:
                    df['label'] = pd.NA

                # Create a new DataFrame with standardized columns
                new_df = pd.DataFrame()

                # Ensure all base columns are present
                for col in base_columns:
                    # Try to find the column using the mapping
                    found = False
                    for original, standardized in column_mappings.items():
                        if standardized == col and original in df.columns:
                            new_df[col] = df[original]
                            found = True
                            break

                    # If not found through mapping, check if it exists directly
                    if not found and col in df.columns:
                        new_df[col] = df[col]
                        found = True

                    # If the column is still not found, track it and add empty column
                    if not found:
                        missing_columns_report[col].append(csv_path)
                        # Special handling for Ionization - we don't want to fill it with 0.0 yet
                        # if we want to detect it missing later, but standardizing usually fills defaults
                        new_df[col] = 0.0

                # Save the processed CSV
                processed_csv_path = os.path.join(dirpath, os.path.splitext(file)[0] + '_processed.csv')
                new_df.to_csv(processed_csv_path, index=False)
                logger.info(f"Processed and standardized columns: {processed_csv_path}")
                processed_count += 1

    # Log summary of column standardization
    logger.info(f"Column standardization complete. Processed {processed_count} files, skipped {skipped_count} files.")

    # Log completely missing columns
    for col, files in missing_columns_report.items():
        if files:
            logger.warning(f"Column '{col}' missing in {len(files)} files")
            for file_path in files[:5]:  # Log first 5 files only to avoid excessive logging
                logger.debug(f"  - {file_path}")
            if len(files) > 5:
                logger.debug(f"  - ... and {len(files) - 5} more files")

    return missing_columns_report


def _split_multi_value_column_into_five(df: pd.DataFrame, source_col: str, target_prefix: str,
                                        logger: logging.Logger) -> Tuple[pd.DataFrame, bool]:
    """Split a source column into five numeric columns, padding with zeros."""
    if source_col not in df.columns:
        return df, False

    # Prepare target columns
    target_cols = [f"{target_prefix}({i})" for i in range(1, 6)]
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0

    any_split = False

    for idx, value in enumerate(df[source_col]):
        if pd.isna(value):
            # leave zeros
            continue

        s = str(value).strip()
        # Normalize separators and remove brackets
        if '[' in s and ']' in s:
            s = s.strip('[]')
        s = s.replace(';', ',')

        parts = [p.strip() for p in s.split(',') if p.strip()] if ',' in s else [s]

        # Coerce to floats, ignore non-numeric as NaN
        numeric_vals = []
        for p in parts:
            try:
                numeric_vals.append(float(p))
            except Exception:
                # if cannot parse, treat as 0
                numeric_vals.append(0.0)

        # Truncate or pad with zeros to length 5
        numeric_vals = (numeric_vals + [0.0] * 5)[:5]

        for j in range(5):
            df.at[idx, target_cols[j]] = numeric_vals[j]

        any_split = True

    # Drop source column after split
    df = df.drop(columns=[source_col])
    return df, any_split


def split_hcd_energy_columns(root_path: str, logger: logging.Logger) -> int:
    """
    Split both 'HCD Energy V' and 'HCD Energy' into five columns each, padding with zeros.
    Creates columns: 'HCD Energy V(1..5)' and 'HCD Energy(1..5)'.
    """
    processed_count = 0
    skipped_count = 0

    logger.info("Starting HCD Energy V and HCD Energy column splitting into 5 columns...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(csv_path)

                    if df.empty:
                        logger.warning(f"Skipping empty CSV: {csv_path}")
                        skipped_count += 1
                        continue

                    df, did_v = _split_multi_value_column_into_five(df, 'HCD Energy V', 'HCD Energy V', logger)
                    df, did_e = _split_multi_value_column_into_five(df, 'HCD Energy', 'HCD Energy', logger)

                    if not (did_v or did_e):
                        logger.debug(f"No HCD Energy columns to split in: {csv_path}")
                        df.to_csv(csv_path, index=False)
                        continue

                    # Save the updated CSV
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Split HCD Energy columns in: {csv_path}")
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing HCD Energy columns in {csv_path}: {e}")
                    skipped_count += 1

    logger.info(
        f"HCD Energy column splitting complete. Processed {processed_count} files, skipped {skipped_count} files.")
    return processed_count


# Step 3.6: Transform categorical columns to binary values
def transform_categorical_columns(root_path: str, logger: logging.Logger) -> int:
    """
    Process CSV files to transform categorical columns to binary values:
    - "Polarity": Positive→1, Negative→0
    - "Mild Trapping Mode": On→1, Off→0
    - "Activation1": HCD→1, others→0
    - "Ionization": ESI→1, NSI→0

    Returns:
        int: Number of files processed
    """
    processed_count = 0
    skipped_count = 0

    logger.info("Starting categorical column transformation process...")

    # Define the categorical mappings
    categorical_mappings = {
        "Polarity": {
            "Positive": 1,
            "Negative": 0
        },
        "Mild Trapping Mode": {
            "On": 1,
            "Off": 0,
            "on": 1,
            "off": 0,
            "ON": 1,
            "OFF": 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        },
        "Activation1": {
            "HCD": 1,
            "hcd": 1,
            "Hcd": 1,
            "CID": 0,
            "cid": 0
        },
        # UPDATED: Added Ionization mapping
        "Ionization": {
            "ESI": 1,
            "esi": 1,
            "NSI": 0,
            "nsi": 0
        }
    }

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(csv_path)

                    if df.empty:
                        logger.warning(f"Skipping empty CSV: {csv_path}")
                        skipped_count += 1
                        continue

                    transformed_columns = []

                    # Process each categorical column
                    for column_name, mapping in categorical_mappings.items():
                        if column_name in df.columns:
                            # Create a copy of the original column for logging/checking
                            original_values = df[column_name].copy()

                            # Check if the column has any non-null values
                            if original_values.isna().all():
                                # Even if empty, we might want to fill with 0, but let's log warning
                                logger.warning(
                                    f"Column '{column_name}' is completely empty in {csv_path}")
                                # We still proceed to try and map/fill defaults below

                            # UPDATED: Specific check for Ionization (and others) to report unexpected values
                            # Convert to string to check against mapping keys
                            current_vals_str = original_values.astype(str).unique()
                            # Get valid keys (keys in our mapping dict)
                            valid_keys = set(str(k) for k in mapping.keys())
                            # Include '0.0' or '0' if it was filled by standardization
                            valid_keys.update(['0.0', '0', 'nan', '<NA>', 'None'])

                            unexpected_values = []
                            for val in current_vals_str:
                                if val not in valid_keys:
                                    unexpected_values.append(val)

                            if unexpected_values:
                                # This handles the requirement: "print the filepaths of those csvs and say which value Ionization col has"
                                logger.warning(f"FILE: {csv_path} contains unexpected values for '{column_name}': {unexpected_values}")

                            # Apply the transformation
                            df[column_name] = df[column_name].astype(str).map(mapping)

                            # Fill NaN values (unmapped values) with 0
                            df[column_name] = df[column_name].fillna(0)

                            # Convert to integer type
                            df[column_name] = df[column_name].astype(int)

                            transformed_columns.append(column_name)
                        else:
                            # Ionization might be missing if not in original CSV (though standardize ensures it exists as 0.0)
                            logger.warning(f"Column '{column_name}' not found in {csv_path}")

                    # Save the updated CSV
                    df.to_csv(csv_path, index=False)

                    if transformed_columns:
                        logger.info(f"Transformed categorical columns {transformed_columns} in: {csv_path}")
                        processed_count += 1
                    else:
                        logger.debug(f"No categorical columns to transform in: {csv_path}")
                        processed_count += 1

                except Exception as e:
                    logger.error(f"Error transforming categorical columns in {csv_path}: {e}")
                    skipped_count += 1

    logger.info(
        f"Categorical column transformation complete. Processed {processed_count} files, skipped {skipped_count} files.")
    return processed_count


def filter_msorder_rows(root_path: str, logger: logging.Logger) -> int:
    """
    Process CSV files to filter rows and keep only those where MSOrder = 2.
    Modifies the '_processed.csv' files in-place instead of creating new files.
    """
    processed_count = 0
    skipped_count = 0

    logger.info("Starting MSOrder filtering process (keeping only MSOrder = 2)...")

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(dirpath, file)

                try:
                    df = pd.read_csv(csv_path)

                    if df.empty:
                        logger.warning(f"Skipping empty CSV: {csv_path}")
                        skipped_count += 1
                        continue

                    if 'MSOrder' not in df.columns:
                        logger.warning(f"Skipping {csv_path}: 'MSOrder' column not found")
                        skipped_count += 1
                        continue

                    if df['MSOrder'].isna().all():
                        logger.warning(f"Skipping {csv_path}: 'MSOrder' column is completely empty")
                        skipped_count += 1
                        continue

                    rows_before = len(df)

                    # Filter to keep only rows where MSOrder = 2
                    df_filtered = df[df['MSOrder'] == 2].copy()
                    rows_after = len(df_filtered)

                    if rows_after == 0:
                        logger.warning(f"No rows with MSOrder = 2 found in {csv_path}")
                        skipped_count += 1
                        continue

                    df_filtered.to_csv(csv_path, index=False)
                    logger.info(f"Filtered {csv_path}: {rows_before} → {rows_after} rows (MSOrder = 2)")
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error filtering MSOrder rows in {csv_path}: {e}")
                    skipped_count += 1

    logger.info(f"MSOrder filtering complete. Processed {processed_count} files, skipped {skipped_count} files.")
    logger.info(f"Files modified in-place: '_processed.csv' files now contain only MSOrder = 2 rows")
    return processed_count


# Step 4: Label data using TSV information
def label_data(root_path: str, tsv_path: str, logger: logging.Logger) -> tuple[int, int]:
    """
    Process CSV files to add labels based on TSV data matching.
    """
    if not os.path.exists(tsv_path):
        logger.error(f"TSV file not found: {tsv_path}")
        return 0, 0

    logger.info(f"Loading TSV data from {tsv_path}")
    try:
        tsv_cols = ['scan_number', 'mgf_path', 'compound_name']
        tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=lambda c: c in tsv_cols)
        if 'compound_name' not in tsv_data.columns:
            tsv_data = pd.read_csv(tsv_path, sep='\t', usecols=['scan', 'full_CCMS_path'])
            tsv_data['compound_name'] = pd.NA
        tsv_data['scan_number'] = tsv_data['scan_number'].astype(int)
        logger.info(f"Loaded {len(tsv_data)} entries from TSV file")
    except Exception as e:
        logger.error(f"Failed to load TSV file: {e}")
        return 0, 0

    labeled_count = 0
    positive_samples_count = 0

    for dirpath, _, filenames in os.walk(root_path):
        if '/invalid' in dirpath or '\\invalid' in dirpath:
            logger.debug(f"Skipping invalid directory: {dirpath}")
            continue
        for file in filenames:
            if file.endswith('_processed.csv'):
                csv_path = os.path.join(dirpath, file)

                try:
                    csv_data = pd.read_csv(csv_path)

                    if 'Scan' not in csv_data.columns:
                        logger.warning(f"Skipping labeling for {csv_path}: 'Scan' column not found")
                        continue

                    csv_data['Scan'] = csv_data['Scan'].astype(int)
                    base_name = os.path.basename(csv_path).replace('_processed.csv', '')
                    matching_scans = tsv_data[tsv_data['mgf_path'].str.contains(base_name, regex=False, na=False)]
                    matching_scan_values = matching_scans['scan_number'].values if not matching_scans.empty else []

                    scan_to_name = {}
                    if not matching_scans.empty and 'compound_name' in matching_scans.columns:
                        for _, row in matching_scans.iterrows():
                            sc = int(row['scan_number'])
                            if sc not in scan_to_name:
                                scan_to_name[sc] = row.get('compound_name', pd.NA)

                    csv_data['label'] = csv_data['Scan'].apply(lambda x: 1 if x in matching_scan_values else 0)
                    positive_in_file = (csv_data['label'] == 1).sum()
                    positive_samples_count += positive_in_file

                    if 'Compound_name' not in csv_data.columns:
                        csv_data['Compound_name'] = pd.NA
                    csv_data['Compound_name'] = csv_data['Scan'].map(scan_to_name).where(csv_data['label'] == 1,
                                                                                         other=pd.NA)

                    annotated_csv_path = csv_path.replace('_processed.csv', '_annotated_21_12.csv')
                    csv_data.to_csv(annotated_csv_path, index=False)
                    logger.info(f"Labeled data saved to: {annotated_csv_path} (positive samples: {positive_in_file})")
                    labeled_count += 1

                except Exception as e:
                    logger.error(f"Error labeling {csv_path}: {e}")

    logger.info(f"Labeling complete. Successfully labeled {labeled_count} files with {positive_samples_count} positive samples.")
    return labeled_count, positive_samples_count


# Step 5: Organize processed files
def organize_processed_files(root_path: str, output_dir: str, logger: logging.Logger) -> int:
    """
    Copies processed and annotated files to the output directory,
    preserving the folder structure.
    """
    copied_count = 0

    logger.info(f"Organizing processed files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    copy_tasks = []

    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('_annotated_21_12.csv') :
                source_path = os.path.join(dirpath, file)

                rel_path = os.path.relpath(dirpath, root_path)
                dest_dir = os.path.join(output_dir, rel_path)
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, file)
                copy_tasks.append((source_path, dest_path))

                base_name = os.path.splitext(file.replace('_annotated_21_12', '').replace('_processed_annotated', ''))[0]
                # Try handling both naming conventions if present
                base_name_clean = base_name.replace('_processed', '')

                raw_path = os.path.join(dirpath, base_name_clean + ".raw")
                if os.path.exists(raw_path):
                    raw_dest_path = os.path.join(dest_dir, base_name_clean + ".raw")
                    copy_tasks.append((raw_path, raw_dest_path))

    for src, dst in copy_tasks:
        try:
            shutil.copy2(src, dst)
            copied_count += 1
            logger.debug(f"Copied: {src} -> {dst}")
        except Exception as e:
            logger.error(f"Failed to copy {src}: {e}")

    logger.info(f"Successfully copied {copied_count} processed files to {output_dir}")
    return copied_count


# Step 6: Generate summary report
def generate_summary(root_path: str, output_dir: str, moved_count: int,
                     missing_msorder: List[str], missing_columns_report: Dict[str, List[str]],
                     processed_count: int, hcd_energy_split_count: int, categorical_transform_count: int,
                     msorder_filter_count: int, labeled_count: int, positive_samples_count: int, copied_count: int,
                     logger: logging.Logger) -> None:
    """Generate a summary report of the pipeline execution."""

    missing_cols_count = {}
    for col, files in missing_columns_report.items():
        missing_cols_count[col] = len(files)

    summary = {
        "Files Processed": {
            "Total CSV files found": sum(1 for _ in find_files_by_extension(root_path, ".csv")),
            "Files moved to invalid subfolder (incomplete MSOrder)": moved_count,
            "Files with missing MSOrder values": len(missing_msorder),
            "Files with standardized columns and MSOrder filtering": processed_count,
            "Files with HCD Energy V columns split": hcd_energy_split_count,
            "Files with categorical columns transformed": categorical_transform_count,
            "Files filtered to MSOrder = 2": msorder_filter_count,
            "Files labeled with TSV data": labeled_count,
            "Number of samples labeled as 1": positive_samples_count,
            "Final files copied to output directory": copied_count
        },
        "Missing Columns Summary": missing_cols_count,
        "Pipeline Execution": {
            "Input directory": os.path.abspath(root_path),
            "Output directory": os.path.abspath(output_dir),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    logger.info("Pipeline Execution Summary:")
    for section, items in summary.items():
        logger.info(f"--- {section} ---")
        for key, value in items.items():
            logger.info(f"  {key}: {value}")


# Utility function to find files by extension
def find_files_by_extension(root_path: str, extension: str) -> List[str]:
    """Find all files with the given extension in the root_path directory tree."""
    result = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(extension):
                result.append(os.path.join(dirpath, file))
    return result


# Main pipeline function
def run_pipeline(root_path: str, output_dir: str, tsv_path: str, required_columns: List[str],
                 logger: logging.Logger) -> None:
    """Run the complete MS data processing pipeline."""
    logger.info(f"Starting MS data processing pipeline on {root_path}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Move incomplete MSOrder files to invalid subfolder
    moved_count = move_incomplete_msorder_files(root_path, logger)

    # Step 2: Check remaining CSV files for quality
    missing_msorder, missing_columns_list = check_csv_files(root_path, required_columns, logger)

    # Step 3: Standardize columns in CSV files
    missing_columns_report = standardize_columns(root_path, logger)

    # Step 3.5: Split HCD Energy V column into three separate columns
    hcd_energy_split_count = split_hcd_energy_columns(root_path, logger)

    # Step 3.6: Transform categorical columns to binary values
    categorical_transform_count = transform_categorical_columns(root_path, logger)

    # Step 3.7: Filter rows to keep only MSOrder = 2
    msorder_filter_count = filter_msorder_rows(root_path, logger)

    # Step 4: Label data using TSV information
    labeled_count, positive_samples_count = label_data(root_path, tsv_path, logger)

    # Step 5: Organize processed files
    copied_count = organize_processed_files(root_path, output_dir, logger)

    # Step 6: Generate summary
    generate_summary(
        root_path=root_path,
        output_dir=output_dir,
        moved_count=moved_count,
        missing_msorder=missing_msorder,
        missing_columns_report=missing_columns_report,
        processed_count=sum(1 for _ in find_files_by_extension(root_path, "_processed_annotated.csv")),
        hcd_energy_split_count=hcd_energy_split_count,
        categorical_transform_count=categorical_transform_count,
        msorder_filter_count=msorder_filter_count,
        labeled_count=labeled_count,
        positive_samples_count= positive_samples_count,
        copied_count=copied_count,
        logger=logger
    )

    logger.info("Pipeline execution completed successfully!")


def main():
    """Main entry point of the script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="MS Data Processing Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing MS data files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed files")
    parser.add_argument("--tsv", "-t", required=True, help="Path to the TSV file for labeling")
    parser.add_argument("--log-dir", "-l", default="logs_11_00", help="Directory for log files")
    args = parser.parse_args()

    # Define required columns for MS data
    required_columns = [
        "Scan", "MSOrder", "Polarity", "RT [min]", "LowMass", "HighMass",
        "TIC", "BasePeakPosition", "BasePeakIntensity", "Charge State",
        "Monoisotopic M/Z", "Ion Injection Time (ms)", "MS2 Isolation Width",
        "Conversion Parameter C", "LM Search Window (ppm)", "Number of LM Found",
        "Mild Trapping Mode", "Source CID eV", "SelectedMass1", "Activation1",
        "Energy1", "Orbitrap Resolution", "AGC Target",
        "HCD Energy V(1)", "HCD Energy V(2)", "HCD Energy V(3)", "HCD Energy V(4)", "HCD Energy V(5)",
        "HCD Energy(1)", "HCD Energy(2)", "HCD Energy(3)", "HCD Energy(4)", "HCD Energy(5)",
        "Micro Scan Count", "Number of Lock Masses", "LM m/z-Correction (ppm)",
        "Ionization" # Added Ionization to required list for main execution
    ]

    # Set up logging
    logger = setup_logging(args.log_dir)

    try:
        # Run the pipeline
        run_pipeline(
            root_path=args.input,
            output_dir=args.output,
            tsv_path=args.tsv,
            required_columns=required_columns,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()