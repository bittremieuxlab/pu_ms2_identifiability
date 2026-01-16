import os
import pandas as pd
import matchms
from matchms.importing import load_from_mgf
from matchms.filtering import (
    default_filters,
    normalize_intensities,
    select_by_intensity,
    select_by_mz,
)
from matchms.similarity import PrecursorMzMatch, CosineGreedy
from matchms import calculate_scores
from matchms.logging_functions import set_matchms_logger_level
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Silence matchms warnings
set_matchms_logger_level("ERROR")


def is_valid_spectrum(spectrum):
    """Check if a spectrum has non-empty peaks after processing."""
    return spectrum is not None and spectrum.peaks is not None and len(spectrum.peaks.mz) > 0


def peak_processing(spectrum):
    """Clean and filter a spectrum."""
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


def get_scan_number(spectrum, index):
    """Extract scan number from spectrum metadata with fallback options."""
    metadata = spectrum.metadata
    scan_keys = [
        "scans",
        "scan",
        "scan_number",
        "scannumber",
        "ms_level",
        "spectrum_id",
        "spectrumid",
        "id",
    ]
    for key in scan_keys:
        if key in metadata and metadata[key] is not None:
            val = metadata[key]
            if isinstance(val, (int, float)):
                return int(val)
            elif isinstance(val, str):
                import re

                numbers = re.findall(r"\d+", val)
                if numbers:
                    return int(numbers[0])
    return index


def get_polarity_from_csv(mgf_path):
    """
    Looks for a CSV file with the same name as the MGF.
    Reads the 'Polarity' column to determine if it is Positive or Negative.
    """
    # Construct expected CSV path: /path/to/file.mgf -> /path/to/file.csv
    base, _ = os.path.splitext(mgf_path)
    csv_path = base + ".csv"

    if not os.path.exists(csv_path):
        print(f"Warning: Corresponding CSV not found for {mgf_path}")
        return None

    try:
        # Read only the first few rows to check polarity
        df = pd.read_csv(csv_path, nrows=5)

        # Normalize column names to handle case sensitivity
        df.columns = [c.strip() for c in df.columns]

        if "Polarity" not in df.columns:
            print(f"Warning: 'Polarity' column not found in {csv_path}")
            return None

        # Get the first value
        polarity_val = str(df["Polarity"].iloc[0]).strip().lower()

        if "positive" in polarity_val or "pos" in polarity_val:
            return "Positive"
        elif "negative" in polarity_val or "neg" in polarity_val:
            return "Negative"
        else:
            print(f"Warning: Unknown polarity value '{polarity_val}' in {csv_path}")
            return None

    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None


def spectral_library_matching(
    query_mgf, db_spectra, parent_mz_tol=2, msms_mz_tol=0.5, min_cos=0.7, min_peaks=6
):
    """Perform spectral library matching for a single MGF file."""
    spectra_query = list(load_from_mgf(query_mgf))
    if not spectra_query:
        print(f"Warning: No spectra found in {query_mgf}")
        return pd.DataFrame()

    # Clean/filter spectra
    spectra_query = [peak_processing(s) for s in spectra_query if s is not None]
    spectra_query = [s for s in spectra_query if is_valid_spectrum(s)]

    # Prepare scoring objects
    precursor_match = PrecursorMzMatch(tolerance=parent_mz_tol, tolerance_type="Dalton")
    chunks_query = [spectra_query[x : x + 1000] for x in range(0, len(spectra_query), 1000)]

    # cosine = CosineGreedy(tolerance=msms_mz_tol)

    results = []
    for chunk in chunks_query:
        scores = calculate_scores(chunk, db_spectra, precursor_match)
        idx_row = scores.scores[:, :][0]
        idx_col = scores.scores[:, :][1]
        import numpy as np

        scans_id_map = {i: int(s.metadata["scans"]) for i, s in enumerate(chunk)}
        cosine = CosineGreedy(tolerance=msms_mz_tol)


        for x, y in zip(idx_row, idx_col):
            if x < y:
                cos_score, n_matches = cosine.pair(chunk[x], db_spectra[y])[()]
                if cos_score >= min_cos and n_matches >= min_peaks:
                    results.append(
                        {
                            "msms_score": cos_score,
                            "matched_peaks": n_matches,
                            "scan_number": scans_id_map[x],
                            "reference_id": y + 1,
                            "adduct": db_spectra[y].get("precusortype"),
                            "charge": db_spectra[y].get("charge"),
                            "ionmode": db_spectra[y].get("ionmode"),
                            "instrument": db_spectra[y].get("instrument"),
                            "instrument_type": db_spectra[y].get("instrumenttype"),
                            "comment": db_spectra[y].get("comment"),
                            "inchikey": db_spectra[y].get("inchikey"),
                            "inchi": (db_spectra[y].get("inchi") or "").strip("\"'"),
                            "smiles": db_spectra[y].get("smiles"),
                            "compound_name": db_spectra[y].get("compound_name"),
                            "mgf_path": query_mgf,  # Add file path info
                        }
                    )
    if results:
        df = pd.DataFrame(results)
        df["Spectral_library_ID"] = df["comment"].str.extract(r"DB#=(.*?);")
        df["Spectral_library"] = df["comment"].str.extract(r"origin=(.*?)(?:;|$)")
        df = df.loc[df.groupby("scan_number")["msms_score"].idxmax()].reset_index(drop=True)
    else:
        df = pd.DataFrame(
            columns=[
                "msms_score",
                "matched_peaks",
                "feature_id",
                "reference_id",
                "adduct",
                "charge",
                "ionmode",
                "instrument",
                "instrument_type",
                "comment",
                "inchikey",
                "inchi",
                "smiles",
                "compound_name",
                "Spectral_library_ID",
                "Spectral_library",
                "mgf_path",
            ]
        )
    return df


# Top-level worker function
def worker(
    mgf_info, db_spectra_pos, db_spectra_neg, parent_mz_tol, msms_mz_tol, min_cos, min_peaks
):
    mgf_path, rel_path = mgf_info

    # Determine Polarity from CSV
    polarity = get_polarity_from_csv(mgf_path)

    if polarity == "Positive":
        print(f"Processing {rel_path} (Positive Mode)...")
        target_db = db_spectra_pos
    elif polarity == "Negative":
        print(f"Processing {rel_path} (Negative Mode)...")
        target_db = db_spectra_neg
    else:
        print(f"Skipping {rel_path}: Could not determine polarity or CSV missing.")
        return pd.DataFrame()  # Return empty DF

    if not target_db:
        print(f"Error: Target database for {polarity} is empty.")
        return pd.DataFrame()

    df = spectral_library_matching(
        mgf_path,
        target_db,
        parent_mz_tol=parent_mz_tol,
        msms_mz_tol=msms_mz_tol,
        min_cos=min_cos,
        min_peaks=min_peaks,
    )
    if not df.empty:
        df["mgf_path"] = rel_path
        df["Polarity"] = polarity
    return df


def process_folder_parallel(
    msv_parent, reference_mgf_pos, reference_mgf_neg, output_file, num_cpus=None
):
    # Load both libraries
    print(f"Loading Positive Library: {reference_mgf_pos}")
    db_spectra_pos = list(load_from_mgf(reference_mgf_pos))
    print(f"Loaded {len(db_spectra_pos)} positive spectra.")

    print(f"Loading Negative Library: {reference_mgf_neg}")
    db_spectra_neg = list(load_from_mgf(reference_mgf_neg))
    print(f"Loaded {len(db_spectra_neg)} negative spectra.")

    all_results = []

    # Gather all mgf paths
    mgf_files = []
    for root, dirs, files in os.walk(msv_parent):
        for file in files:
            if file.endswith(".mgf"):
                mgf_path = os.path.join(root, file)
                rel_path = os.path.relpath(mgf_path, start=os.path.dirname(msv_parent))
                mgf_files.append((mgf_path, rel_path))

    print(f"Found {len(mgf_files)} MGF files to process.")

    # Set number of CPUs
    import multiprocessing

    if num_cpus is None:
        num_cpus = max(1, multiprocessing.cpu_count() - 2)

    print(f"Using {num_cpus} CPU cores for parallel processing.")

    # Partial function to pass databases and params
    worker_partial = partial(
        worker,
        db_spectra_pos=db_spectra_pos,
        db_spectra_neg=db_spectra_neg,
        parent_mz_tol=0.05,
        msms_mz_tol=0.05,
        min_cos=0.7,
        min_peaks=6,
    )

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(worker_partial, mgf_info) for mgf_info in mgf_files]
        for future in as_completed(futures):
            try:
                df_result = future.result()
                if not df_result.empty:
                    all_results.append(df_result)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error processing a file: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, sep="\t", index=False)
    print(f"All results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel spectral library matching with Polarity detection"
    )
    parser.add_argument("--msv_folder", required=True, help="Path to folder with MGF files")
    parser.add_argument(
        "--reference_mgf_positive", required=True, help="Path to Positive spectral library MGF"
    )
    parser.add_argument(
        "--reference_mgf_negative", required=True, help="Path to Negative spectral library MGF"
    )
    parser.add_argument("--output_tsv", required=True, help="Output TSV file path")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPU cores to use")

    args = parser.parse_args()

    process_folder_parallel(
        msv_parent=args.msv_folder,
        reference_mgf_pos=args.reference_mgf_positive,
        reference_mgf_neg=args.reference_mgf_negative,
        output_file=args.output_tsv,
        num_cpus=args.num_cpus,
    )
