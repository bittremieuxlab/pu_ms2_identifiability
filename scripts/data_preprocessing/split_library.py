import os
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf

# =================CONFIGURATION=================
# Path to your downloaded GNPS library file (combined, both polarities)
reference_mgf = "data/libraries/cleaned_spectra.mgf"

# Output file names (will be created in the same directory)
output_positive = "data/libraries/spectral_db_positive.mgf"
output_negative = "data/libraries/spectral_db_negative.mgf"

# NOTE: Update these paths to match your file locations if needed
# ===============================================

def filter_spectra_by_ion_mode(spectra, ion_mode_start):
    """
    Filters a list of spectra based on the ionmode metadata field.
    """
    filtered_spectra = []
    for spectrum in spectra:
        metadata = spectrum.metadata
        # Check if metadata exists, has 'ionmode', and starts with the requested string
        if metadata and 'ionmode' in metadata and metadata.get('ionmode', '').lower().startswith(
                ion_mode_start.lower()):
            filtered_spectra.append(spectrum)

    return filtered_spectra


def main():
    # 1. Check if input file exists
    if not os.path.exists(reference_mgf):
        print(f"Error: The file '{reference_mgf}' was not found.")
        return

    # 2. Load the spectra
    print(f"Loading spectra from: {reference_mgf}...")
    # We convert to a list immediately so we can iterate over it multiple times
    db_spectra = list(load_from_mgf(reference_mgf))
    print(f"Total spectra loaded: {len(db_spectra)}")

    print("-" * 30)

    # 3. Filter and Save POSITIVE
    print("Processing Positive Mode...")
    positive_spectra = filter_spectra_by_ion_mode(db_spectra, "positive")

    if len(positive_spectra) > 0:
        save_as_mgf(positive_spectra, output_positive)
        print(f"  -> Saved {len(positive_spectra)} spectra to '{output_positive}'")
    else:
        print("  -> No positive spectra found. File not created.")

    print("-" * 30)

    # 4. Filter and Save NEGATIVE
    print("Processing Negative Mode...")
    negative_spectra = filter_spectra_by_ion_mode(db_spectra, "negative")

    if len(negative_spectra) > 0:
        save_as_mgf(negative_spectra, output_negative)
        print(f"  -> Saved {len(negative_spectra)} spectra to '{output_negative}'")
    else:
        print("  -> No negative spectra found. File not created.")

    print("-" * 30)
    print("Done.")


if __name__ == "__main__":
    main()