import os
import pandas as pd
import ppx
import argparse
import time


def download_with_retry(proj, raw_files, ds_id, local_dir, max_retries=3):
    """Download files with retry logic and connection refresh."""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries} for {ds_id}")
            proj.download(raw_files)
            print(f"Successfully downloaded {len(raw_files)} RAW files to {local_dir}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                # Recreate project object to refresh connection
                try:
                    proj = ppx.find_project(ds_id, local=local_dir, timeout=600)
                except Exception as conn_error:
                    print(f"Failed to refresh connection: {conn_error}")
            else:
                print(f"All {max_retries} attempts failed for {ds_id}")
                raise
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MSV datasets from datasets.csv (training_datasets, validation_datasets, test_datasets)"
    )
    parser.add_argument("--new_csv", required=True, help="Path to datasets.csv")
    parser.add_argument(
        "--base_dir", required=True, help="Base directory where MSV folders will be created"
    )
    parser.add_argument(
        "--download_limit", type=int, default=2, help="Maximum number of datasets to download"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Maximum retry attempts per dataset"
    )
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)


    # Read input CSV
    df_new = pd.read_csv(args.new_csv, sep=";")

    # Clean column names
    df_new.columns = df_new.columns.str.strip()

    # Extract unique Dataset IDs from the new CSV
    dataset_ids = df_new["dataset_id"].dropna().unique()

    download_count = 0
    failed_datasets = []

    for ds_id in dataset_ids:
        if download_count >= args.download_limit:
            print("\nReached download limit. Stopping.")
            break

        print(f"\nProcessing {ds_id}...")

        try:
            # Create project object with longer timeout
            proj = ppx.find_project(ds_id, timeout=600)

            # Check for RAW files
            raw_files = proj.remote_files("*.raw")
            if not raw_files:
                print(f"No RAW files found for {ds_id}")
                continue

            print(f"Found {len(raw_files)} RAW files for {ds_id}")

            # Create local directory for this dataset
            local_dir = os.path.join(args.base_dir, ds_id)
            os.makedirs(local_dir, exist_ok=True)

            # Reinitialize project with local directory
            proj = ppx.find_project(ds_id, local=local_dir, timeout=600)

            # Download with retry logic
            success = download_with_retry(proj, raw_files, ds_id, local_dir, args.max_retries)

            if not success:
                failed_datasets.append(ds_id)
                continue

            download_count += 1

        except Exception as e:
            print(f"Error processing {ds_id}: {str(e)}")
            failed_datasets.append(ds_id)
            continue

    print(f"\n{'=' * 60}")
    print(f"Download Summary:")
    print(f"Successfully downloaded: {download_count} datasets")
    if failed_datasets:
        print(f"Failed datasets ({len(failed_datasets)}): {', '.join(failed_datasets)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()