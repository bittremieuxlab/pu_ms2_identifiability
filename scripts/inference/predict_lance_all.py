import os
import sys
import argparse
import logging
import numpy as np
import torch
import lance
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import multiprocessing as mp

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Critical Model Import ---
try:
    from src.transformers.model_nn_pu_loss_detach_diff_polarity import SimpleSpectraTransformer
except ImportError:
    print("ERROR: Could not import SimpleSpectraTransformer.")
    print(
        "Please run this script from your project's root directory or ensure 'src' is in your PYTHONPATH."
    )
    sys.exit(1)

# --- Logging Setup (Keep unchanged) ---
RANK = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))
logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {RANK}] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
script_logger = logging.getLogger(__name__)


# --- End Logging Setup ---


class LanceIndexDataset(Dataset):
    """(Keep unchanged)"""

    def __init__(self, lance_dataset_path: str, rank: str = "0"):
        super().__init__()
        self.lance_dataset_path = lance_dataset_path
        if not os.path.exists(lance_dataset_path):
            script_logger.error(f"Dataset not found at: {lance_dataset_path}")
            raise FileNotFoundError(f"Lance dataset not found at: {lance_dataset_path}")
        ds = lance.dataset(self.lance_dataset_path)
        self.length = ds.count_rows()
        script_logger.info(
            f"LanceIndexDataset (rank {rank}) initialized for {self.lance_dataset_path}. Length: {self.length}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class LanceBatchCollator:
    """(Keep unchanged)"""

    def __init__(self, lance_dataset_path: str, rank: str = "0"):
        self.lance_dataset_path = lance_dataset_path
        self.ds = None
        self.rank = rank
        self.log_count = 0

    def __call__(self, batch_indices: list[int]) -> dict:
        if self.ds is None:
            self.ds = lance.dataset(self.lance_dataset_path)
            script_logger.info(f"Collate (rank {self.rank}) opened dataset handle.")

        if self.log_count < 1:
            script_logger.info(
                f"Collate (rank {self.rank}) fetching indices (first 10): {batch_indices[:10]}"
            )
            self.log_count += 1

        data_batch = self.ds.take(batch_indices).to_pydict()
        batch_size = len(batch_indices)

        mz_arrays = [torch.tensor(arr, dtype=torch.float32) for arr in data_batch["mz_array"]]
        intensity_arrays = [
            torch.tensor(arr, dtype=torch.float32) for arr in data_batch["intensity_array"]
        ]

        instrument_settings = torch.tensor(data_batch["instrument_settings"], dtype=torch.float32)
        labels = torch.tensor(data_batch["label"], dtype=torch.float32).view(-1, 1)
        precursor_mz = torch.tensor(data_batch["precursor_mz"], dtype=torch.float32).view(-1, 1)

        max_len = max(mz.shape[0] for mz in mz_arrays)
        mz_padded = torch.zeros(batch_size, max_len)
        intensity_padded = torch.zeros(batch_size, max_len)

        for i in range(batch_size):
            length = mz_arrays[i].shape[0]
            mz_padded[i, :length] = mz_arrays[i]
            intensity_padded[i, :length] = intensity_arrays[i]

        dataset_ids = data_batch.get("dataset_id", None)

        batch_dict = {
            "mz": mz_padded,
            "intensity": intensity_padded,
            "instrument_settings": instrument_settings,
            "labels": labels,
            "precursor_mz": precursor_mz,
        }
        if dataset_ids is not None:
            batch_dict["dataset_id"] = dataset_ids
        return batch_dict


def select_all_samples(args):
    """
    Modified to run predictions on ALL samples and save index, probability, and label.
    """
    pl.seed_everything(1, workers=True)

    # 1. Load Model (Keep unchanged)
    script_logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = SimpleSpectraTransformer.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 2. Load Lance Dataset & Prepare ALL Indices
    script_logger.info(f"Opening Lance dataset at: {args.lance_path}")
    # Note: Using args.lance_path for clarity instead of old args.train_lance_path

    base_index_dataset = LanceIndexDataset(args.lance_path, rank=RANK)
    total_rows = len(base_index_dataset)
    all_indices = np.arange(total_rows)  # Get indices for ALL samples

    # We need all labels for the final output, fetch them once.
    script_logger.info("Fetching all labels...")
    ds = lance.dataset(args.lance_path)
    labels_table = ds.to_table(columns=["label"])
    all_labels = labels_table["label"].to_numpy()

    script_logger.info(f"Total samples to process: {total_rows}")

    # 3. Create Dataset/Loader
    # Use the base_index_dataset directly (no need for Subset)
    collator = LanceBatchCollator(args.lance_path, rank=RANK)

    dataloader = DataLoader(
        base_index_dataset,  # Process all indices
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # 4. Run predictions on ALL samples
    probabilities = []
    script_logger.info("Running predictions on all samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            logits = model(batch)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().numpy())

    # 5. Collate Results (Index, Probability, Label)
    # The order of predictions matches the order of 'all_indices' (0 to N-1)
    probabilities = [p[0] for p in probabilities]

    results_df = pd.DataFrame(
        {
            "original_index": all_indices,
            "probability": probabilities,
            "label": all_labels,  # Already fetched and aligned with indices 0 to N-1
        }
    )

    # Optional: Sort by probability for quick viewing of lowest/highest prob samples
    results_df = results_df.sort_values("probability", ascending=True)

    # 6. Fetch Metadata (Simplified/Removed unless needed for other purpose)
    # The prompt only asked for index, probability, and label.
    # If you still want the mzml_filepath/scan_number, you can adapt the
    # fetching logic from the original script here, but the requested info is done.

    if args.fetch_metadata:
        script_logger.info("Fetching optional metadata (filepath/scan_number)...")
        # Since we sorted, the order is no longer 0 to N-1.
        # We must re-sort by index to efficiently fetch metadata.
        fetch_df = results_df.sort_values("original_index")

        sorted_indices = fetch_df["original_index"].values
        metadata_columns = ["mzml_filepath", "ms2_scan_number"]

        # Batch the fetching
        batch_size = 100000
        metadata_chunks = []
        for i in tqdm(range(0, len(sorted_indices), batch_size), desc="Fetching Metadata"):
            batch_idx = sorted_indices[i : i + batch_size]
            arrow_tbl = ds.take(batch_idx, columns=metadata_columns)
            chunk_df = arrow_tbl.to_pandas()
            metadata_chunks.append(chunk_df)

        metadata_df = pd.concat(metadata_chunks, ignore_index=True)

        # Add metadata back to the main DataFrame, which is sorted by original_index
        fetch_df["mzml_filepath"] = metadata_df["mzml_filepath"].values
        fetch_df["scan_number"] = metadata_df["ms2_scan_number"].values

        # Re-sort for final output by probability
        results_df = fetch_df.sort_values("probability", ascending=True)

    # 7. Save to CSV
    script_logger.info(f"Saving {len(results_df)} results to {args.output_csv}...")
    results_df.to_csv(args.output_csv, index=False)

    script_logger.info(f"Done. Results saved to {args.output_csv}")


# --- Main function update for arguments and calling the new function ---


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Predict on test dataset and save results."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained .ckpt model checkpoint",
    )
    # Changed argument name from train_lance_path to the more general lance_path
    parser.add_argument(
        "--lance_path",
        type=str,
        required=True,
        help="Path to the Lance dataset directory (e.g., /path/to/data)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output .csv file of selected samples",
    )

    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument(
        "--fetch_metadata",
        action="store_true",
        help="Include this flag to also fetch and save mzml_filepath and scan_number.",
    )

    args = parser.parse_args()

    # Call the new function
    select_all_samples(args)


if __name__ == "__main__":
    main()
