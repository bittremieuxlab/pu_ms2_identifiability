import lance
import os
import argparse
import pandas as pd


def analyze_lance_stats_ohe_logic(dataset_path, polarity):
    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} does not exist.")
        return

    ds = lance.dataset(dataset_path)

    # 1. Setup Scanner

    scanner = ds.scanner(columns=["instrument_settings", "label"])
    batch_reader = scanner.to_batches()

    total_count = 0
    positive_label_count = 0

    # OHE Mapping Constants
    POLARITY_NEG_IDX = 14
    POLARITY_POS_IDX = 15

    # 2. Iteration and OHE Masking Logic
    for batch in batch_reader:
        df = batch.to_pandas()

        if polarity is not None:
            if polarity == 1:  # Positive
                mask = df['instrument_settings'].apply(lambda x: x[POLARITY_POS_IDX] > 0.5)
            else:  # Negative
                mask = df['instrument_settings'].apply(lambda x: x[POLARITY_NEG_IDX] > 0.5)

            filtered_df = df[mask]
        else:
            filtered_df = df

        total_count += len(filtered_df)
        positive_label_count += (filtered_df['label'] == 1).sum()

    # 3. Output Results
    pol_label = "Positive" if polarity == 1 else "Negative" if polarity == 0 else "All"
    print("-" * 45)
    print(f"Results for Polarity: {pol_label}")
    print("-" * 45)
    print(f"Total samples matched:        {total_count}")
    print(f"Samples with Label=1:         {positive_label_count}")

    if total_count > 0:
        rate = (positive_label_count / total_count) * 100
        print(f"Positivity Rate:              {rate:.2f}%")
    else:
        print("No samples found matching these criteria.")
    print("-" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the .lance directory")
    parser.add_argument("--polarity", type=int, choices=[0, 1], required=True, help="0 for Neg, 1 for Pos")

    args = parser.parse_args()
    analyze_lance_stats_ohe_logic(args.path, args.polarity)