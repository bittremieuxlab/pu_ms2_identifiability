# Inference and Prediction Guide

This guide explains how to use trained models to predict MS2 spectral quality.

## Prerequisites

- **Trained nnPU model checkpoint** - Download from Zenodo (see `checkpoints/README.md`) or use your own trained model
- **Lance dataset** - Test Set 3 for final evaluation (download from Zenodo, see `data/README.md`)
- **GPU** - The current implementation was tested using **2x Nvidia Tesla V100 (32 GB)**. The code was run on a cluster node equipped with Intel Xeon Gold 5218 CPUs and 384 GB of RAM.
## Running Inference

Use the provided SLURM script to generate quality predictions for data in a Lance dataset (Test set 3):

```bash
sbatch slurm_scripts/inference/run_predict_lance.sh
```


**Configuration** (edit the script before running):

```bash
python scripts/inference/predict_lance_all.py \
    --checkpoint_path /path/to/checkpoints/best_model_nnpu.ckpt \
    --lance_path /path/to/data/lance_data_test_set_3 \
    --output_csv results/predictions.csv \
    --batch_size 2048 \
    --fetch_metadata
```

**Key Arguments:**
- `--checkpoint_path`: Path to trained model (`.ckpt` file)
- `--lance_path`: Path to Lance dataset directory
- `--output_csv`: Output file for predictions
- `--batch_size`: Batch size for inference (default: 2048)
- `--fetch_metadata`: Include file paths and scan numbers in output



## Output Format

The script generates a CSV file with predictions:

| Column | Description |
|--------|-------------|
| `original_index` | Index in Lance dataset |
| `probability` | Predicted quality score (0-1, higher = better quality) |
| `label` | Ground truth label if available (1 = positive, 0 = unlabeled) |
| `mzml_filepath` |  Source mzML file path |
| `scan_number` |  MS2 scan number |


## Interpreting Results


### Decision Threshold

To classify spectra as "high quality" or "low quality", a threshold is applied:

**Our threshold**: 0.767

This threshold was determined using Test Set 2 (5th percentile of known positive sample probabilities).


## Expected Performance

On Test Set 3, the nnPU model achieves:

| Metric | Value           |
|--------|-----------------|
| **Recall** | 0.8986 (89.86%) |
| **Threshold** | 0.767           |
| **Test Set** | 8 datasets      |

**Note**:  We primarily evaluate recall on known positive samples, as true negatives are unavailable.

---



## Summary

**Quick Workflow**:

1. Download pre-trained model and test data from Zenodo
2. Run inference: `sbatch slurm_scripts/inference/run_predict_lance.sh`
3. Apply threshold (0.767) to classify spectra



---

