# Benchmark Data

Short explanation of the benchmark datasets and how they are used. **The actual data lives in `data/local_datasets/`**; this folder only documents the benchmark split and usage.

---

## What the data is

- **Source:** Mempool snapshots extracted from BigQuery at 10-minute intervals. Each snapshot is the **mempool at that time**: transactions that were unconfirmed or later confirmed, with `txid`, `first_seen_timestamp`, `block_timestamp`, `fee`, `virtual_size`, `fee_rate`, etc.
- **Layout:** Each dataset is a directory under `data/local_datasets/<name>/` with:
  - `snapshots/YYYYMMDD/snapshot_YYYYMMDDTHHMMSS.parquet` — one Parquet file per snapshot
  - `index.parquet` — lookup index (snapshot paths, tx counts)
  - `metadata.json` — date range, expected snapshot count, validation result
  - `block_index.parquet` or `complete_block_index.parquet` (optional) — needed for “confirm within X hours” inclusion checks
- **Quality:** Datasets are validated internally (transaction counts, completeness, temporal order, fee-rate sanity). Quality score 0–100; ≥75 is good, ≥60 acceptable. One dataset (Dec 20–26, 2017) has also been validated against mempool.space as a good representation of the mempool.

---

## Where the data lives

This folder contains a **copy** of the benchmark datasets (score ≥ 75) at:

```
benchmark/data/local_datasets/<dataset_name>/
```

Paths are relative to the **project root**. The original data remains in `data/local_datasets/`; use either the copy here or the original.

For which datasets to use as **development** vs. **test**, see [dev_test_split.md](dev_test_split.md). For validation config, see [validation_config.example.yaml](validation_config.example.yaml).

---

## How to use this for the benchmark

1. **Development (tuning):** Use the datasets listed as *Development* in [dev_test_split.md](dev_test_split.md). Point your code at `data/local_datasets/<name>/`.
2. **Test (final evaluation):** Use only the *Test* datasets when you are done changing your method. Run the benchmark once and report those results.
3. **Guidelines:** Follow [BENCHMARK_USAGE_GUIDELINES.md](BENCHMARK_USAGE_GUIDELINES.md) (no overfitting to the benchmark, temporal ordering, report fully).

Do not copy or move the actual snapshot files; keep everything in `data/local_datasets/` and reference it by path.

---

## Data Schema

Each snapshot contains transactions with the following columns:

### Feature Columns (inputs for prediction)

| Column | Type | Description |
|--------|------|-------------|
| `fee_rate` | float | Fee rate in satoshis per virtual byte |
| `virtual_size` | float | Virtual size in vbytes (weight/4) |
| `size` | float | Raw transaction size in bytes |
| `fee` | float | Total fee in satoshis |
| `num_of_inputs` | float | Number of transaction inputs |
| `output_value` | float | Total output value in satoshis |

### Target Columns (what we predict)

| Column | Type | Description |
|--------|------|-------------|
| `block_timestamp` | datetime | When transaction was confirmed (NaT if unconfirmed) |
| `block_height` | float | Block number where transaction was confirmed |

### Other Columns

| Column | Type | Description |
|--------|------|-------------|
| `txid` | string | Transaction hash (unique identifier) |
| `first_seen_timestamp` | datetime | When transaction first entered mempool |
| `snapshot_start` | datetime | Start of this snapshot window |
| `snapshot_end` | datetime | End of this snapshot window |
| `mempool_exit` | datetime | When transaction left the mempool |
| `seconds_in_snapshot` | float | Duration transaction was in this snapshot |

---

## Train/Validation/Test Splits

Within each dataset, use **time-based splits** to respect temporal ordering:

- **Train (60%):** First 60% of snapshots chronologically
- **Validation (20%):** Next 20% of snapshots
- **Test (20%):** Final 20% of snapshots

This prevents look-ahead bias by ensuring the model only trains on data from before the validation/test periods.

Use `get_train_val_test_splits()` from `benchmarks.data_utils` to get these splits automatically.

---

## Loading Data with Python

Use the `benchmarks` package to load and work with datasets:

```python
from benchmarks import (
    load_dataset,
    iter_snapshots,
    get_features,
    compute_inclusion_target,
    get_train_val_test_splits,
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS,
)

# List available datasets
from benchmarks import list_datasets
print(list_datasets())  # ['high_fees_20171215_20171222', ...]

# Load a dataset
dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
print(dataset['metadata'])

# Get train/val/test splits
splits = get_train_val_test_splits(dataset)
print(f"Train: {len(splits['train'])} snapshots")
print(f"Val: {len(splits['val'])} snapshots")
print(f"Test: {len(splits['test'])} snapshots")

# Iterate over snapshots
for timestamp, snapshot in iter_snapshots(dataset):
    features = get_features(snapshot)
    target_3h = compute_inclusion_target(snapshot, '3h')
    print(f"{timestamp}: {len(snapshot)} txs, {target_3h.sum()} included in 3h")
    break  # Just show first snapshot
```
