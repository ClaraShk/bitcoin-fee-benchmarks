"""
Data loading utilities for Bitcoin fee prediction benchmark.

This module provides functions to load and work with benchmark datasets.
Datasets contain mempool snapshots at 10-minute intervals with transaction
details including fee rates, sizes, and confirmation times.

Data Schema:
    Each snapshot contains transactions with the following key columns:

    Identifiers:
        - txid: Transaction hash (unique identifier)

    Features (available for prediction):
        - fee_rate: Fee rate in satoshis per virtual byte
        - virtual_size: Virtual size in vbytes (weight/4)
        - size: Raw transaction size in bytes
        - fee: Total fee in satoshis
        - num_of_inputs: Number of transaction inputs
        - output_value: Total output value in satoshis

    Timestamps:
        - first_seen_timestamp: When transaction first entered mempool
        - block_timestamp: When transaction was confirmed (NaT if unconfirmed)
        - snapshot_start: Start of this snapshot window
        - snapshot_end: End of this snapshot window
        - mempool_exit: When transaction left the mempool

    Target derivation:
        - block_height: Block number where transaction was confirmed
        - seconds_in_snapshot: Duration transaction was in this snapshot

Usage:
    >>> from benchmarks.data_utils import load_dataset, get_features
    >>> dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
    >>> features = get_features(dataset['snapshots'][0])
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from benchmarks.constants import (
    FEATURE_COLUMNS,
    HORIZON_SECONDS,
    MINIMAL_COLUMNS,
    TX_ID_COLUMN,
)


def load_dataset(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> Dict:
    """
    Load a benchmark dataset from disk.

    Args:
        path: Path to dataset directory (e.g., 'data/local_datasets/high_fees_20171215_20171222')
        columns: Optional list of columns to load. If None, loads all columns.

    Returns:
        Dictionary containing:
            - 'metadata': Dataset metadata (date range, quality score, etc.)
            - 'index': DataFrame with snapshot file paths and transaction counts
            - 'block_index': DataFrame of confirmed transactions (if available)
            - 'path': Path to dataset directory

    Example:
        >>> dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
        >>> print(dataset['metadata']['dataset_name'])
        'high_fees_20171215_20171222'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    result = {'path': path}

    # Load metadata
    metadata_path = path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            result['metadata'] = json.load(f)
    else:
        result['metadata'] = {}

    # Load index
    index_path = path / 'index.parquet'
    if index_path.exists():
        result['index'] = pd.read_parquet(index_path)
    else:
        raise FileNotFoundError(f"Index not found: {index_path}")

    # Load block index if available
    for block_index_name in ['complete_block_index.parquet', 'block_index.parquet']:
        block_index_path = path / block_index_name
        if block_index_path.exists():
            result['block_index'] = pd.read_parquet(block_index_path, columns=columns)
            break

    return result


def load_snapshot(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a single snapshot parquet file.

    Args:
        path: Path to snapshot parquet file
        columns: Optional list of columns to load

    Returns:
        DataFrame containing snapshot data
    """
    return pd.read_parquet(path, columns=columns)


def iter_snapshots(
    dataset: Dict,
    columns: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """
    Iterate over snapshots in a dataset.

    Args:
        dataset: Dataset dictionary from load_dataset()
        columns: Optional list of columns to load
        start_time: Optional start time filter (ISO format)
        end_time: Optional end time filter (ISO format)

    Yields:
        Tuple of (timestamp, DataFrame) for each snapshot

    Example:
        >>> dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
        >>> for ts, snapshot in iter_snapshots(dataset, columns=MINIMAL_COLUMNS):
        ...     print(f"{ts}: {len(snapshot)} transactions")
    """
    index = dataset['index'].copy()

    # Apply time filters
    if start_time:
        index = index[index['timestamp'] >= start_time]
    if end_time:
        index = index[index['timestamp'] < end_time]

    for _, row in index.iterrows():
        file_path = dataset['path'] / row['file_path']
        snapshot = load_snapshot(file_path, columns=columns)
        yield row['timestamp'], snapshot


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract feature columns from a snapshot DataFrame.

    Args:
        df: Snapshot DataFrame

    Returns:
        DataFrame containing only feature columns that exist in the input
    """
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    return df[available].copy()


def compute_inclusion_target(
    df: pd.DataFrame,
    horizon: str,
    reference_time: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Compute whether transactions were included within a given time horizon.

    This derives the target variable for prediction: did the transaction
    get confirmed within `horizon` time from when it was first seen?

    Args:
        df: Snapshot DataFrame with 'first_seen_timestamp' and 'block_timestamp'
        horizon: One of '3h' (3 hours) or '1d' (1 day)
        reference_time: Optional reference time. If None, uses first_seen_timestamp.

    Returns:
        Boolean Series where True means transaction was included within horizon

    Example:
        >>> snapshot = load_snapshot('path/to/snapshot.parquet')
        >>> included_3h = compute_inclusion_target(snapshot, '3h')
        >>> print(f"Included within 3h: {included_3h.sum()} / {len(included_3h)}")
    """
    if horizon not in HORIZON_SECONDS:
        raise ValueError(f"Unknown horizon: {horizon}. Use one of {list(HORIZON_SECONDS.keys())}")

    horizon_seconds = HORIZON_SECONDS[horizon]

    # Use first_seen_timestamp as reference if not provided
    if reference_time is not None:
        ref_time = reference_time
    else:
        ref_time = df['first_seen_timestamp']

    # Calculate deadline
    deadline = ref_time + pd.Timedelta(seconds=horizon_seconds)

    # Transaction is included if block_timestamp exists and is before deadline
    has_block = df['block_timestamp'].notna()
    within_horizon = df['block_timestamp'] <= deadline

    return has_block & within_horizon


def get_confirmation_time(df: pd.DataFrame) -> pd.Series:
    """
    Get the time from first seen to block confirmation in seconds.

    Args:
        df: Snapshot DataFrame with 'first_seen_timestamp' and 'block_timestamp'

    Returns:
        Series of confirmation times in seconds (NaN for unconfirmed transactions)
    """
    delta = df['block_timestamp'] - df['first_seen_timestamp']
    return delta.dt.total_seconds()


def list_datasets(data_dir: Union[str, Path] = 'data/local_datasets') -> List[str]:
    """
    List available datasets in the data directory.

    Args:
        data_dir: Path to data directory

    Returns:
        List of dataset names
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    datasets = []
    for path in data_dir.iterdir():
        if path.is_dir() and (path / 'index.parquet').exists():
            datasets.append(path.name)

    return sorted(datasets)


def get_train_val_test_splits(
    dataset: Dict,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, List[str]]:
    """
    Get time-based train/validation/test splits for a dataset.

    Splits are done temporally: earlier snapshots for training,
    middle for validation, and latest for testing. This respects
    the temporal nature of the data and avoids look-ahead bias.

    Args:
        dataset: Dataset dictionary from load_dataset()
        train_ratio: Fraction of data for training (default 0.6)
        val_ratio: Fraction of data for validation (default 0.2)

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        a list of snapshot timestamps in that split

    Example:
        >>> dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
        >>> splits = get_train_val_test_splits(dataset)
        >>> print(f"Train: {len(splits['train'])} snapshots")
    """
    index = dataset['index'].copy()
    index = index.sort_values('timestamp')

    n = len(index)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    timestamps = index['timestamp'].tolist()

    return {
        'train': timestamps[:train_end],
        'val': timestamps[train_end:val_end],
        'test': timestamps[val_end:],
    }
