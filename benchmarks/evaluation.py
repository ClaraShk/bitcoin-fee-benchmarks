"""
Evaluation framework for Bitcoin fee prediction benchmark.

This module provides functions to evaluate fee predictors against
benchmark datasets using standardized metrics and procedures.

Inclusion Criterion (0.05 Percentile):
    A prediction is considered successful if the predicted fee rate
    exceeds the 0.05 percentile (5th percentile) of actual fees in
    the target block. This criterion:

    - Filters out anomalous low-fee transactions (miner payouts, CPFP)
    - Represents the practical minimum for block inclusion
    - Provides a realistic assessment of prediction quality

    Example: If a block has 2000 transactions with fees ranging from
    1 to 500 sat/vbyte, the 5th percentile might be ~15 sat/vbyte.
    A prediction of 20 sat/vbyte would be counted as "included."

Usage:
    >>> from benchmarks.evaluation import evaluate_predictor
    >>> from examples.naive_predictor import NaivePredictor
    >>> results = evaluate_predictor(
    ...     predictor=NaivePredictor(horizon='3h'),
    ...     dataset_path='data/local_datasets/high_fees_20171215_20171222'
    ... )
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor
from benchmarks.constants import HORIZON_SECONDS
from benchmarks.data_utils import (
    get_features,
    get_train_val_test_splits,
    iter_snapshots,
    load_dataset,
    load_snapshot,
)
from benchmarks.metrics import (
    compute_all_metrics,
    directional_accuracy,
    inclusion_accuracy,
    mae,
    rmse,
)


# Default inclusion criterion: 0.05 percentile (5th percentile)
DEFAULT_INCLUSION_PERCENTILE = 0.05


def get_block_fee_rates(
    dataset: Dict,
    snapshot_time: str,
    horizon: str,
) -> Optional[List[float]]:
    """
    Get fee rates from blocks within the prediction horizon.

    Args:
        dataset: Dataset dictionary from load_dataset()
        snapshot_time: Timestamp of the prediction snapshot
        horizon: Prediction horizon ('3h' or '1d')

    Returns:
        List of fee rates from blocks in the horizon window, or None if unavailable
    """
    if 'block_index' not in dataset:
        return None

    block_index = dataset['block_index']
    snapshot_ts = pd.Timestamp(snapshot_time)
    horizon_seconds = HORIZON_SECONDS[horizon]
    horizon_end = snapshot_ts + pd.Timedelta(seconds=horizon_seconds)

    # Handle timezone-aware vs naive timestamps
    block_timestamps = block_index['block_timestamp']
    if block_timestamps.dt.tz is not None:
        # Block timestamps are timezone-aware, make snapshot timestamps match
        snapshot_ts = snapshot_ts.tz_localize('UTC') if snapshot_ts.tz is None else snapshot_ts
        horizon_end = horizon_end.tz_localize('UTC') if horizon_end.tz is None else horizon_end

    # Filter to blocks within horizon
    mask = (
        (block_timestamps >= snapshot_ts) &
        (block_timestamps < horizon_end)
    )
    horizon_blocks = block_index[mask]

    if len(horizon_blocks) == 0:
        return None

    return horizon_blocks['fee_rate'].dropna().tolist()


def get_block_data_past(
    dataset: Dict,
    snapshot_time: str,
    lookback_hours: float = 3.0,
) -> Optional[pd.DataFrame]:
    """
    Get block data (block_height, block_timestamp, fee_rate) for blocks
    in the past lookback window, for use as predictor context.

    Args:
        dataset: Dataset dictionary from load_dataset()
        snapshot_time: Timestamp of the prediction snapshot
        lookback_hours: Hours of block history (default 3.0)

    Returns:
        DataFrame with columns block_height, block_timestamp, fee_rate for
        blocks in [snapshot_ts - lookback_hours, snapshot_ts), or None if
        block_index is not available.
    """
    if 'block_index' not in dataset:
        return None

    block_index = dataset['block_index']
    snapshot_ts = pd.Timestamp(snapshot_time)
    start_ts = snapshot_ts - pd.Timedelta(hours=lookback_hours)

    block_timestamps = block_index['block_timestamp']
    if block_timestamps.dt.tz is not None:
        snapshot_ts = snapshot_ts.tz_localize('UTC') if snapshot_ts.tz is None else snapshot_ts
        start_ts = start_ts.tz_localize('UTC') if start_ts.tz is None else start_ts
    elif hasattr(snapshot_ts, 'tz') and snapshot_ts.tz is not None:
        start_ts = start_ts.tz_localize(None) if start_ts.tz is not None else start_ts

    mask = (block_timestamps >= start_ts) & (block_timestamps < snapshot_ts)
    past = block_index[mask]

    required = ['block_height', 'block_timestamp', 'fee_rate']
    if not all(c in past.columns for c in required):
        return None

    return past[required].copy()


def evaluate_single_snapshot(
    predictor: BasePredictor,
    features: pd.DataFrame,
    block_fee_rates: Optional[List[float]] = None,
    inclusion_percentile: float = DEFAULT_INCLUSION_PERCENTILE,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a predictor on a single snapshot.

    Args:
        predictor: Predictor instance
        features: Snapshot features
        block_fee_rates: Fee rates from target block(s)
        inclusion_percentile: Percentile threshold for inclusion
        context: Optional context (timestamp, block_data) for predictors that need it

    Returns:
        Dictionary with prediction and metrics
    """
    if context is not None:
        try:
            prediction = predictor.predict(features, context=context)
        except TypeError:
            prediction = predictor.predict(features)
    else:
        prediction = predictor.predict(features)

    result = {
        'prediction': float(prediction) if np.isscalar(prediction) else prediction,
        'mempool_size': len(features),
        'mempool_median_fee': float(features['fee_rate'].median()),
    }

    if block_fee_rates is not None and len(block_fee_rates) > 0:
        threshold = np.percentile(block_fee_rates, inclusion_percentile * 100)
        result['block_threshold'] = float(threshold)
        result['block_median_fee'] = float(np.median(block_fee_rates))
        result['included'] = float(prediction) >= threshold

    return result


def evaluate_predictor(
    predictor: BasePredictor,
    dataset_path: Union[str, Path],
    split: str = 'test',
    inclusion_percentile: float = DEFAULT_INCLUSION_PERCENTILE,
    max_snapshots: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a predictor on a benchmark dataset.

    Runs the predictor on each snapshot in the specified split and
    computes metrics using the 0.05 percentile inclusion criterion.

    Args:
        predictor: Predictor instance to evaluate
        dataset_path: Path to dataset directory
        split: Which split to evaluate ('train', 'val', 'test')
        inclusion_percentile: Percentile threshold for inclusion criterion.
            Default 0.05 (5th percentile).
        max_snapshots: Maximum number of snapshots to evaluate (for debugging)
        verbose: Print progress information

    Returns:
        Dictionary containing:
            - predictor_name: Name of the predictor
            - predictor_params: Predictor hyperparameters
            - horizon: Prediction horizon
            - dataset: Dataset name
            - split: Evaluation split
            - evaluation_criterion: Description of inclusion criterion
            - metrics: Computed metrics dict
            - n_snapshots: Number of snapshots evaluated
            - n_included: Number of successful inclusions
            - timestamp: Evaluation timestamp

    Example:
        >>> predictor = NaivePredictor(horizon='3h')
        >>> results = evaluate_predictor(
        ...     predictor,
        ...     'data/local_datasets/high_fees_20171215_20171222'
        ... )
        >>> print(f"Inclusion rate: {results['metrics']['inclusion_accuracy']:.2%}")
    """
    dataset = load_dataset(dataset_path)
    splits = get_train_val_test_splits(dataset)

    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

    split_timestamps = set(splits[split])

    predictions = []
    actuals = []
    all_block_fees = []
    snapshot_results = []

    count = 0
    for timestamp, snapshot in iter_snapshots(dataset):
        if timestamp not in split_timestamps:
            continue

        if max_snapshots is not None and count >= max_snapshots:
            break

        features = get_features(snapshot)
        if len(features) == 0:
            continue

        # Get block fee rates for this horizon
        block_fee_rates = get_block_fee_rates(
            dataset, timestamp, predictor.horizon
        )

        # Build context for predictors that use block history (e.g. past 3h)
        context = None
        block_data_past = get_block_data_past(dataset, timestamp, lookback_hours=3.0)
        if block_data_past is not None:
            context = {
                'timestamp': pd.Timestamp(timestamp),
                'block_data': block_data_past,
            }

        # Evaluate on this snapshot
        result = evaluate_single_snapshot(
            predictor, features, block_fee_rates, inclusion_percentile, context=context
        )
        result['timestamp'] = timestamp
        snapshot_results.append(result)

        # Collect for aggregate metrics
        predictions.append(result['prediction'])
        actuals.append(result['mempool_median_fee'])

        if block_fee_rates is not None:
            all_block_fees.append(block_fee_rates)

        count += 1
        if verbose and count % 100 == 0:
            print(f"Evaluated {count} snapshots...")

    if len(predictions) == 0:
        raise ValueError(f"No snapshots found in {split} split")

    # Compute aggregate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = {
        'mae': mae(predictions, actuals),
        'rmse': rmse(predictions, actuals),
        'directional_accuracy': directional_accuracy(predictions, actuals),
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
    }

    # Compute inclusion accuracy if block data available
    if len(all_block_fees) > 0:
        inclusion_results = [r.get('included', None) for r in snapshot_results]
        inclusion_results = [r for r in inclusion_results if r is not None]
        if len(inclusion_results) > 0:
            metrics['inclusion_accuracy'] = float(np.mean(inclusion_results))
            metrics['n_included'] = sum(inclusion_results)

    return {
        'predictor_name': predictor.get_name(),
        'predictor_params': predictor.get_params(),
        'horizon': predictor.horizon,
        'dataset': Path(dataset_path).name,
        'split': split,
        'evaluation_criterion': f'{inclusion_percentile}_percentile',
        'inclusion_percentile': inclusion_percentile,
        'metrics': metrics,
        'n_snapshots': len(predictions),
        'timestamp': datetime.now().isoformat(),
        'snapshot_results': snapshot_results,
    }


def time_series_cv(
    predictor: BasePredictor,
    dataset_path: Union[str, Path],
    n_splits: int = 5,
    train_ratio: float = 0.6,
    inclusion_percentile: float = DEFAULT_INCLUSION_PERCENTILE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Perform time-series cross-validation with rolling windows.

    Uses expanding window approach where training data grows while
    test windows slide forward in time. Respects temporal ordering
    to prevent look-ahead bias.

    Args:
        predictor: Predictor instance (will be re-fit for each fold)
        dataset_path: Path to dataset directory
        n_splits: Number of CV folds
        train_ratio: Minimum ratio of data for initial training
        inclusion_percentile: Percentile threshold for inclusion
        verbose: Print progress information

    Returns:
        Dictionary with:
            - fold_results: List of results per fold
            - mean_metrics: Average metrics across folds
            - std_metrics: Standard deviation of metrics

    Example:
        >>> predictor = MovingAveragePredictor(horizon='3h')
        >>> cv_results = time_series_cv(predictor, dataset_path, n_splits=5)
        >>> print(f"Mean MAE: {cv_results['mean_metrics']['mae']:.2f}")
    """
    dataset = load_dataset(dataset_path)
    index = dataset['index'].sort_values('timestamp')
    n_total = len(index)

    min_train = int(n_total * train_ratio)
    test_size = (n_total - min_train) // n_splits

    fold_results = []

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_total:
            break

        train_timestamps = index['timestamp'].iloc[:train_end].tolist()
        test_timestamps = index['timestamp'].iloc[test_start:test_end].tolist()

        if verbose:
            print(f"Fold {fold + 1}/{n_splits}: "
                  f"train={len(train_timestamps)}, test={len(test_timestamps)}")

        # Get training data for fit (if predictor uses it)
        train_features = []
        for ts, snapshot in iter_snapshots(dataset):
            if ts in train_timestamps:
                train_features.append(get_features(snapshot))

        if train_features:
            train_data = pd.concat(train_features, ignore_index=True)
            predictor.fit(train_data)

        # Evaluate on test set
        test_timestamps_set = set(test_timestamps)
        predictions = []
        actuals = []
        inclusions = []

        for ts, snapshot in iter_snapshots(dataset):
            if ts not in test_timestamps_set:
                continue

            features = get_features(snapshot)
            if len(features) == 0:
                continue

            pred = predictor.predict(features)
            predictions.append(pred)
            actuals.append(features['fee_rate'].median())

            block_fees = get_block_fee_rates(dataset, ts, predictor.horizon)
            if block_fees:
                threshold = np.percentile(block_fees, inclusion_percentile * 100)
                inclusions.append(pred >= threshold)

        fold_metrics = {
            'mae': mae(predictions, actuals),
            'rmse': rmse(predictions, actuals),
        }
        if inclusions:
            fold_metrics['inclusion_accuracy'] = float(np.mean(inclusions))

        fold_results.append({
            'fold': fold,
            'n_train': len(train_timestamps),
            'n_test': len(test_timestamps),
            'metrics': fold_metrics,
        })

    # Aggregate across folds
    metric_names = fold_results[0]['metrics'].keys()
    mean_metrics = {}
    std_metrics = {}

    for metric in metric_names:
        values = [f['metrics'][metric] for f in fold_results]
        mean_metrics[metric] = float(np.mean(values))
        std_metrics[metric] = float(np.std(values))

    return {
        'predictor_name': predictor.get_name(),
        'predictor_params': predictor.get_params(),
        'n_splits': n_splits,
        'fold_results': fold_results,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
    }
