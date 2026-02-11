"""
Metrics for evaluating Bitcoin fee prediction models.

This module provides metrics to assess prediction quality, including:
- Error metrics (MAE, RMSE)
- Directional accuracy
- Quantile loss
- Inclusion accuracy (with 0.05 percentile criterion)

Inclusion Criterion:
    A predicted fee rate is considered sufficient for block inclusion if it
    exceeds the 0.05 percentile (5th percentile) of actual fee rates in the
    target block. This filters out edge cases like consolidation transactions
    and CPFP bundles that don't reflect true fee market dynamics.

    Why 0.05 percentile?
    - Blocks often contain low-fee transactions from miners (e.g., payout txs)
    - Some transactions are accelerated via CPFP (child-pays-for-parent)
    - Consolidation transactions may have artificially low fees
    - The 5th percentile represents the practical "floor" for inclusion
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def mae(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    actuals: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Mean Absolute Error between predictions and actuals.

    MAE measures the average magnitude of errors without considering
    direction. It's in the same units as the input (sat/vbyte).

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates

    Returns:
        Mean absolute error in sat/vbyte

    Example:
        >>> mae([10, 20, 30], [12, 18, 35])
        3.0
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    return float(np.mean(np.abs(predictions - actuals)))


def rmse(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    actuals: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate Root Mean Squared Error between predictions and actuals.

    RMSE penalizes large errors more heavily than MAE. Useful when
    large prediction errors are particularly costly.

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates

    Returns:
        Root mean squared error in sat/vbyte

    Example:
        >>> rmse([10, 20, 30], [12, 18, 35])
        3.109...
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def quantile_loss(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    actuals: Union[np.ndarray, pd.Series, Sequence[float]],
    quantile: float = 0.5,
) -> float:
    """
    Calculate quantile (pinball) loss.

    Quantile loss asymmetrically penalizes over- and under-predictions.
    For fee prediction:
    - quantile > 0.5: Penalizes under-prediction more (conservative)
    - quantile < 0.5: Penalizes over-prediction more (cost-saving)
    - quantile = 0.5: Equivalent to MAE (symmetric)

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates
        quantile: Target quantile (0 to 1). Default 0.5.

    Returns:
        Quantile loss value

    Example:
        >>> quantile_loss([10, 20], [15, 15], quantile=0.9)
        # Penalizes under-prediction (10 < 15) more heavily
    """
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    errors = actuals - predictions

    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    return float(np.mean(loss))


def directional_accuracy(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    actuals: Union[np.ndarray, pd.Series, Sequence[float]],
    previous: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
) -> float:
    """
    Calculate directional accuracy (hit rate).

    Measures how often the prediction correctly anticipates the direction
    of fee rate movement (up or down from previous value).

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates
        previous: Previous fee rates. If None, uses shifted actuals.

    Returns:
        Fraction of correct directional predictions (0 to 1)

    Example:
        >>> # Fees went: 10 -> 15 -> 12
        >>> # Predictions were: 12 (up), 14 (down)
        >>> directional_accuracy([12, 14], [15, 12], [10, 15])
        0.5  # Got the "up" right, "down" wrong
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if previous is None:
        # Use shifted actuals as baseline
        if len(actuals) < 2:
            return float('nan')
        previous = np.concatenate([[actuals[0]], actuals[:-1]])
    else:
        previous = np.asarray(previous)

    pred_direction = np.sign(predictions - previous)
    actual_direction = np.sign(actuals - previous)

    # Count matches (including both predicting no change correctly)
    matches = pred_direction == actual_direction
    return float(np.mean(matches))


def inclusion_accuracy(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    block_fee_rates: Union[np.ndarray, pd.Series, pd.DataFrame],
    percentile: float = 0.05,
) -> float:
    """
    Calculate inclusion accuracy using the percentile criterion.

    A prediction is considered correct if the predicted fee rate would
    have been sufficient for inclusion in the target block. The threshold
    is the specified percentile of actual fees in that block.

    Default criterion: 0.05 percentile (5th percentile)
    - Filters out anomalous low-fee transactions
    - Represents practical minimum for block inclusion
    - Accounts for miner transactions and CPFP effects

    Args:
        predictions: Predicted fee rates (one per block)
        block_fee_rates: Either:
            - 1D array: Single block's fee rates
            - 2D array/DataFrame: Multiple blocks (rows=blocks, cols=tx fees)
            - List of arrays: Fee rates for each block
        percentile: Threshold percentile (0-1). Default 0.05.

    Returns:
        Fraction of predictions that meet the inclusion criterion (0 to 1)

    Example:
        >>> # Block had fees: [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        >>> # 5th percentile = ~7.25
        >>> predictions = [8, 5]  # First passes, second fails
        >>> block_fees = [[5, 10, 20, 30, 40, 50, 60, 70, 80, 90]] * 2
        >>> inclusion_accuracy(predictions, block_fees, percentile=0.05)
        0.5
    """
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    predictions = np.asarray(predictions)

    # Handle different input formats for block_fee_rates
    if isinstance(block_fee_rates, pd.DataFrame):
        # Each row is a block
        thresholds = block_fee_rates.apply(
            lambda row: np.nanpercentile(row.dropna(), percentile * 100),
            axis=1
        ).values
    elif isinstance(block_fee_rates, (list, tuple)):
        # List of arrays, one per block
        thresholds = np.array([
            np.nanpercentile(fees, percentile * 100)
            for fees in block_fee_rates
        ])
    else:
        # Single array - treat as one block
        block_fee_rates = np.asarray(block_fee_rates)
        if block_fee_rates.ndim == 1:
            threshold = np.nanpercentile(block_fee_rates, percentile * 100)
            return float(np.mean(predictions >= threshold))
        else:
            # 2D array - each row is a block
            thresholds = np.nanpercentile(block_fee_rates, percentile * 100, axis=1)

    # Check if predictions meet threshold for each block
    included = predictions >= thresholds
    return float(np.mean(included))


def overpayment_ratio(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    min_required: Union[np.ndarray, pd.Series, Sequence[float]],
) -> float:
    """
    Calculate average overpayment ratio.

    Measures how much extra fee was paid compared to the minimum required.
    A ratio of 1.0 means exact payment, 2.0 means paying double.

    Args:
        predictions: Predicted (paid) fee rates
        min_required: Minimum fee rates that would have been accepted

    Returns:
        Average ratio of predicted to minimum required fee rate

    Example:
        >>> overpayment_ratio([20, 30], [10, 20])
        1.75  # Average of 2.0 and 1.5
    """
    predictions = np.asarray(predictions)
    min_required = np.asarray(min_required)

    # Avoid division by zero
    min_required = np.where(min_required > 0, min_required, 1e-10)

    ratios = predictions / min_required
    return float(np.mean(ratios))


def compute_all_metrics(
    predictions: Union[np.ndarray, pd.Series, Sequence[float]],
    actuals: Union[np.ndarray, pd.Series, Sequence[float]],
    block_fee_rates: Optional[Union[list, np.ndarray]] = None,
    inclusion_percentile: float = 0.05,
) -> dict:
    """
    Compute all standard metrics for fee predictions.

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates (e.g., fees paid or block median)
        block_fee_rates: Optional fee rates for each block (for inclusion metric)
        inclusion_percentile: Percentile threshold for inclusion. Default 0.05.

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {
        'mae': mae(predictions, actuals),
        'rmse': rmse(predictions, actuals),
        'quantile_loss_50': quantile_loss(predictions, actuals, 0.5),
        'quantile_loss_90': quantile_loss(predictions, actuals, 0.9),
        'directional_accuracy': directional_accuracy(predictions, actuals),
    }

    if block_fee_rates is not None:
        metrics['inclusion_accuracy'] = inclusion_accuracy(
            predictions, block_fee_rates, percentile=inclusion_percentile
        )
        metrics['inclusion_percentile'] = inclusion_percentile

    return metrics
