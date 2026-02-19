"""
Minimum P50 (50th percentile / median) predictor for Bitcoin fee prediction benchmark.

This predictor uses the minimum of the 50th percentile (median) fee rates from
blocks mined in the past 3 hours to predict the fee rate needed for confirmation
in the next 3 hours.

The approach:
1. Look at all blocks mined in the past 3 hours
2. For each block, compute the 50th percentile (median) of fee rates
3. Return the minimum of these per-block medians

This predictor requires the evaluation context (timestamp and block_data).
It raises ValueError if context is missing or no blocks fall in the lookback window.

Usage:
    >>> from examples.min_p50_predictor import MinP50Past3hPredictor
    >>> predictor = MinP50Past3hPredictor(horizon='3h')
    >>> predicted_fee = predictor.predict(features, context)
"""

from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor


class MinP50Past3hPredictor(BasePredictor):
    """
    Predict fee rate as the minimum of 50th percentile (median) from recent blocks.

    For each block mined in the past N hours, computes the 50th percentile
    (median) of fee rates in that block. The prediction is the minimum of these
    per-block medians (a conservative floor for the next horizon).

    This approach:
    - Uses only blocks strictly before the prediction time (no look-ahead)
    - Uses the 50th percentile (median) per block
    - Takes the minimum across blocks for a conservative estimate

    Requires context with 'timestamp' and 'block_data'. No fallback:
    raises ValueError if context is missing or no blocks in the lookback window.
    """

    def __init__(
        self,
        horizon: str = '3h',
        lookback_hours: float = 3.0,
    ) -> None:
        """
        Initialize the Min P50 predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            lookback_hours: Hours of block history to consider (default 3.0)
        """
        super().__init__(horizon=horizon, name='MinP50Past3hPredictor')

        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        self.lookback_hours = lookback_hours

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using minimum of per-block 50th percentiles (medians).

        Args:
            features: DataFrame with mempool transaction features.
            context: Context dict containing:
                - 'timestamp': Current snapshot timestamp (pd.Timestamp)
                - 'block_data': DataFrame of confirmed transactions with
                  block_height, block_timestamp, fee_rate columns

        Returns:
            Predicted fee rate in sat/vbyte

        Raises:
            ValueError: If context is None or missing 'timestamp' or 'block_data',
                or if there are no blocks in the lookback window / no valid medians.
        """
        if context is None or 'block_data' not in context or 'timestamp' not in context:
            raise ValueError(
                "MinP50Past3hPredictor requires context with 'timestamp' and 'block_data'"
            )

        block_data = context['block_data']
        target_time = context['timestamp']

        start_time = target_time - timedelta(hours=self.lookback_hours)
        block_timestamps = block_data['block_timestamp']

        # Handle timezone matching
        if block_timestamps.dt.tz is not None and target_time.tz is None:
            target_time = target_time.tz_localize('UTC')
            start_time = start_time.tz_localize('UTC')
        elif block_timestamps.dt.tz is None and hasattr(target_time, 'tz') and target_time.tz is not None:
            target_time = target_time.tz_localize(None)
            start_time = start_time.tz_localize(None)

        mask = (block_timestamps >= start_time) & (block_timestamps < target_time)
        past_blocks = block_data[mask]

        if len(past_blocks) == 0:
            raise ValueError(
                f"No blocks in lookback window [{start_time}, {target_time}). "
                "MinP50Past3hPredictor requires at least one block."
            )

        # Compute 50th percentile (median) per block
        p50_per_block = []
        for block_height in past_blocks['block_height'].unique():
            block_txs = past_blocks[past_blocks['block_height'] == block_height]
            block_fees = block_txs['fee_rate'].dropna()

            if len(block_fees) == 0:
                continue

            p50 = float(np.percentile(block_fees, 50))
            p50_per_block.append(p50)

        if not p50_per_block:
            raise ValueError(
                "No blocks in lookback window had valid fee_rate data. "
                "MinP50Past3hPredictor requires at least one block with non-empty fee rates."
            )

        return float(np.min(p50_per_block))

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'lookback_hours': self.lookback_hours,
            'percentile': 50,
        }


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    sample_features = pd.DataFrame({
        'fee_rate': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'virtual_size': [200] * 10,
    })

    sample_block_data = pd.DataFrame({
        'block_height': [100] * 50 + [101] * 50 + [102] * 50,
        'block_timestamp': (
            [pd.Timestamp('2021-01-01 10:00:00', tz='UTC')] * 50 +
            [pd.Timestamp('2021-01-01 10:30:00', tz='UTC')] * 50 +
            [pd.Timestamp('2021-01-01 11:00:00', tz='UTC')] * 50
        ),
        'fee_rate': (
            list(range(5, 55)) +   # Block 100: 5-54 sat/vB
            list(range(10, 60)) +  # Block 101: 10-59 sat/vB
            list(range(15, 65))    # Block 102: 15-64 sat/vB
        ),
    })

    sample_context = {
        'timestamp': pd.Timestamp('2021-01-01 12:00:00', tz='UTC'),
        'block_data': sample_block_data,
    }

    print("=== MinP50Past3hPredictor ===")
    predictor = MinP50Past3hPredictor(horizon='3h')
    print(f"Predictor: {predictor}")
    print(f"Parameters: {predictor.get_params()}")

    # With valid context
    prediction = predictor.predict(sample_features, context=sample_context)
    print(f"Prediction (with context): {prediction:.1f} sat/vbyte")
    # Expected: min of median for blocks 100, 101, 102 -> 29.5 (from block 100)

    # Without context raises
    try:
        predictor.predict(sample_features)
    except ValueError as e:
        print(f"Without context (expected): {e!r}")
