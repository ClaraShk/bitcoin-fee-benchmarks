"""
Average P05 (5th percentile) predictor for Bitcoin fee prediction benchmark.

This predictor computes the average of 5th percentile fee rates from blocks
mined in the past 3 hours. It uses historical block data to predict what
fee rate is needed for confirmation.

The approach:
1. Look at all blocks mined in the past 3 hours
2. For each block, compute the 5th percentile of fee rates
3. Return the average of these per-block p5 values

This predictor requires the evaluation context (timestamp and block_data)
to access historical block information.

Usage:
    >>> from examples.avg_p05_predictor import AvgP05Past3hPredictor
    >>> predictor = AvgP05Past3hPredictor(horizon='3h')
    >>> predicted_fee = predictor.predict(features, context)
"""

from datetime import timedelta
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor


class AvgP05Past3hPredictor(BasePredictor):
    """
    Predict fee rate as the average of 5th percentile from recent blocks.

    For each block mined in the past N hours, computes the 5th percentile
    of fee rates in that block. The prediction is the mean of these
    per-block p5 values.

    This approach:
    - Uses only blocks strictly before the prediction time (no look-ahead)
    - Filters out anomalous low-fee transactions via the 5th percentile
    - Smooths over multiple blocks to reduce noise

    Requires context with 'timestamp' and 'block_data' to function.
    Falls back to mempool median if context is unavailable.
    """

    def __init__(
        self,
        horizon: str = '3h',
        lookback_hours: float = 3.0,
    ) -> None:
        """
        Initialize the Avg P05 predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            lookback_hours: Hours of block history to consider (default 3.0)
        """
        super().__init__(horizon=horizon, name='AvgP05Past3hPredictor')

        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        self.lookback_hours = lookback_hours

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using average of per-block 5th percentiles.

        Args:
            features: DataFrame with mempool transaction features.
            context: Context dict containing:
                - 'timestamp': Current snapshot timestamp (pd.Timestamp)
                - 'block_data': DataFrame of confirmed transactions with
                  block_height, block_timestamp, fee_rate columns

        Returns:
            Predicted fee rate in sat/vbyte
        """
        # Fallback if context not available
        if context is None or 'block_data' not in context or 'timestamp' not in context:
            # Use mempool median as fallback
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0  # Absolute fallback

        block_data = context['block_data']
        target_time = context['timestamp']

        # Compute time window
        start_time = target_time - timedelta(hours=self.lookback_hours)

        # Filter to blocks in the lookback window (strictly before target_time)
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
            # No blocks in window, fallback to mempool median
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0

        # Compute 5th percentile per block
        p5_per_block = []
        for block_height in past_blocks['block_height'].unique():
            block_txs = past_blocks[past_blocks['block_height'] == block_height]
            block_fees = block_txs['fee_rate'].dropna()

            if len(block_fees) == 0:
                continue

            p5 = float(np.percentile(block_fees, 5))
            p5_per_block.append(p5)

        if not p5_per_block:
            # No valid blocks, fallback
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0

        # Return average of per-block p5 values
        return float(np.mean(p5_per_block))

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'lookback_hours': self.lookback_hours,
            'percentile': 5,
        }


# Also provide a configurable version
class AvgPercentilePredictor(BasePredictor):
    """
    Generalized version with configurable percentile and lookback window.

    This allows experimenting with different percentiles (e.g., p10, p25)
    and lookback windows.
    """

    def __init__(
        self,
        horizon: str = '3h',
        lookback_hours: float = 3.0,
        percentile: float = 5.0,
    ) -> None:
        """
        Initialize the configurable percentile predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            lookback_hours: Hours of block history to consider
            percentile: Which percentile to compute per block (0-100)
        """
        super().__init__(horizon=horizon, name='AvgPercentilePredictor')

        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        self.lookback_hours = lookback_hours
        self.percentile = percentile

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using average of per-block percentiles.

        Args:
            features: DataFrame with mempool transaction features.
            context: Context dict with 'timestamp' and 'block_data'

        Returns:
            Predicted fee rate in sat/vbyte
        """
        if context is None or 'block_data' not in context or 'timestamp' not in context:
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0

        block_data = context['block_data']
        target_time = context['timestamp']
        start_time = target_time - timedelta(hours=self.lookback_hours)

        # Handle timezone matching
        block_timestamps = block_data['block_timestamp']
        if block_timestamps.dt.tz is not None and target_time.tz is None:
            target_time = target_time.tz_localize('UTC')
            start_time = start_time.tz_localize('UTC')
        elif block_timestamps.dt.tz is None and hasattr(target_time, 'tz') and target_time.tz is not None:
            target_time = target_time.tz_localize(None)
            start_time = start_time.tz_localize(None)

        mask = (block_timestamps >= start_time) & (block_timestamps < target_time)
        past_blocks = block_data[mask]

        if len(past_blocks) == 0:
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0

        # Compute percentile per block
        percentile_per_block = []
        for block_height in past_blocks['block_height'].unique():
            block_txs = past_blocks[past_blocks['block_height'] == block_height]
            block_fees = block_txs['fee_rate'].dropna()

            if len(block_fees) == 0:
                continue

            pct = float(np.percentile(block_fees, self.percentile))
            percentile_per_block.append(pct)

        if not percentile_per_block:
            if 'fee_rate' in features.columns:
                return float(features['fee_rate'].median())
            return 1.0

        return float(np.mean(percentile_per_block))

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'lookback_hours': self.lookback_hours,
            'percentile': self.percentile,
        }


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    import pandas as pd

    # Create sample mempool features
    sample_features = pd.DataFrame({
        'fee_rate': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'virtual_size': [200] * 10,
    })

    # Create sample block data
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

    # Sample context
    sample_context = {
        'timestamp': pd.Timestamp('2021-01-01 12:00:00', tz='UTC'),
        'block_data': sample_block_data,
    }

    print("=== AvgP05Past3hPredictor ===")
    predictor = AvgP05Past3hPredictor(horizon='3h')
    print(f"Predictor: {predictor}")
    print(f"Parameters: {predictor.get_params()}")

    # Test without context (fallback)
    prediction_no_ctx = predictor.predict(sample_features)
    print(f"Prediction (no context): {prediction_no_ctx:.1f} sat/vbyte")

    # Test with context
    prediction_with_ctx = predictor.predict(sample_features, context=sample_context)
    print(f"Prediction (with context): {prediction_with_ctx:.1f} sat/vbyte")

    # Expected: avg of p5 for blocks 100, 101, 102
    # Block 100 p5: ~7.45
    # Block 101 p5: ~12.45
    # Block 102 p5: ~17.45
    # Avg: ~12.45
    print()
    print("=== AvgPercentilePredictor (p25) ===")
    predictor_p25 = AvgPercentilePredictor(horizon='3h', percentile=25)
    prediction_p25 = predictor_p25.predict(sample_features, context=sample_context)
    print(f"Prediction (p25): {prediction_p25:.1f} sat/vbyte")
