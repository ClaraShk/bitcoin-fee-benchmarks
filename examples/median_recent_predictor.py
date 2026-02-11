"""
Median recent fee predictor - returns median of recent fee rates.

The median is more robust to outliers than the mean, making this
a reliable baseline predictor. Particularly useful when the fee
distribution has extreme values (very high or very low fee txs).

Usage:
    >>> from examples.median_recent_predictor import MedianRecentPredictor
    >>> predictor = MedianRecentPredictor(horizon='3h', n_recent=200)
    >>> predicted_fee = predictor.predict(features)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor


class MedianRecentPredictor(BasePredictor):
    """
    Median-based baseline predictor.

    Returns the median fee rate from the last N transactions in
    the mempool snapshot. The median is robust to outliers, making
    this predictor stable even when there are extreme fee rates.

    Why median over mean:
    - Not affected by whale transactions with very high fees
    - Not pulled down by old low-fee transactions stuck in mempool
    - Represents "what most transactions are paying"

    Default N values:
    - 3h horizon: 200 transactions
    - 1d horizon: 1000 transactions
    """

    # Default N values by horizon
    DEFAULT_N = {
        '3h': 200,    # Focus on recent activity for short-term
        '1d': 1000,   # Broader sample for long-term stability
    }

    def __init__(
        self,
        horizon: str = '3h',
        n_recent: Optional[int] = None,
    ) -> None:
        """
        Initialize the median recent predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            n_recent: Number of recent transactions to consider.
                If None, uses default based on horizon.
        """
        super().__init__(horizon=horizon, name='MedianRecentPredictor')

        if n_recent is not None:
            self.n_recent = n_recent
        else:
            self.n_recent = self.DEFAULT_N.get(horizon, 200)

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using median of recent transactions.

        Args:
            features: DataFrame with mempool transaction features.
                Must contain 'fee_rate' column.
            context: Optional context dict (unused by this predictor)

        Returns:
            Median fee rate in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        fee_rates = features['fee_rate']

        # Use last n_recent transactions, or all if fewer available
        n = min(self.n_recent, len(fee_rates))
        recent_fees = fee_rates.tail(n)

        return float(recent_fees.median())

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'n_recent': self.n_recent,
        }


class PercentilePredictor(BasePredictor):
    """
    Percentile-based fee predictor.

    Returns a specific percentile of recent fee rates. Useful for
    tuning the aggressiveness of fee predictions:

    - Higher percentile (e.g., 75-90): More aggressive, faster confirmation
    - Lower percentile (e.g., 25-50): More conservative, saves fees

    Default percentiles:
    - 3h horizon: 75th percentile (need to beat more competition)
    - 1d horizon: 50th percentile (can wait for cheaper slot)
    """

    # Default percentiles by horizon
    DEFAULT_PERCENTILE = {
        '3h': 75,   # More aggressive for short-term
        '1d': 50,   # More conservative for long-term
    }

    def __init__(
        self,
        horizon: str = '3h',
        percentile: Optional[float] = None,
        n_recent: Optional[int] = None,
    ) -> None:
        """
        Initialize the percentile predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            percentile: Which percentile to return (0-100).
                If None, uses default based on horizon.
            n_recent: Number of recent transactions to consider.
                If None, uses all transactions in the snapshot.
        """
        super().__init__(horizon=horizon, name='PercentilePredictor')

        if percentile is not None:
            if not 0 <= percentile <= 100:
                raise ValueError("percentile must be between 0 and 100")
            self.percentile = percentile
        else:
            self.percentile = self.DEFAULT_PERCENTILE.get(horizon, 50)

        self.n_recent = n_recent

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using the specified percentile.

        Args:
            features: DataFrame with 'fee_rate' column
            context: Optional context dict (unused by this predictor)

        Returns:
            Percentile fee rate in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        fee_rates = features['fee_rate']

        # Optionally limit to recent transactions
        if self.n_recent is not None:
            n = min(self.n_recent, len(fee_rates))
            fee_rates = fee_rates.tail(n)

        return float(fee_rates.quantile(self.percentile / 100))

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        params = {
            'horizon': self.horizon,
            'percentile': self.percentile,
        }
        if self.n_recent is not None:
            params['n_recent'] = self.n_recent
        return params


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    # Test with sample data (including outliers)
    sample_features = pd.DataFrame({
        'fee_rate': [5, 10, 15, 20, 25, 30, 35, 40, 45, 500],  # 500 is outlier
        'virtual_size': [200] * 10,
    })

    print("=== MedianRecentPredictor ===")
    print("(Note: outlier fee_rate=500 in data)")
    for horizon in ['3h', '1d']:
        predictor = MedianRecentPredictor(horizon=horizon)
        prediction = predictor.predict(sample_features)
        print(f"{horizon}: {prediction:.1f} sat/vbyte (n={predictor.n_recent})")

    print()
    print("=== Compare median vs mean (with outlier) ===")
    print(f"Mean: {sample_features['fee_rate'].mean():.1f} sat/vbyte")
    print(f"Median: {sample_features['fee_rate'].median():.1f} sat/vbyte")
    print("(Median is robust to the 500 outlier)")

    print()
    print("=== PercentilePredictor ===")
    for horizon in ['3h', '1d']:
        predictor = PercentilePredictor(horizon=horizon)
        prediction = predictor.predict(sample_features)
        print(f"{horizon}: {prediction:.1f} sat/vbyte (p={predictor.percentile})")

    print()
    print("=== Custom percentiles ===")
    for p in [25, 50, 75, 90]:
        predictor = PercentilePredictor(horizon='3h', percentile=p)
        prediction = predictor.predict(sample_features)
        print(f"p{p}: {prediction:.1f} sat/vbyte")
