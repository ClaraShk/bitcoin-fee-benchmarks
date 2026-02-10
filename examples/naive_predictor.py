"""
Naive fee predictor - returns the last observed fee rate.

This is the simplest possible baseline: assume the next fee rate
needed will be similar to what we just saw. Useful as a lower bound
for comparison with more sophisticated approaches.

Usage:
    >>> from examples.naive_predictor import NaivePredictor
    >>> predictor = NaivePredictor(horizon='3h')
    >>> predicted_fee = predictor.predict(features)
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor


class NaivePredictor(BasePredictor):
    """
    Naive baseline predictor that returns the last observed fee rate.

    This predictor simply returns the most recent fee rate from the
    mempool snapshot. It assumes minimal change between observations.

    This is intentionally simplistic - it ignores:
    - The target horizon (3h vs 1d)
    - Mempool congestion trends
    - Transaction-specific features

    Use this as a baseline to ensure more complex predictors
    provide meaningful improvement.
    """

    def __init__(self, horizon: str = '3h') -> None:
        """
        Initialize the naive predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d').
                Note: This predictor ignores the horizon and always
                returns the same prediction regardless.
        """
        super().__init__(horizon=horizon, name='NaivePredictor')

    def predict(
        self,
        features: pd.DataFrame,
    ) -> float:
        """
        Predict fee rate using the last observed value.

        Returns the maximum fee rate from the most recent transactions,
        approximating "what fee rate was just needed to get in."

        Args:
            features: DataFrame with mempool transaction features.
                Must contain 'fee_rate' column.

        Returns:
            Last observed fee rate in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        # Return the last (most recent) fee rate
        # In practice, this is the fee rate of a recently confirmed tx
        return float(features['fee_rate'].iloc[-1])

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'method': 'last_value',
        }


class NaiveMaxPredictor(BasePredictor):
    """
    Naive predictor that returns the maximum recent fee rate.

    More conservative than NaivePredictor - assumes you need to
    beat the highest fee rate currently in the mempool.
    """

    def __init__(self, horizon: str = '3h') -> None:
        """
        Initialize the naive max predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
        """
        super().__init__(horizon=horizon, name='NaiveMaxPredictor')

    def predict(self, features: pd.DataFrame) -> float:
        """
        Predict fee rate using the maximum observed value.

        Args:
            features: DataFrame with 'fee_rate' column

        Returns:
            Maximum fee rate in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        return float(features['fee_rate'].max())

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'method': 'max_value',
        }


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    # Test with sample data
    sample_features = pd.DataFrame({
        'fee_rate': [10, 20, 30, 40, 50],
        'virtual_size': [200, 300, 400, 500, 600],
    })

    print("=== NaivePredictor ===")
    predictor = NaivePredictor(horizon='3h')
    prediction = predictor.predict(sample_features)
    print(f"Predictor: {predictor}")
    print(f"Prediction: {prediction} sat/vbyte")
    print(f"Parameters: {predictor.get_params()}")

    print()
    print("=== NaiveMaxPredictor ===")
    predictor_max = NaiveMaxPredictor(horizon='1d')
    prediction_max = predictor_max.predict(sample_features)
    print(f"Predictor: {predictor_max}")
    print(f"Prediction: {prediction_max} sat/vbyte")
    print(f"Parameters: {predictor_max.get_params()}")
