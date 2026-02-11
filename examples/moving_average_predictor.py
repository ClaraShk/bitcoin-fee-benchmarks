"""
Moving average fee predictor - returns average of recent fee rates.

This predictor smooths out short-term fluctuations by averaging
recent fee rates. Uses different default windows for different
prediction horizons.

Usage:
    >>> from examples.moving_average_predictor import MovingAveragePredictor
    >>> predictor = MovingAveragePredictor(horizon='3h', window=100)
    >>> predicted_fee = predictor.predict(features)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from benchmarks.base_predictor import BasePredictor


class MovingAveragePredictor(BasePredictor):
    """
    Moving average baseline predictor.

    Computes the mean fee rate over the last N transactions in the
    mempool snapshot. The window size can be configured, with sensible
    defaults based on the prediction horizon.

    Default windows:
    - 3h horizon: 100 transactions (more responsive to recent changes)
    - 1d horizon: 500 transactions (smoother, less reactive)

    This is a simple smoothing baseline. More sophisticated approaches
    might use exponential moving averages, weighted by transaction size,
    or incorporate time-based decay.
    """

    # Default window sizes by horizon
    DEFAULT_WINDOWS = {
        '3h': 100,   # Smaller window for short-term prediction
        '1d': 500,   # Larger window for longer-term prediction
    }

    def __init__(
        self,
        horizon: str = '3h',
        window: Optional[int] = None,
    ) -> None:
        """
        Initialize the moving average predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            window: Number of recent transactions to average.
                If None, uses default based on horizon.
        """
        super().__init__(horizon=horizon, name='MovingAveragePredictor')

        # Set window size
        if window is not None:
            self.window = window
        else:
            self.window = self.DEFAULT_WINDOWS.get(horizon, 100)

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using moving average of recent transactions.

        Args:
            features: DataFrame with mempool transaction features.
                Must contain 'fee_rate' column.
            context: Optional context dict (unused by this predictor)

        Returns:
            Average fee rate over the window in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        fee_rates = features['fee_rate']

        # Use last `window` transactions, or all if fewer available
        n = min(self.window, len(fee_rates))
        recent_fees = fee_rates.tail(n)

        return float(recent_fees.mean())

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'window': self.window,
        }


class ExponentialMovingAveragePredictor(BasePredictor):
    """
    Exponential moving average (EMA) fee predictor.

    Gives more weight to recent observations while still considering
    historical data. The decay factor (alpha) controls how quickly
    old observations lose influence.

    - alpha close to 1: More weight on recent values (reactive)
    - alpha close to 0: More weight on historical values (smooth)
    """

    # Default alpha values by horizon
    DEFAULT_ALPHA = {
        '3h': 0.3,   # More reactive for short-term
        '1d': 0.1,   # Smoother for long-term
    }

    def __init__(
        self,
        horizon: str = '3h',
        alpha: Optional[float] = None,
    ) -> None:
        """
        Initialize the EMA predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
            alpha: Smoothing factor (0 < alpha <= 1).
                Higher = more weight on recent values.
                If None, uses default based on horizon.
        """
        super().__init__(horizon=horizon, name='EMAPredictor')

        if alpha is not None:
            if not 0 < alpha <= 1:
                raise ValueError("alpha must be between 0 and 1")
            self.alpha = alpha
        else:
            self.alpha = self.DEFAULT_ALPHA.get(horizon, 0.2)

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using exponential moving average.

        Args:
            features: DataFrame with 'fee_rate' column
            context: Optional context dict (unused by this predictor)

        Returns:
            EMA of fee rates in sat/vbyte
        """
        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        fee_rates = features['fee_rate']

        # Compute EMA using pandas
        ema = fee_rates.ewm(alpha=self.alpha, adjust=False).mean()

        # Return the last EMA value
        return float(ema.iloc[-1])

    def get_params(self) -> Dict[str, Any]:
        """Return predictor parameters."""
        return {
            'horizon': self.horizon,
            'alpha': self.alpha,
        }


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    # Test with sample data
    sample_features = pd.DataFrame({
        'fee_rate': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'virtual_size': [200] * 10,
    })

    print("=== MovingAveragePredictor ===")
    for horizon in ['3h', '1d']:
        predictor = MovingAveragePredictor(horizon=horizon)
        prediction = predictor.predict(sample_features)
        print(f"{horizon}: {prediction:.1f} sat/vbyte (window={predictor.window})")

    print()
    print("=== Custom window ===")
    predictor = MovingAveragePredictor(horizon='3h', window=5)
    prediction = predictor.predict(sample_features)
    print(f"Window=5: {prediction:.1f} sat/vbyte (last 5 avg)")

    print()
    print("=== ExponentialMovingAveragePredictor ===")
    for horizon in ['3h', '1d']:
        predictor = ExponentialMovingAveragePredictor(horizon=horizon)
        prediction = predictor.predict(sample_features)
        print(f"{horizon}: {prediction:.1f} sat/vbyte (alpha={predictor.alpha})")
