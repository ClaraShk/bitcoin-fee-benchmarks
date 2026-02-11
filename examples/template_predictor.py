"""
Template predictor for Bitcoin fee prediction benchmark.

This is a minimal working example showing how to implement a custom
predictor. Copy this file and modify it to create your own predictor.

Usage:
    >>> from examples.template_predictor import MedianFeePredictor
    >>> predictor = MedianFeePredictor(horizon='3h')
    >>> predicted_fee = predictor.predict(features)
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Import the base class
from benchmarks.base_predictor import BasePredictor


class MedianFeePredictor(BasePredictor):
    """
    Simple baseline predictor using median fee rate.

    This predictor returns the median fee rate from the current
    mempool snapshot. It's a naive baseline that ignores the
    target horizon and transaction-specific features.

    This is NOT a good predictor - it's just a template to show
    the interface. Replace with your own logic.
    """

    def __init__(self, horizon: str = '3h') -> None:
        """
        Initialize the median fee predictor.

        Args:
            horizon: Target time horizon ('3h' or '1d')
        """
        # Always call the parent __init__
        super().__init__(horizon=horizon)

        # Add any predictor-specific state here
        # Example: self.history = []

    def fit(
        self,
        train_data: Union[pd.DataFrame, Dict[str, Any]],
    ) -> 'MedianFeePredictor':
        """
        Fit the predictor (no-op for this simple baseline).

        More sophisticated predictors might:
        - Learn fee rate distributions
        - Train ML models
        - Build lookup tables

        Args:
            train_data: Training data (unused in this baseline)

        Returns:
            self
        """
        # This simple predictor doesn't need training
        # Your predictor might do something like:
        #
        # if isinstance(train_data, pd.DataFrame):
        #     self.model.fit(train_data[FEATURE_COLUMNS], train_data['target'])
        #
        return self

    def predict(
        self,
        features: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict fee rate using median of current mempool.

        Args:
            features: DataFrame with mempool transaction features.
                Must contain 'fee_rate' column.
            context: Optional context dict with timestamp and block_data
                for predictors that need historical information.

        Returns:
            Recommended fee rate in sat/vbyte
        """
        # -----------------------------------------------------
        # REPLACE THIS WITH YOUR PREDICTION LOGIC
        # -----------------------------------------------------
        #
        # You have access to these feature columns:
        #   - fee_rate: Current fee rates in mempool
        #   - virtual_size: Transaction virtual sizes
        #   - size: Transaction raw sizes
        #   - fee: Transaction fees in satoshis
        #   - num_of_inputs: Number of inputs
        #   - output_value: Total output values
        #
        # If context is provided, you also have access to:
        #   - context['timestamp']: Current snapshot timestamp
        #   - context['block_data']: DataFrame of confirmed transactions
        #
        # Your goal: predict the fee rate needed to confirm
        # within self.horizon ('3h' or '1d')
        #
        # Example approaches:
        #   - Return a percentile of current fee rates
        #   - Use historical block data from context
        #   - Train an ML model
        #   - Use mempool size/congestion as a signal
        #
        # -----------------------------------------------------

        if 'fee_rate' not in features.columns:
            raise ValueError("features must contain 'fee_rate' column")

        # Simple baseline: return median fee rate
        return float(features['fee_rate'].median())

    def get_params(self) -> Dict[str, Any]:
        """
        Return predictor parameters.

        Include any hyperparameters that affect predictions.
        This helps with logging and reproducibility.

        Returns:
            Dictionary of parameters
        """
        params = super().get_params()
        # Add predictor-specific params here
        # Example: params['percentile'] = self.percentile
        return params


# -----------------------------------------------------
# QUICK TEST
# -----------------------------------------------------
if __name__ == '__main__':
    # Create sample data
    sample_features = pd.DataFrame({
        'fee_rate': [10, 20, 30, 40, 50],
        'virtual_size': [200, 300, 400, 500, 600],
        'fee': [2000, 6000, 12000, 20000, 30000],
    })

    # Test the predictor
    predictor = MedianFeePredictor(horizon='3h')
    prediction = predictor.predict(sample_features)

    print(f"Predictor: {predictor}")
    print(f"Prediction: {prediction} sat/vbyte")
    print(f"Parameters: {predictor.get_params()}")
