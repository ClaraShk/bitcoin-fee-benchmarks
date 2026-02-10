"""
Base predictor interface for Bitcoin fee prediction benchmark.

All predictors should inherit from BasePredictor and implement
the required methods. This ensures consistent evaluation across
different prediction approaches.

Example:
    >>> from benchmarks.base_predictor import BasePredictor
    >>> class MyPredictor(BasePredictor):
    ...     def predict(self, features):
    ...         return features['fee_rate'].median()  # Simple baseline
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    Abstract base class for fee rate predictors.

    A predictor estimates the fee rate (in sat/vbyte) needed for a
    transaction to be included within the specified time horizon.

    Attributes:
        horizon: Target time horizon ('3h' or '1d')
        name: Human-readable name for the predictor

    Subclasses must implement:
        - predict(): Return fee rate predictions

    Subclasses may optionally override:
        - fit(): Train the predictor on historical data
        - get_params(): Return hyperparameters
    """

    def __init__(self, horizon: str = '3h', name: Optional[str] = None) -> None:
        """
        Initialize the predictor.

        Args:
            horizon: Target time horizon for predictions.
                '3h' = confirm within 3 hours
                '1d' = confirm within 1 day (24 hours)
            name: Optional custom name for this predictor instance.
                If not provided, uses the class name.

        Raises:
            ValueError: If horizon is not '3h' or '1d'
        """
        valid_horizons = ('3h', '1d')
        if horizon not in valid_horizons:
            raise ValueError(f"horizon must be one of {valid_horizons}, got '{horizon}'")

        self.horizon = horizon
        self._name = name

    def fit(
        self,
        train_data: Union[pd.DataFrame, Dict[str, Any]],
    ) -> 'BasePredictor':
        """
        Fit the predictor to training data (optional).

        Override this method if your predictor needs to learn from
        historical data. Many simple predictors (e.g., percentile-based)
        may not need training.

        Args:
            train_data: Training data. Can be:
                - DataFrame: Snapshot with features and targets
                - Dict: Custom training data structure

        Returns:
            self: The fitted predictor instance

        Example:
            >>> predictor = MyPredictor(horizon='3h')
            >>> predictor.fit(training_snapshots)
            >>> predictions = predictor.predict(test_features)
        """
        # Default implementation: no-op
        return self

    @abstractmethod
    def predict(
        self,
        features: pd.DataFrame,
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Predict fee rate(s) for the given features.

        This is the core method that must be implemented by all predictors.
        Given current mempool features, return the recommended fee rate(s)
        for transactions to be confirmed within the target horizon.

        Args:
            features: DataFrame containing feature columns. Available columns:
                - fee_rate: Current fee rates in mempool (sat/vbyte)
                - virtual_size: Transaction virtual sizes
                - size: Transaction raw sizes
                - fee: Transaction fees in satoshis
                - num_of_inputs: Number of inputs per transaction
                - output_value: Total output values

        Returns:
            Predicted fee rate(s) in sat/vbyte. Can be:
                - float: Single fee rate recommendation
                - np.ndarray: Array of predictions (one per row)
                - pd.Series: Series of predictions (one per row)

        Example:
            >>> features = get_features(snapshot)
            >>> recommended_fee = predictor.predict(features)
            >>> print(f"Recommended fee rate: {recommended_fee:.1f} sat/vbyte")
        """
        pass

    def get_name(self) -> str:
        """
        Get the predictor's name.

        Returns:
            Human-readable name for this predictor
        """
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    def get_params(self) -> Dict[str, Any]:
        """
        Get predictor hyperparameters.

        Override this method to return any configurable parameters
        that affect the predictor's behavior. Useful for logging
        and reproducibility.

        Returns:
            Dictionary of parameter names to values

        Example:
            >>> predictor = PercentilePredictor(percentile=75)
            >>> predictor.get_params()
            {'percentile': 75, 'horizon': '3h'}
        """
        return {'horizon': self.horizon}

    def __repr__(self) -> str:
        """String representation of the predictor."""
        params = self.get_params()
        param_str = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f"{self.get_name()}({param_str})"
