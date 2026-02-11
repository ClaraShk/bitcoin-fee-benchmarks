"""
Bitcoin fee prediction benchmark suite.
"""

from benchmarks.constants import (
    FEATURE_COLUMNS,
    HORIZON_SECONDS,
    PREDICTION_HORIZONS,
)
from benchmarks.data_utils import (
    compute_inclusion_target,
    get_features,
    get_train_val_test_splits,
    iter_snapshots,
    list_datasets,
    load_dataset,
    load_snapshot,
)
from benchmarks.metrics import (
    compute_all_metrics,
    directional_accuracy,
    inclusion_accuracy,
    mae,
    overpayment_ratio,
    quantile_loss,
    rmse,
)
from benchmarks.evaluation import (
    DEFAULT_INCLUSION_PERCENTILE,
    evaluate_predictor,
    time_series_cv,
)

__all__ = [
    # Constants
    'FEATURE_COLUMNS',
    'HORIZON_SECONDS',
    'PREDICTION_HORIZONS',
    # Data utilities
    'compute_inclusion_target',
    'get_features',
    'get_train_val_test_splits',
    'iter_snapshots',
    'list_datasets',
    'load_dataset',
    'load_snapshot',
    # Metrics
    'compute_all_metrics',
    'directional_accuracy',
    'inclusion_accuracy',
    'mae',
    'overpayment_ratio',
    'quantile_loss',
    'rmse',
    # Evaluation
    'DEFAULT_INCLUSION_PERCENTILE',
    'evaluate_predictor',
    'time_series_cv',
]
