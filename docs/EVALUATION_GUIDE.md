# Evaluation Guide

This guide explains how to evaluate fee predictors using the benchmark framework.

## Inclusion Criterion: 0.05 Percentile

The benchmark uses a **0.05 percentile (5th percentile) inclusion criterion** to determine if a predicted fee rate would result in block inclusion.

### Why 0.05 Percentile?

Not all transactions in a block reflect the true fee market:

1. **Miner Transactions**: Miners include their own payout transactions at minimal fees
2. **CPFP (Child-Pays-For-Parent)**: Low-fee parent transactions pulled in by high-fee children
3. **Consolidation Transactions**: Exchanges/services consolidating UTXOs at low fees during off-peak times
4. **Priority Transactions**: Some transactions have non-economic priority

The 5th percentile filters out these edge cases, representing the practical minimum fee rate for organic block inclusion.

### Example

Consider a block with 2000 transactions:
- Fee rates range from 1 to 500 sat/vbyte
- 5th percentile = 15 sat/vbyte

A prediction of 20 sat/vbyte → **Included** (20 ≥ 15)
A prediction of 10 sat/vbyte → **Not Included** (10 < 15)

## Metrics

### Error Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `mae` | Mean Absolute Error | sat/vbyte |
| `rmse` | Root Mean Squared Error | sat/vbyte |

### Accuracy Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `inclusion_accuracy` | Fraction of predictions meeting inclusion criterion | 0-1 |
| `directional_accuracy` | Fraction of correct direction predictions | 0-1 |

### Other Metrics

| Metric | Description |
|--------|-------------|
| `quantile_loss` | Asymmetric loss penalizing under/over-prediction |
| `overpayment_ratio` | Ratio of predicted to minimum required fee |

## Running Evaluations

### CLI Tool

```bash
# Evaluate a single predictor
python evaluate.py examples/naive_predictor.py --horizon 3h

# With options
python evaluate.py examples/moving_average_predictor.py \
    --horizon 1d \
    --dataset high_fees_20171220_20171226 \
    --split test \
    --verbose
```

### Python API

```python
from benchmarks import evaluate_predictor
from examples.naive_predictor import NaivePredictor

# Create predictor
predictor = NaivePredictor(horizon='3h')

# Run evaluation
results = evaluate_predictor(
    predictor=predictor,
    dataset_path='data/local_datasets/high_fees_20171215_20171222',
    split='test',
    inclusion_percentile=0.05,  # Default
)

# Access results
print(f"MAE: {results['metrics']['mae']:.2f}")
print(f"Inclusion: {results['metrics']['inclusion_accuracy']:.2%}")
```

### Batch Evaluation

```bash
# Evaluate all baselines
python examples/run_baseline_evaluation.py

# Quick test with fewer snapshots
python examples/run_baseline_evaluation.py --max-snapshots 50
```

## Output Format

Results are saved as JSON with this schema:

```json
{
  "predictor_name": "NaivePredictor",
  "predictor_params": {"horizon": "3h"},
  "horizon": "3h",
  "dataset": "high_fees_20171215_20171222",
  "split": "test",
  "evaluation_criterion": "0.05_percentile",
  "inclusion_percentile": 0.05,
  "metrics": {
    "mae": 45.23,
    "rmse": 67.89,
    "inclusion_accuracy": 0.72,
    "directional_accuracy": 0.54
  },
  "n_snapshots": 231,
  "timestamp": "2024-01-15T10:30:00"
}
```

## Cross-Validation

For robust evaluation, use time-series cross-validation:

```python
from benchmarks import time_series_cv
from examples.moving_average_predictor import MovingAveragePredictor

predictor = MovingAveragePredictor(horizon='3h')

cv_results = time_series_cv(
    predictor=predictor,
    dataset_path='data/local_datasets/high_fees_20171215_20171222',
    n_splits=5,
)

print(f"Mean MAE: {cv_results['mean_metrics']['mae']:.2f} ± {cv_results['std_metrics']['mae']:.2f}")
```

## Best Practices

1. **Use test split for final evaluation**: Only evaluate on test after development is complete
2. **Report all metrics**: Include MAE, RMSE, and inclusion accuracy
3. **Specify the criterion**: Always note the 0.05 percentile inclusion criterion
4. **Multiple datasets**: Evaluate on different market regimes (high fees, stable, etc.)
5. **Both horizons**: Report results for both 3h and 1d targets
