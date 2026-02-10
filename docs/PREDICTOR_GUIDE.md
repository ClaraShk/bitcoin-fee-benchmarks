# Predictor Implementation Guide

This guide explains how to implement a custom fee rate predictor for the Bitcoin fee prediction benchmark.

## Overview

A **predictor** estimates the fee rate (in sat/vbyte) that a transaction needs to be confirmed within a target time horizon. The benchmark evaluates predictors on how well their recommendations lead to timely confirmations without excessive overpayment.

## Quick Start

1. Copy the template: `examples/template_predictor.py`
2. Rename your class and implement `predict()`
3. Test locally with sample data
4. Run the benchmark evaluation

```python
from benchmarks.base_predictor import BasePredictor

class MyPredictor(BasePredictor):
    def predict(self, features):
        # Your prediction logic here
        return recommended_fee_rate
```

## The BasePredictor Interface

All predictors must inherit from `BasePredictor` and implement the `predict()` method.

### Required Method

#### `predict(features) -> fee_rate`

Given current mempool features, return the recommended fee rate.

```python
def predict(self, features: pd.DataFrame) -> float:
    """
    Args:
        features: DataFrame with current mempool transactions

    Returns:
        Recommended fee rate in sat/vbyte
    """
    # Your logic here
    return 50.0  # Example: recommend 50 sat/vbyte
```

### Optional Methods

#### `fit(train_data) -> self`

Train your predictor on historical data. Many simple predictors don't need this.

```python
def fit(self, train_data):
    # Learn from historical patterns
    self.model.fit(train_data)
    return self
```

#### `get_params() -> dict`

Return hyperparameters for logging and reproducibility.

```python
def get_params(self):
    return {
        'horizon': self.horizon,
        'percentile': self.percentile,
    }
```

## Available Features

The `features` DataFrame contains these columns:

| Column | Type | Description |
|--------|------|-------------|
| `fee_rate` | float | Fee rate in sat/vbyte |
| `virtual_size` | float | Transaction virtual size in vbytes |
| `size` | float | Raw transaction size in bytes |
| `fee` | float | Total fee in satoshis |
| `num_of_inputs` | float | Number of transaction inputs |
| `output_value` | float | Total output value in satoshis |

Example usage:

```python
def predict(self, features):
    # Get fee rate distribution
    fee_rates = features['fee_rate']

    # Calculate percentiles
    p50 = fee_rates.quantile(0.50)
    p75 = fee_rates.quantile(0.75)

    # Use mempool size as congestion signal
    mempool_size = features['virtual_size'].sum()

    # Your logic to combine signals
    return recommended_fee
```

## Prediction Horizons

Predictors are initialized with a target horizon:

- **`'3h'`**: Transaction should confirm within 3 hours
- **`'1d'`**: Transaction should confirm within 1 day (24 hours)

Access via `self.horizon` in your predictor:

```python
def predict(self, features):
    if self.horizon == '3h':
        # More aggressive fee for faster confirmation
        return features['fee_rate'].quantile(0.80)
    else:
        # Can afford lower fee for 1-day target
        return features['fee_rate'].quantile(0.60)
```

## Example Predictors

### Percentile-Based Predictor

```python
class PercentilePredictor(BasePredictor):
    def __init__(self, horizon='3h', percentile=75):
        super().__init__(horizon=horizon)
        self.percentile = percentile

    def predict(self, features):
        return features['fee_rate'].quantile(self.percentile / 100)

    def get_params(self):
        return {
            'horizon': self.horizon,
            'percentile': self.percentile,
        }
```

### Congestion-Aware Predictor

```python
class CongestionPredictor(BasePredictor):
    def __init__(self, horizon='3h', base_percentile=50):
        super().__init__(horizon=horizon)
        self.base_percentile = base_percentile

    def predict(self, features):
        mempool_vbytes = features['virtual_size'].sum()

        # Adjust percentile based on congestion
        # Higher congestion -> higher percentile needed
        congestion_factor = min(mempool_vbytes / 1_000_000, 1.0)
        adjusted_percentile = self.base_percentile + (50 * congestion_factor)

        return features['fee_rate'].quantile(adjusted_percentile / 100)
```

## Input/Output Specification

### Input Format

Your `predict()` method receives a pandas DataFrame:

```
   fee_rate  virtual_size   size      fee  num_of_inputs  output_value
0     150.5         225.0  225.0  33862.0            1.0    1500000.0
1      75.2         450.0  450.0  33840.0            2.0    2500000.0
2     200.0         180.0  180.0  36000.0            1.0     500000.0
...
```

### Output Format

Return one of:
- **`float`**: Single fee rate recommendation (most common)
- **`np.ndarray`**: One prediction per input row
- **`pd.Series`**: One prediction per input row

```python
# Single recommendation (most common)
return 75.5

# Per-transaction recommendations
return features['fee_rate'] * 1.1  # 10% above current rate
```

## Tips for Getting Started

1. **Start simple**: Begin with a percentile-based approach, then add complexity.

2. **Understand the tradeoff**: Higher fee rates increase confirmation probability but waste money. Lower rates save money but risk delayed confirmation.

3. **Consider the horizon**: A 3-hour target needs more aggressive fees than a 1-day target.

4. **Use mempool signals**:
   - Total mempool size (congestion)
   - Fee rate distribution (competition)
   - Transaction count (demand)

5. **Avoid look-ahead bias**: Only use information available at prediction time. See `BENCHMARK_USAGE_GUIDELINES.md`.

6. **Test locally first**:
   ```python
   from benchmarks import load_dataset, iter_snapshots, get_features

   dataset = load_dataset('data/local_datasets/high_fees_20171215_20171222')
   for ts, snapshot in iter_snapshots(dataset):
       features = get_features(snapshot)
       prediction = your_predictor.predict(features)
       print(f"{ts}: {prediction:.1f} sat/vbyte")
       break
   ```

## File Structure

Place your predictor in the `examples/` directory:

```
examples/
├── template_predictor.py    # Copy and modify this
├── my_predictor.py          # Your implementation
└── another_predictor.py     # Another approach
```

## Next Steps

After implementing your predictor:

1. Test it manually with sample data
2. Run the benchmark evaluation (see main README)
3. Compare results with baselines
4. Iterate and improve
