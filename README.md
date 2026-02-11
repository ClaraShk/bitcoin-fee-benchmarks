# Bitcoin Fee Benchmarks

A benchmark suite for evaluating Bitcoin fee prediction algorithms.

## Overview

This repository provides standardized datasets and evaluation tools for comparing different approaches to Bitcoin transaction fee prediction. Whether you're building a simple heuristic or a sophisticated ML model, this benchmark helps you:

- Evaluate prediction accuracy with standardized metrics
- Compare against baseline predictors
- Ensure fair comparisons with consistent train/test splits
- Avoid common pitfalls like lookahead bias

Data files are stored using Git LFS (Large File Storage).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ClaraShk/bitcoin-fee-benchmarks.git
cd bitcoin-fee-benchmarks

# Install Git LFS and pull data
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run a baseline predictor
python evaluate.py examples/naive_predictor.py --horizon 3h

# Compare multiple results
python scripts/compare_results.py results/*.json

# Interactive exploration
jupyter notebook notebooks/compare_predictors.ipynb
```

## Installation

### Requirements

- Python 3.9+
- Git LFS (for data files)

### Setup

```bash
# Install Git LFS (macOS)
brew install git-lfs

# Or download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Clone and pull data
git clone https://github.com/ClaraShk/bitcoin-fee-benchmarks.git
cd bitcoin-fee-benchmarks
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For advanced visualizations and ML models:
```bash
pip install scikit-learn tensorflow torch  # ML frameworks
pip install plotly bokeh                    # Interactive plots
```

## Data

The benchmark includes mempool snapshot datasets from various high-fee periods:

- **Source:** 10-minute mempool snapshots from BigQuery
- **Features:** `fee_rate`, `virtual_size`, `fee`, `num_of_inputs`, `output_value`
- **Targets:** Block inclusion within 3 hours or 1 day

### Prediction Horizons

| Horizon | Description |
|---------|-------------|
| `3h` | Predict if transaction confirms within 3 hours |
| `1d` | Predict if transaction confirms within 1 day |

### Train/Val/Test Split

Each dataset is split temporally (no data leakage):
- **Train:** First 60% of snapshots
- **Validation:** Next 20%
- **Test:** Final 20%

See [data/README.md](data/README.md) for detailed schema and column descriptions.

## Creating Your Own Predictor

### Basic Structure

```python
from benchmarks.base_predictor import BasePredictor

class MyPredictor(BasePredictor):
    def __init__(self, horizon='3h'):
        super().__init__(horizon=horizon)
        # Your initialization here

    def predict(self, features):
        # features is a DataFrame with columns:
        # fee_rate, virtual_size, size, fee, num_of_inputs, output_value

        # Return predicted fee rate (sat/vbyte)
        return features['fee_rate'].median()  # Simple example
```

### Steps

1. Copy `examples/template_predictor.py` as a starting point
2. Implement your prediction logic in `predict()`
3. Optionally implement `fit()` for training
4. Test: `python evaluate.py my_predictor.py --horizon 3h`

See [docs/PREDICTOR_GUIDE.md](docs/PREDICTOR_GUIDE.md) for detailed guidance.

## Using LLMs to Build Predictors

Large Language Models are excellent tools for rapidly prototyping fee predictors.

### Why Use LLMs?

- **Rapid prototyping:** Try many approaches quickly
- **Explore techniques:** Implement methods you're not familiar with
- **Feature engineering:** Get ideas for derived features
- **Debug and improve:** Share results and iterate

### Example Prompts

**For time series models:**
```
I have Bitcoin mempool data with features: fee_rate, virtual_size,
num_of_inputs, output_value, and timestamps.

Build a predictor using ARIMA that predicts the fee rate needed for
3-hour confirmation. The predictor should inherit from BasePredictor
in benchmarks/base_predictor.py.

Key requirements:
- Use only data available at prediction time (no lookahead)
- Handle missing values appropriately
- Return a single fee rate prediction in sat/vbyte
```

**For ML models:**
```
Create a gradient boosting predictor for Bitcoin fees using these features:
- fee_rate: current mempool fee rates
- virtual_size: transaction sizes
- output_value: transaction values

The predictor should:
1. Inherit from BasePredictor (in benchmarks/base_predictor.py)
2. Implement fit() to train on historical data
3. Implement predict() to return fee rate predictions
4. Handle the time-series nature (no lookahead bias)
```

**For hybrid approaches:**
```
Combine a percentile-based baseline with a neural network that learns
to predict residuals (errors from the baseline).

Features available: fee_rate, virtual_size, fee, num_of_inputs
Target: fee rate needed for 1-day confirmation

Use the BasePredictor interface and ensure no future data is used.
```

### Tips for LLM-Generated Predictors

1. **Start with the template:** Ask the LLM to enhance `template_predictor.py`

2. **Validate thoroughly:** Always test with `evaluate.py`

3. **Watch for data leakage:** LLMs sometimes accidentally use future data
   - Include in your prompt: "Ensure no lookahead bias in features"
   - Check that predictions only use `first_seen_timestamp < prediction_time`

4. **Common pitfalls to mention:**
   - "Use only data available at prediction time"
   - "Handle NaN values appropriately"
   - "The predict() method receives a snapshot DataFrame"

5. **Iterate:** Share results with the LLM and ask for improvements:
   ```
   My predictor achieved MAE=45 and inclusion_accuracy=65%.
   The baseline gets MAE=20. How can I improve?

   Here's my current code: [paste code]
   ```

### Workflow Example

1. **Describe your idea** to an LLM with context about the data
2. **Get generated code** and save as `examples/my_predictor.py`
3. **Test it:**
   ```bash
   python evaluate.py examples/my_predictor.py --horizon 3h
   ```
4. **Review results** and share with LLM for refinement
5. **Compare** using `scripts/compare_results.py`

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error (sat/vbyte) |
| `rmse` | Root Mean Squared Error |
| `inclusion_accuracy` | Fraction meeting inclusion criterion |
| `directional_accuracy` | Fraction of correct direction predictions |

### Inclusion Criterion

A prediction is successful if the fee rate exceeds the **0.05 percentile** (5th percentile) of actual fees in the target block. This filters out anomalous transactions (miner payouts, CPFP) that don't reflect true fee market dynamics.

### Running Evaluations

```bash
# Single predictor
python evaluate.py examples/my_predictor.py --horizon 3h

# With options
python evaluate.py examples/my_predictor.py \
    --horizon 1d \
    --dataset high_fees_20171220_20171226 \
    --split test \
    --verbose
```

### Interpreting Results

- **MAE < 20:** Excellent prediction accuracy
- **MAE 20-50:** Good, competitive with baselines
- **MAE > 100:** Needs improvement

- **Inclusion > 90%:** Nearly all predictions would confirm
- **Inclusion 70-90%:** Good success rate
- **Inclusion < 50%:** Predictions too conservative

See [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) for detailed guidance.

## Sharing Results

### Generate Reports

```bash
# Compare multiple predictors
python scripts/compare_results.py results/*.json

# Single predictor report
python scripts/generate_report.py results/my_predictor_3h.json

# Interactive notebook
jupyter notebook notebooks/compare_predictors.ipynb
```

### Share Your Work

We encourage sharing your predictors and results!

1. **GitHub Discussions:** Post in "Show & Tell" with:
   - Brief description of your approach
   - Results JSON file
   - Key insights

2. **Submit a PR:** Add your predictor to `examples/`

3. **Open an Issue:** Use the results sharing template

Include:
- Your predictor code (or description)
- Results JSON from evaluation
- Any interesting observations

## Project Structure

```
bitcoin-fee-benchmarks/
├── benchmarks/           # Core library
│   ├── base_predictor.py    # BasePredictor class
│   ├── constants.py         # Configuration constants
│   ├── data_utils.py        # Data loading utilities
│   ├── evaluation.py        # Evaluation framework
│   ├── metrics.py           # Metric functions
│   └── visualization.py     # Plotting utilities
├── data/                 # Datasets (Git LFS)
│   └── local_datasets/      # Parquet files
├── docs/                 # Documentation
│   ├── EVALUATION_GUIDE.md
│   └── PREDICTOR_GUIDE.md
├── examples/             # Example predictors
│   ├── template_predictor.py
│   ├── naive_predictor.py
│   ├── moving_average_predictor.py
│   └── median_recent_predictor.py
├── notebooks/            # Jupyter notebooks
│   └── compare_predictors.ipynb
├── results/              # Evaluation results
├── scripts/              # Utility scripts
│   ├── compare_results.py
│   └── generate_report.py
├── evaluate.py           # CLI evaluation tool
└── requirements.txt
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Submitting new predictors
- Sharing results
- Code style
- Pull request workflow

## License

MIT
