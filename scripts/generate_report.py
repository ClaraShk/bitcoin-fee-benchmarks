#!/usr/bin/env python3
"""
Generate a detailed report for a single predictor result.

Usage:
    python scripts/generate_report.py results/naive_3h.json
    python scripts/generate_report.py results/my_predictor.json --output results/my_report.md

Output:
    - Detailed markdown report with all metrics
    - Plots saved alongside the report
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_report(result: dict, output_path: Path) -> str:
    """Generate detailed markdown report for a single predictor."""
    lines = []

    name = result['predictor_name']
    horizon = result['horizon']

    lines.append(f"# {name} Evaluation Report")
    lines.append("")
    lines.append(f"**Horizon:** {horizon}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Predictor | {name} |")
    lines.append(f"| Horizon | {horizon} |")
    lines.append(f"| Dataset | {result.get('dataset', 'N/A')} |")
    lines.append(f"| Split | {result.get('split', 'N/A')} |")
    lines.append(f"| Snapshots Evaluated | {result.get('n_snapshots', 'N/A')} |")
    lines.append(f"| Evaluation Criterion | {result.get('evaluation_criterion', 'N/A')} |")
    lines.append(f"| Evaluation Time | {result.get('timestamp', 'N/A')} |")
    lines.append("")

    # Parameters
    lines.append("## Predictor Parameters")
    lines.append("")
    params = result.get('predictor_params', {})
    if params:
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in params.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("No parameters recorded.")
    lines.append("")

    # Metrics
    lines.append("## Metrics")
    lines.append("")
    metrics = result.get('metrics', {})

    # Main metrics table
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Value | Description |")
    lines.append("|--------|-------|-------------|")

    metric_descriptions = {
        'mae': 'Mean Absolute Error - average prediction error magnitude',
        'rmse': 'Root Mean Squared Error - penalizes large errors more',
        'inclusion_accuracy': 'Fraction of predictions meeting inclusion criterion',
        'directional_accuracy': 'Fraction of correct direction predictions',
        'mean_prediction': 'Average predicted fee rate',
        'std_prediction': 'Standard deviation of predictions',
        'n_included': 'Number of successful inclusions',
    }

    for k, v in metrics.items():
        desc = metric_descriptions.get(k, '')
        if isinstance(v, float):
            if 'accuracy' in k:
                lines.append(f"| {k} | {v:.2%} | {desc} |")
            else:
                lines.append(f"| {k} | {v:.4f} | {desc} |")
        else:
            lines.append(f"| {k} | {v} | {desc} |")

    lines.append("")

    # Interpretation
    lines.append("### Interpretation")
    lines.append("")

    mae = metrics.get('mae', float('nan'))
    inc = metrics.get('inclusion_accuracy', float('nan'))

    if not np.isnan(mae):
        if mae < 20:
            lines.append(f"- **MAE ({mae:.1f}):** Excellent - predictions are very close to actual fees")
        elif mae < 50:
            lines.append(f"- **MAE ({mae:.1f}):** Good - reasonable prediction accuracy")
        elif mae < 100:
            lines.append(f"- **MAE ({mae:.1f}):** Moderate - room for improvement")
        else:
            lines.append(f"- **MAE ({mae:.1f}):** High - significant prediction errors")

    if not np.isnan(inc):
        if inc > 0.9:
            lines.append(f"- **Inclusion ({inc:.1%}):** Excellent - nearly all predictions would result in inclusion")
        elif inc > 0.7:
            lines.append(f"- **Inclusion ({inc:.1%}):** Good - most predictions succeed")
        elif inc > 0.5:
            lines.append(f"- **Inclusion ({inc:.1%}):** Moderate - about half of predictions succeed")
        else:
            lines.append(f"- **Inclusion ({inc:.1%}):** Low - predictions often too conservative")

    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Inclusion Criterion")
    lines.append("")
    lines.append("A prediction is considered successful if the predicted fee rate exceeds ")
    lines.append("the **0.05 percentile** (5th percentile) of actual fee rates in the target block.")
    lines.append("")
    lines.append("This criterion filters out:")
    lines.append("- Miner payout transactions (low/zero fee)")
    lines.append("- CPFP (Child-Pays-For-Parent) transactions")
    lines.append("- Consolidation transactions during off-peak times")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if not np.isnan(mae) and mae > 50:
        lines.append("- Consider using a more adaptive prediction method")
        lines.append("- Evaluate performance on different market regimes separately")

    if not np.isnan(inc) and inc < 0.7:
        lines.append("- Predictions may be too conservative; consider adjusting parameters")
        lines.append("- Compare with more aggressive baseline predictors")

    if len([l for l in lines if l.startswith("-")]) == 0:
        lines.append("- Continue monitoring performance on new data")
        lines.append("- Consider testing on different datasets/regimes")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed report for a single predictor"
    )
    parser.add_argument(
        "file",
        help="Result JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output markdown file path (default: same name as input with .md)"
    )

    args = parser.parse_args()

    # Load result
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: {args.file} not found")
        sys.exit(1)

    with open(input_path) as f:
        result = json.load(f)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.md')

    # Generate report
    report = generate_report(result, output_path)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
