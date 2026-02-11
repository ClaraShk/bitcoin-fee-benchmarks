#!/usr/bin/env python3
"""
Compare multiple predictor results and generate a comparison report.

Usage:
    python scripts/compare_results.py results/naive*.json results/moving_average*.json
    python scripts/compare_results.py results/*.json --output results/comparison_report.md

Output:
    - Markdown report with comparison tables
    - Comparison plots saved to results/plots/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from benchmarks.visualization import (
    create_comparison_summary,
    plot_metrics_comparison,
    plot_performance_by_horizon,
)


def load_results(filepaths: list) -> list:
    """Load result JSON files."""
    results = []
    for fp in filepaths:
        path = Path(fp)
        if not path.exists():
            print(f"Warning: {fp} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
            data['_filepath'] = str(path)
            results.append(data)
    return results


def generate_markdown_report(results: list, output_dir: Path) -> str:
    """Generate markdown comparison report."""
    lines = []

    lines.append("# Predictor Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Predictor | Horizon | MAE | RMSE | Inclusion | Dataset |")
    lines.append("|-----------|---------|-----|------|-----------|---------|")

    for r in sorted(results, key=lambda x: x['metrics'].get('mae', float('inf'))):
        mae = r['metrics'].get('mae', float('nan'))
        rmse = r['metrics'].get('rmse', float('nan'))
        inc = r['metrics'].get('inclusion_accuracy', float('nan'))
        lines.append(
            f"| {r['predictor_name']} | {r['horizon']} | "
            f"{mae:.2f} | {rmse:.2f} | {inc:.1%} | {r.get('dataset', 'N/A')} |"
        )

    lines.append("")

    # Best performers
    lines.append("## Best Performers")
    lines.append("")

    # By MAE
    best_mae = min(results, key=lambda x: x['metrics'].get('mae', float('inf')))
    lines.append(f"**Lowest MAE:** {best_mae['predictor_name']} ({best_mae['horizon']}) "
                 f"- {best_mae['metrics']['mae']:.2f} sat/vbyte")

    # By inclusion
    best_inc = max(results, key=lambda x: x['metrics'].get('inclusion_accuracy', 0))
    lines.append(f"**Highest Inclusion:** {best_inc['predictor_name']} ({best_inc['horizon']}) "
                 f"- {best_inc['metrics'].get('inclusion_accuracy', 0):.1%}")

    lines.append("")

    # Plots
    lines.append("## Visualizations")
    lines.append("")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Comparison summary
    fig = create_comparison_summary(results)
    plot_path = plot_dir / "comparison_summary.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    lines.append(f"### Overall Comparison")
    lines.append(f"![Comparison Summary](plots/comparison_summary.png)")
    lines.append("")

    # Metrics comparison
    fig = plot_metrics_comparison(results)
    plot_path = plot_dir / "metrics_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    lines.append(f"### Metrics Breakdown")
    lines.append(f"![Metrics Comparison](plots/metrics_comparison.png)")
    lines.append("")

    # By horizon
    fig = plot_performance_by_horizon(results)
    plot_path = plot_dir / "by_horizon.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    lines.append(f"### Performance by Horizon")
    lines.append(f"![By Horizon](plots/by_horizon.png)")
    lines.append("")

    # Individual results details
    lines.append("## Individual Results")
    lines.append("")

    for r in results:
        lines.append(f"### {r['predictor_name']} ({r['horizon']})")
        lines.append("")
        lines.append(f"- **Dataset:** {r.get('dataset', 'N/A')}")
        lines.append(f"- **Split:** {r.get('split', 'N/A')}")
        lines.append(f"- **Snapshots:** {r.get('n_snapshots', 'N/A')}")
        lines.append(f"- **Criterion:** {r.get('evaluation_criterion', 'N/A')}")
        lines.append("")
        lines.append("**Parameters:**")
        for k, v in r.get('predictor_params', {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("**Metrics:**")
        for k, v in r.get('metrics', {}).items():
            if isinstance(v, float):
                if 'accuracy' in k:
                    lines.append(f"- {k}: {v:.2%}")
                else:
                    lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Inclusion Criterion")
    lines.append("")
    lines.append("A prediction is considered successful if it exceeds the **0.05 percentile** ")
    lines.append("(5th percentile) of actual fee rates in the target block. This filters out ")
    lines.append("anomalous low-fee transactions (miner payouts, CPFP, consolidation) that ")
    lines.append("don't reflect true fee market dynamics.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple predictor results"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result JSON files to compare"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/comparison_report.md",
        help="Output markdown file path"
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.files)

    if not results:
        print("Error: No valid result files found")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")

    # Generate report
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)

    report = generate_markdown_report(results, output_dir)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print(f"Plots saved to: {output_dir / 'plots'}/")


if __name__ == "__main__":
    main()
