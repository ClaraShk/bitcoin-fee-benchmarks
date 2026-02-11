#!/usr/bin/env python3
"""
Evaluate all baseline predictors on benchmark data.

This script demonstrates the evaluation workflow by running all
baseline predictors on both 3h and 1d horizons and saving results.

Usage:
    python examples/run_baseline_evaluation.py
    python examples/run_baseline_evaluation.py --dataset high_fees_20171220_20171226
    python examples/run_baseline_evaluation.py --max-snapshots 50  # Quick test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data_utils import list_datasets
from benchmarks.evaluation import DEFAULT_INCLUSION_PERCENTILE, evaluate_predictor

# Import baseline predictors
from examples.naive_predictor import NaivePredictor, NaiveMaxPredictor
from examples.moving_average_predictor import MovingAveragePredictor, ExponentialMovingAveragePredictor
from examples.median_recent_predictor import MedianRecentPredictor, PercentilePredictor


# All baseline predictors to evaluate
BASELINE_PREDICTORS = [
    NaivePredictor,
    NaiveMaxPredictor,
    MovingAveragePredictor,
    ExponentialMovingAveragePredictor,
    MedianRecentPredictor,
    PercentilePredictor,
]

HORIZONS = ['3h', '1d']


def run_evaluation(
    dataset_path: str,
    output_dir: str = 'results',
    max_snapshots: int = None,
    verbose: bool = False,
) -> list:
    """
    Evaluate all baseline predictors.

    Args:
        dataset_path: Path to dataset
        output_dir: Directory for output files
        max_snapshots: Maximum snapshots per evaluation (for testing)
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = []
    dataset_name = Path(dataset_path).name

    print(f"Dataset: {dataset_name}")
    print(f"Inclusion criterion: {DEFAULT_INCLUSION_PERCENTILE} percentile")
    print("=" * 60)
    print()

    for horizon in HORIZONS:
        print(f"--- Horizon: {horizon} ---")
        print()

        for predictor_class in BASELINE_PREDICTORS:
            predictor = predictor_class(horizon=horizon)
            predictor_name = predictor.get_name()

            print(f"Evaluating {predictor_name}...", end=' ', flush=True)

            try:
                results = evaluate_predictor(
                    predictor=predictor,
                    dataset_path=dataset_path,
                    split='test',
                    inclusion_percentile=DEFAULT_INCLUSION_PERCENTILE,
                    max_snapshots=max_snapshots,
                    verbose=False,
                )

                mae = results['metrics']['mae']
                inc_acc = results['metrics'].get('inclusion_accuracy', float('nan'))

                print(f"MAE: {mae:.2f}, Inclusion: {inc_acc:.2%}")

                # Save individual result
                results_to_save = {k: v for k, v in results.items() if k != 'snapshot_results'}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{predictor_name}_{horizon}_{timestamp}.json"

                with open(output_dir / filename, 'w') as f:
                    json.dump(results_to_save, f, indent=2, default=str)

                all_results.append(results_to_save)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        print()

    return all_results


def print_summary(results: list):
    """Print a summary table of all results."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    # Group by horizon
    for horizon in HORIZONS:
        horizon_results = [r for r in results if r['horizon'] == horizon]
        if not horizon_results:
            continue

        print(f"Horizon: {horizon}")
        print("-" * 50)
        print(f"{'Predictor':<30} {'MAE':>8} {'Inclusion':>10}")
        print("-" * 50)

        for r in sorted(horizon_results, key=lambda x: x['metrics']['mae']):
            name = r['predictor_name']
            mae = r['metrics']['mae']
            inc = r['metrics'].get('inclusion_accuracy', float('nan'))
            print(f"{name:<30} {mae:>8.2f} {inc:>10.2%}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline predictors"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (default: first available)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Maximum snapshots per evaluation (for quick testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Find dataset
    datasets = list_datasets()
    if not datasets:
        print("Error: No datasets found in data/local_datasets/")
        sys.exit(1)

    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found")
            print(f"Available: {', '.join(datasets)}")
            sys.exit(1)
        dataset_name = args.dataset
    else:
        dataset_name = datasets[0]
        print(f"Using first available dataset: {dataset_name}")
        print()

    dataset_path = f"data/local_datasets/{dataset_name}"

    # Run evaluations
    results = run_evaluation(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        max_snapshots=args.max_snapshots,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(results)

    # Save combined results
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"baseline_evaluation_{timestamp}.json"

    with open(combined_file, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'timestamp': timestamp,
            'inclusion_criterion': f'{DEFAULT_INCLUSION_PERCENTILE}_percentile',
            'results': results,
        }, f, indent=2, default=str)

    print(f"Combined results saved to: {combined_file}")


if __name__ == "__main__":
    main()
