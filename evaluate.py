#!/usr/bin/env python3
"""
CLI tool for evaluating Bitcoin fee predictors.

Usage:
    python evaluate.py examples/naive_predictor.py --horizon 3h
    python evaluate.py examples/moving_average_predictor.py --horizon 1d --dataset high_fees_20171215_20171222

Output:
    - Prints summary to console
    - Saves detailed results to results/<predictor>_<horizon>_<timestamp>.json
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.base_predictor import BasePredictor
from benchmarks.data_utils import list_datasets
from benchmarks.evaluation import DEFAULT_INCLUSION_PERCENTILE, evaluate_predictor


def load_predictor_from_file(filepath: str, horizon: str) -> BasePredictor:
    """
    Dynamically load a predictor class from a Python file.

    Looks for the first class that inherits from BasePredictor.

    Args:
        filepath: Path to Python file containing predictor class
        horizon: Prediction horizon to pass to predictor

    Returns:
        Instantiated predictor
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Predictor file not found: {filepath}")

    # Load module from file
    spec = importlib.util.spec_from_file_location("predictor_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["predictor_module"] = module
    spec.loader.exec_module(module)

    # Find predictor class
    predictor_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and
            issubclass(obj, BasePredictor) and
            obj is not BasePredictor):
            predictor_class = obj
            break

    if predictor_class is None:
        raise ValueError(f"No BasePredictor subclass found in {filepath}")

    return predictor_class(horizon=horizon)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Bitcoin fee predictor on benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py examples/naive_predictor.py --horizon 3h
    python evaluate.py examples/moving_average_predictor.py --horizon 1d
    python evaluate.py examples/median_recent_predictor.py --horizon 3h --dataset high_fees_20171220_20171226

The evaluation uses a 0.05 percentile (5th percentile) inclusion criterion:
a prediction is considered successful if it exceeds the 5th percentile
of actual fee rates in the target block.
        """
    )

    parser.add_argument(
        "predictor",
        help="Path to Python file containing predictor class"
    )
    parser.add_argument(
        "--horizon",
        choices=["3h", "1d"],
        default="3h",
        help="Prediction horizon (default: 3h)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (default: first available)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Data split to evaluate (default: test)"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_INCLUSION_PERCENTILE,
        help=f"Inclusion percentile threshold (default: {DEFAULT_INCLUSION_PERCENTILE})"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Maximum snapshots to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
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

    dataset_path = f"data/local_datasets/{dataset_name}"

    # Load predictor
    print(f"Loading predictor from: {args.predictor}")
    try:
        predictor = load_predictor_from_file(args.predictor, args.horizon)
    except Exception as e:
        print(f"Error loading predictor: {e}")
        sys.exit(1)

    print(f"Predictor: {predictor}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(f"Horizon: {args.horizon}")
    print(f"Inclusion criterion: {args.percentile} percentile")
    print()

    # Run evaluation
    print("Running evaluation...")
    try:
        results = evaluate_predictor(
            predictor=predictor,
            dataset_path=dataset_path,
            split=args.split,
            inclusion_percentile=args.percentile,
            max_snapshots=args.max_snapshots,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    # Print summary
    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Predictor:         {results['predictor_name']}")
    print(f"Horizon:           {results['horizon']}")
    print(f"Dataset:           {results['dataset']}")
    print(f"Split:             {results['split']}")
    print(f"Snapshots:         {results['n_snapshots']}")
    print(f"Criterion:         {results['evaluation_criterion']}")
    print()
    print("Metrics:")
    for name, value in results['metrics'].items():
        if isinstance(value, float):
            if 'accuracy' in name or 'ratio' in name:
                print(f"  {name:25s}: {value:.2%}")
            else:
                print(f"  {name:25s}: {value:.4f}")
        else:
            print(f"  {name:25s}: {value}")

    # Save results
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Remove detailed snapshot_results for JSON (too large)
        results_to_save = {k: v for k, v in results.items() if k != 'snapshot_results'}
        results_to_save['evaluation_criterion'] = f'{args.percentile}_percentile'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results['predictor_name']}_{args.horizon}_{timestamp}.json"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        print()
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
