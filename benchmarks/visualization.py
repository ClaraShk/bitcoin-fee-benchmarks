"""
Visualization tools for Bitcoin fee prediction benchmark.

This module provides plotting functions for analyzing prediction
performance and comparing different predictors.

All functions return matplotlib Figure objects for flexibility
in saving or displaying.

Usage:
    >>> from benchmarks.visualization import plot_predictions_vs_actual
    >>> fig = plot_predictions_vs_actual(predictions, actuals, "My Predictor")
    >>> fig.savefig("predictions.png")
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions_vs_actual(
    predictions: Sequence[float],
    actuals: Sequence[float],
    title: str = "Predictions vs Actual",
    timestamps: Optional[Sequence] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot predicted vs actual fee rates over time.

    Creates a time series plot showing predictions and actuals side by side,
    useful for visually assessing prediction quality and identifying patterns.

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates (e.g., block median or required fee)
        title: Plot title
        timestamps: Optional x-axis timestamps. If None, uses sequential indices.
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_predictions_vs_actual(preds, actuals, "Naive Predictor")
        >>> fig.savefig("naive_predictions.png", dpi=150)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = timestamps if timestamps is not None else range(len(predictions))

    ax.plot(x, actuals, label='Actual', alpha=0.7, linewidth=1)
    ax.plot(x, predictions, label='Predicted', alpha=0.7, linewidth=1)

    ax.set_xlabel('Time' if timestamps is not None else 'Snapshot')
    ax.set_ylabel('Fee Rate (sat/vbyte)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_distribution(
    predictions: Sequence[float],
    actuals: Sequence[float],
    title: str = "Prediction Error Distribution",
    bins: int = 50,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot histogram of prediction errors.

    Shows the distribution of (prediction - actual) errors, helping
    identify bias (systematic over/under-prediction) and variance.

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    errors = np.array(predictions) - np.array(actuals)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax1.axvline(np.mean(errors), color='orange', linestyle='--',
                label=f'Mean: {np.mean(errors):.2f}')
    ax1.set_xlabel('Error (Predicted - Actual)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Histogram')
    ax1.legend()

    # Q-Q style: sorted predictions vs sorted actuals
    ax2 = axes[1]
    sorted_pred = np.sort(predictions)
    sorted_actual = np.sort(actuals)
    ax2.scatter(sorted_actual, sorted_pred, alpha=0.5, s=10)
    max_val = max(max(predictions), max(actuals))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual Fee Rate')
    ax2.set_ylabel('Predicted Fee Rate')
    ax2.set_title('Predicted vs Actual (sorted)')
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    results_list: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    title: str = "Predictor Comparison",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Create bar chart comparing multiple predictors across metrics.

    Args:
        results_list: List of result dictionaries from evaluate_predictor()
        metrics: Which metrics to compare. If None, uses ['mae', 'rmse', 'inclusion_accuracy']
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> results = [eval_result_1, eval_result_2, eval_result_3]
        >>> fig = plot_metrics_comparison(results)
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'inclusion_accuracy']

    # Extract data
    names = []
    metric_values = {m: [] for m in metrics}

    for result in results_list:
        name = f"{result['predictor_name']} ({result['horizon']})"
        names.append(name)
        for m in metrics:
            value = result['metrics'].get(m, float('nan'))
            metric_values[m].append(value)

    # Create subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for ax, metric in zip(axes, metrics):
        values = metric_values[metric]
        bars = ax.bar(x, values, color=colors)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                if 'accuracy' in metric:
                    label = f'{val:.1%}'
                else:
                    label = f'{val:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       label, ha='center', va='bottom', fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_performance_by_horizon(
    results_list: List[Dict[str, Any]],
    metric: str = 'mae',
    title: str = "Performance by Horizon",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Compare predictor performance across different horizons.

    Groups results by predictor and shows side-by-side bars for each horizon.

    Args:
        results_list: List of result dictionaries
        metric: Which metric to plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Group by predictor name
    predictors = {}
    for result in results_list:
        name = result['predictor_name']
        horizon = result['horizon']
        value = result['metrics'].get(metric, float('nan'))

        if name not in predictors:
            predictors[name] = {}
        predictors[name][horizon] = value

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    horizons = sorted(set(r['horizon'] for r in results_list))
    x = np.arange(len(predictors))
    width = 0.8 / len(horizons)

    for i, horizon in enumerate(horizons):
        values = [predictors[name].get(horizon, float('nan'))
                  for name in predictors]
        offset = (i - len(horizons)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=horizon)

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                if 'accuracy' in metric:
                    label = f'{val:.0%}'
                else:
                    label = f'{val:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       label, ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(list(predictors.keys()), rotation=45, ha='right')
    ax.legend(title='Horizon')

    plt.tight_layout()
    return fig


def plot_performance_by_difficulty(
    results: Dict[str, Any],
    difficulty_labels: List[str],
    metric_values: List[float],
    metric: str = 'mae',
    title: str = "Performance by Difficulty",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot predictor performance broken down by difficulty level.

    Useful for analyzing how prediction accuracy varies with market
    conditions (e.g., low/medium/high congestion).

    Args:
        results: Result dictionary for context (predictor name, etc.)
        difficulty_labels: Labels for difficulty levels (e.g., ['Low', 'Medium', 'High'])
        metric_values: Metric values for each difficulty level
        metric: Name of the metric being plotted
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(difficulty_labels))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(difficulty_labels)))

    bars = ax.bar(x, metric_values, color=colors)

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xlabel('Difficulty Level')
    ax.set_title(f"{title}\n{results.get('predictor_name', 'Predictor')}")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_labels)

    # Add value labels
    for bar, val in zip(bars, metric_values):
        if 'accuracy' in metric:
            label = f'{val:.1%}'
        else:
            label = f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               label, ha='center', va='bottom')

    plt.tight_layout()
    return fig


def plot_cumulative_error(
    predictions: Sequence[float],
    actuals: Sequence[float],
    title: str = "Cumulative Absolute Error",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot cumulative absolute error over time.

    Useful for identifying when errors accumulate or if there are
    periods of particularly poor/good prediction.

    Args:
        predictions: Predicted fee rates
        actuals: Actual fee rates
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    errors = np.abs(np.array(predictions) - np.array(actuals))
    cumulative = np.cumsum(errors)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(cumulative, linewidth=1)
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3)

    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Cumulative Absolute Error')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add mean error line
    mean_error = np.mean(errors)
    ax.axline((0, 0), slope=mean_error, color='red', linestyle='--',
              label=f'Mean Error Rate: {mean_error:.2f}')
    ax.legend()

    plt.tight_layout()
    return fig


def create_comparison_summary(
    results_list: List[Dict[str, Any]],
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Create a comprehensive comparison figure with multiple panels.

    Combines metrics comparison, horizon breakdown, and ranking into
    a single figure for easy overview.

    Args:
        results_list: List of result dictionaries
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Panel 1: MAE comparison
    ax1 = fig.add_subplot(2, 2, 1)
    names = [f"{r['predictor_name']}\n({r['horizon']})" for r in results_list]
    mae_values = [r['metrics'].get('mae', float('nan')) for r in results_list]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax1.barh(names, mae_values, color=colors)
    ax1.set_xlabel('MAE (sat/vbyte)')
    ax1.set_title('Mean Absolute Error')
    for bar, val in zip(bars, mae_values):
        ax1.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}', va='center', fontsize=8)

    # Panel 2: Inclusion accuracy
    ax2 = fig.add_subplot(2, 2, 2)
    inc_values = [r['metrics'].get('inclusion_accuracy', 0) for r in results_list]
    bars = ax2.barh(names, inc_values, color=colors)
    ax2.set_xlabel('Inclusion Accuracy')
    ax2.set_xlim(0, 1)
    ax2.set_title('Inclusion Accuracy (0.05 percentile)')
    for bar, val in zip(bars, inc_values):
        ax2.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1%}', va='center', fontsize=8)

    # Panel 3: RMSE comparison
    ax3 = fig.add_subplot(2, 2, 3)
    rmse_values = [r['metrics'].get('rmse', float('nan')) for r in results_list]
    bars = ax3.barh(names, rmse_values, color=colors)
    ax3.set_xlabel('RMSE (sat/vbyte)')
    ax3.set_title('Root Mean Squared Error')
    for bar, val in zip(bars, rmse_values):
        ax3.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}', va='center', fontsize=8)

    # Panel 4: Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Create ranking table
    table_data = []
    for i, r in enumerate(sorted(results_list, key=lambda x: x['metrics'].get('mae', float('inf')))):
        table_data.append([
            i + 1,
            r['predictor_name'],
            r['horizon'],
            f"{r['metrics'].get('mae', float('nan')):.1f}",
            f"{r['metrics'].get('inclusion_accuracy', 0):.1%}",
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Rank', 'Predictor', 'Horizon', 'MAE', 'Inclusion'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Ranking by MAE')

    plt.tight_layout()
    return fig
