"""
Visualization utilities for time series forecasting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: Optional[np.ndarray] = None,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None
):
    """
    Plot predictions against actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        years: Year labels (optional)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if years is None:
        years = np.arange(len(y_true))
    
    ax.plot(years, y_true, marker='o', label='Actual', linewidth=2, markersize=8)
    ax.plot(years, y_pred, marker='s', label='Predicted', linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Log Scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['MAE', 'RMSE', 'sMAPE', 'MASE'],
    save_path: Optional[str] = None
):
    """
    Create bar plots comparing models across metrics.
    
    Args:
        comparison_df: DataFrame with columns ['Model', metric1, metric2, ...]
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            ax = axes[i]
            comparison_df.plot(
                x='Model',
                y=metric,
                kind='bar',
                ax=ax,
                legend=False,
                color=sns.color_palette("husl", len(comparison_df))
            )
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hierarchical_emissions(
    national_series: pd.Series,
    sector_series: Dict[str, pd.Series],
    save_path: Optional[str] = None
):
    """
    Plot hierarchical emissions (national and sector levels).
    
    Args:
        national_series: National-level time series
        sector_series: Dictionary of sector time series
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # National level
    national_series.plot(ax=ax1, marker='o', linewidth=2.5, color='darkblue')
    ax1.set_title('National-Level GHG Emissions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Log(CO2e Emissions)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Sector level
    for sector, series in sector_series.items():
        ax2.plot(series.index, series.values, marker='o', label=sector, linewidth=2)
    
    ax2.set_title('Sector-Level GHG Emissions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Log(CO2e Emissions)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metric_name: str = 'Loss',
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        metric_name: Name of the metric
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, marker='o', label='Training', linewidth=2)
    ax.plot(epochs, val_losses, marker='s', label='Validation', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot error distribution histogram.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Path to save figure
    """
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_metrics_table(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a formatted metrics comparison table.
    
    Args:
        results_dict: Dictionary {model_name: {metric: value}}
        save_path: Path to save table as CSV
        
    Returns:
        DataFrame with comparison table
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    
    if save_path:
        df.to_csv(save_path)
    
    return df


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    years: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention matrix [seq_len, seq_len]
        years: Year labels (optional)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if years is None:
        years = np.arange(attention_weights.shape[0])
    
    sns.heatmap(
        attention_weights,
        xticklabels=years,
        yticklabels=years,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test visualizations with synthetic data
    np.random.seed(42)
    
    years = np.arange(2010, 2024)
    y_true = np.linspace(100, 150, len(years)) + np.random.normal(0, 5, len(years))
    y_pred = y_true + np.random.normal(0, 3, len(years))
    
    print("Testing visualization functions...")
    
    # Test predictions plot
    plot_predictions_vs_actual(y_true, y_pred, years, title="Test Predictions")
    
    # Test error distribution
    plot_error_distribution(y_true, y_pred)
    
    # Test metrics table
    results = {
        'ARIMA': {'MAE': 3.5, 'RMSE': 4.2, 'sMAPE': 5.1},
        'LSTM': {'MAE': 2.8, 'RMSE': 3.5, 'sMAPE': 4.2},
        'MST': {'MAE': 2.1, 'RMSE': 2.8, 'sMAPE': 3.5}
    }
    df = create_metrics_table(results)
    print("\nMetrics Table:")
    print(df)

