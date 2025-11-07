"""
Comprehensive model evaluation script.

This script evaluates all models (baselines and MST) and generates
comparison results, visualizations, and metrics for the interim report.
"""

import numpy as np
import pandas as pd
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict
import sys

from metrics import compute_all_metrics
from visualizations import (
    plot_model_comparison,
    plot_predictions_vs_actual,
    plot_error_distribution,
    create_metrics_table
)


class ModelEvaluator:
    """Evaluate and compare all models."""
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.figures_dir = self.results_dir / "figures"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def load_baseline_results(self, model_name: str, results_path: str):
        """Load results from baseline models."""
        print(f"Loading {model_name} results...")
        
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'overall_metrics' in data:
            self.results[model_name] = data['overall_metrics']
        else:
            print(f"Warning: No overall metrics found for {model_name}")
    
    def load_dl_results(self, model_name: str, metrics_path: str):
        """Load results from deep learning models (LSTM, MST)."""
        print(f"Loading {model_name} results...")
        
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            # Extract test metrics
            test_metrics = {
                k.replace('test_', ''): v
                for k, v in data.items()
                if k.startswith('test_')
            }
            
            if test_metrics:
                self.results[model_name] = test_metrics
            else:
                print(f"Warning: No test metrics found for {model_name}")
        except FileNotFoundError:
            print(f"Warning: Metrics file not found for {model_name}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all models."""
        if not self.results:
            print("No results loaded. Cannot create comparison table.")
            return pd.DataFrame()
        
        # Standardize metric names
        standard_metrics = ['MAE', 'RMSE', 'sMAPE', 'MASE', 'R2']
        
        comparison_data = {
            'Model': [],
        }
        for metric in standard_metrics:
            comparison_data[metric] = []
        
        for model_name, metrics in self.results.items():
            comparison_data['Model'].append(model_name)
            
            for metric in standard_metrics:
                # Try different possible keys
                value = metrics.get(metric) or metrics.get(metric.lower())
                comparison_data[metric].append(value if value is not None else np.nan)
        
        df = pd.DataFrame(comparison_data)
        df = df.round(4)
        
        return df
    
    def generate_summary_report(self):
        """Generate summary report with all results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        comparison_df = self.create_comparison_table()
        
        if comparison_df.empty:
            print("No results to summarize.")
            return
        
        # Save comparison table
        csv_path = self.metrics_dir / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nSaved comparison table to: {csv_path}")
        
        # Print comparison table
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model for each metric
        print("\nBest Models per Metric:")
        for metric in ['MAE', 'RMSE', 'sMAPE', 'MASE']:
            if metric in comparison_df.columns:
                valid_df = comparison_df[comparison_df[metric].notna()]
                if not valid_df.empty:
                    best_idx = valid_df[metric].idxmin()
                    best_model = valid_df.loc[best_idx, 'Model']
                    best_value = valid_df.loc[best_idx, metric]
                    print(f"  {metric}: {best_model} ({best_value:.4f})")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Model comparison plot
        plot_model_comparison(
            comparison_df,
            metrics=['MAE', 'RMSE', 'sMAPE', 'MASE'],
            save_path=self.figures_dir / "model_comparison.png"
        )
        print(f"  - Saved model comparison plot")
        
        # Save summary statistics
        summary = {
            'num_models_evaluated': len(self.results),
            'models': list(self.results.keys()),
            'metrics_available': list(comparison_df.columns[1:]),
        }
        
        # Best model overall (by MAE)
        if 'MAE' in comparison_df.columns:
            valid_df = comparison_df[comparison_df['MAE'].notna()]
            if not valid_df.empty:
                best_idx = valid_df['MAE'].idxmin()
                summary['best_model'] = valid_df.loc[best_idx, 'Model']
                summary['best_mae'] = float(valid_df.loc[best_idx, 'MAE'])
        
        summary_path = self.metrics_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  - Saved evaluation summary to: {summary_path}")
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Results directory')
    parser.add_argument('--arima_results', type=str, default='../results/metrics/arima_results.pkl',
                       help='Path to ARIMA results')
    parser.add_argument('--prophet_results', type=str, default='../results/metrics/prophet_results.pkl',
                       help='Path to Prophet results')
    parser.add_argument('--lstm_results', type=str, default='../results/metrics/lstm_metrics.json',
                       help='Path to LSTM results')
    parser.add_argument('--mst_results', type=str, default='../results/metrics/mst_metrics.json',
                       help='Path to MST results')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(results_dir=args.results_dir)
    
    # Load results from all models
    # ARIMA
    if Path(args.arima_results).exists():
        evaluator.load_baseline_results('ARIMA', args.arima_results)
    else:
        print(f"ARIMA results not found at {args.arima_results}")
    
    # Prophet
    if Path(args.prophet_results).exists():
        evaluator.load_baseline_results('Prophet', args.prophet_results)
    else:
        print(f"Prophet results not found at {args.prophet_results}")
    
    # LSTM
    if Path(args.lstm_results).exists():
        evaluator.load_dl_results('LSTM', args.lstm_results)
    else:
        print(f"LSTM results not found at {args.lstm_results}")
    
    # MST
    if Path(args.mst_results).exists():
        evaluator.load_dl_results('Multi-Scale Transformer', args.mst_results)
    else:
        print(f"MST results not found at {args.mst_results}")
    
    # Generate summary report
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()

