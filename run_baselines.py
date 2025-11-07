"""
Run all baseline models for GHG emissions forecasting.

This script runs LSTM, Prophet, and ARIMA baselines and generates a comparison report.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}")
        print(e.stdout)
        return False


def load_metrics(results_dir):
    """Load metrics from all baseline models."""
    results_dir = Path(results_dir)
    metrics = {}
    
    # Load ARIMA metrics
    arima_facility = results_dir / 'arima_facility_metrics.json'
    if arima_facility.exists():
        with open(arima_facility, 'r') as f:
            data = json.load(f)
            metrics['ARIMA'] = data['overall_metrics']
            metrics['ARIMA']['num_series'] = data['num_series']
    
    # Load Prophet metrics
    prophet_facility = results_dir / 'prophet_facility_metrics.json'
    if prophet_facility.exists():
        with open(prophet_facility, 'r') as f:
            data = json.load(f)
            metrics['Prophet'] = data['overall_metrics']
            metrics['Prophet']['num_series'] = data['num_series']
    
    # Load LSTM metrics (if available from tensorboard logs)
    # This would require parsing TensorBoard logs or test outputs
    
    return metrics


def create_comparison_table(metrics, output_path):
    """Create a comparison table of all models."""
    if not metrics:
        print("No metrics to compare")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Reorder columns for better readability
    metric_order = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2', 'num_series']
    existing_cols = [col for col in metric_order if col in df.columns]
    df = df[existing_cols]
    
    # Format numbers
    for col in df.columns:
        if col != 'num_series':
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    # Save to CSV
    df.to_csv(output_path)
    
    # Print table
    print("\n" + "="*70)
    print("MODEL COMPARISON - Facility-Level Forecasting")
    print("="*70)
    print(df.to_string())
    print(f"\nComparison table saved to: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run all baseline models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines
  python run_baselines.py --all
  
  # Run only LSTM
  python run_baselines.py --lstm
  
  # Run Prophet and ARIMA
  python run_baselines.py --prophet --arima
  
  # Skip comparison at the end
  python run_baselines.py --all --no-compare
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all baseline models')
    parser.add_argument('--lstm', action='store_true',
                       help='Run LSTM baseline')
    parser.add_argument('--prophet', action='store_true',
                       help='Run Prophet baseline')
    parser.add_argument('--arima', action='store_true',
                       help='Run ARIMA baseline')
    parser.add_argument('--no-compare', action='store_true',
                       help='Skip creating comparison table')
    parser.add_argument('--results-dir', type=str, default='results/metrics',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # If no specific model selected, show help
    if not (args.all or args.lstm or args.prophet or args.arima):
        parser.print_help()
        print("\nError: Please specify which models to run (--all, --lstm, --prophet, or --arima)")
        sys.exit(1)
    
    # Determine which models to run
    run_lstm = args.all or args.lstm
    run_prophet = args.all or args.prophet
    run_arima = args.all or args.arima
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("BASELINE MODELS TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {results_dir}")
    print(f"\nModels to run:")
    print(f"  LSTM:    {run_lstm}")
    print(f"  Prophet: {run_prophet}")
    print(f"  ARIMA:   {run_arima}")
    print("="*70)
    
    results = {}
    
    # Run ARIMA (using simple statsmodels version - no pmdarima needed)
    if run_arima:
        success = run_command(
            [sys.executable, 'src/baselines/train_arima_simple.py', '--config', 'configs/arima_config.yaml'],
            "Training ARIMA Baseline (statsmodels)"
        )
        results['ARIMA'] = success
    
    # Run Prophet
    if run_prophet:
        success = run_command(
            [sys.executable, 'src/baselines/train_prophet.py', '--config', 'configs/prophet_config.yaml'],
            "Training Prophet Baseline"
        )
        results['Prophet'] = success
    
    # Run LSTM
    if run_lstm:
        success = run_command(
            [sys.executable, 'src/baselines/train_lstm.py', '--config', 'configs/lstm_config.yaml'],
            "Training LSTM Baseline"
        )
        results['LSTM'] = success
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModel Status:")
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {model:10s}: {status}")
    
    # Create comparison table
    if not args.no_compare:
        print("\n" + "="*70)
        print("GENERATING COMPARISON TABLE")
        print("="*70)
        
        metrics = load_metrics(results_dir)
        if metrics:
            comparison_path = results_dir / 'baseline_comparison.csv'
            create_comparison_table(metrics, comparison_path)
        else:
            print("No metrics files found. Models may still be training or may have failed.")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults location: {results_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Check individual model results in results/metrics/")
    print("  2. Review comparison table: results/metrics/baseline_comparison.csv")
    print("  3. For LSTM, check TensorBoard logs: results/logs/lstm_baseline/")
    print("  4. Run model evaluation: python src/utils/evaluate_models.py")
    print("="*70)


if __name__ == "__main__":
    main()

