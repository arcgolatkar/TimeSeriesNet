"""
Training script for ARIMA baseline using statsmodels only (no pmdarima).

This is a simplified version that doesn't require pmdarima installation.
"""

import numpy as np
import pandas as pd
import yaml
import argparse
import json
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.arima_model_statsmodels import evaluate_arima_on_multiple_series, evaluate_arima_on_series
from utils.data_loader import load_processed_data


def train_arima(config_path: str):
    """Train ARIMA baseline model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Training ARIMA Baseline (statsmodels)")
    print("="*60)
    
    # Set random seed
    np.random.seed(config['reproducibility']['seed'])
    
    # Load data
    print("\nLoading data...")
    data_dir = config['data'].get('data_dir', 'src/models/data/processed')
    facility_series, sector_series, national_series, facility_metadata = load_processed_data(data_dir)
    
    print(f"Loaded {len(facility_series)} facility time series")
    print(f"Loaded {len(sector_series)} sector time series")
    print(f"Loaded national time series with {len(national_series)} years")
    
    # Training configuration
    train_years = tuple(config['data']['train_years'])
    val_years = tuple(config['data']['val_years'])
    test_years = tuple(config['data']['test_years'])
    
    print(f"\nSplit configuration:")
    print(f"  Train: {train_years[0]}-{train_years[1]}")
    print(f"  Val:   {val_years[0]}-{val_years[1]}")
    print(f"  Test:  {test_years[0]}-{test_years[1]}")
    
    # Use fixed ARIMA parameters (simple and reliable)
    order = tuple(config['model'].get('order', [1, 1, 1]))
    seasonal_order = (0, 0, 0, 0)  # No seasonality for annual data
    
    print(f"\nARIMA configuration:")
    print(f"  order: {order}")
    print(f"  seasonal_order: {seasonal_order}")
    print("  Note: Using fixed parameters (no auto-search)")
    
    # Evaluate on facility-level series
    print("\n" + "="*60)
    print("Evaluating ARIMA on Facility-Level Time Series")
    print("="*60)
    
    facility_results = evaluate_arima_on_multiple_series(
        facility_series,
        train_years=train_years,
        test_years=test_years,
        max_series=None,
        order=order,
        seasonal_order=seasonal_order
    )
    
    # Evaluate on sector-level series
    print("\n" + "="*60)
    print("Evaluating ARIMA on Sector-Level Time Series")
    print("="*60)
    
    sector_results = evaluate_arima_on_multiple_series(
        sector_series,
        train_years=train_years,
        test_years=test_years,
        order=order,
        seasonal_order=seasonal_order
    )
    
    # Evaluate on national-level series
    print("\n" + "="*60)
    print("Evaluating ARIMA on National-Level Time Series")
    print("="*60)
    
    national_result = evaluate_arima_on_series(
        national_series,
        train_years=train_years,
        test_years=test_years,
        order=order,
        seasonal_order=seasonal_order
    )
    
    if national_result:
        print("\nNational-Level Metrics:")
        for name, value in national_result['metrics'].items():
            print(f"  {name}: {value:.4f}")
    
    # Save results
    results_dir = Path(config['logging']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save facility-level results
    facility_metrics_path = results_dir / 'arima_facility_metrics.json'
    with open(facility_metrics_path, 'w') as f:
        json.dump({
            'overall_metrics': {k: float(v) for k, v in facility_results['overall_metrics'].items()},
            'num_series': facility_results['num_series'],
            'config': config,
            'note': 'Using statsmodels directly (no pmdarima)'
        }, f, indent=2)
    
    # Save sector-level results
    sector_metrics_path = results_dir / 'arima_sector_metrics.json'
    with open(sector_metrics_path, 'w') as f:
        json.dump({
            'overall_metrics': {k: float(v) for k, v in sector_results['overall_metrics'].items()},
            'num_series': sector_results['num_series'],
            'config': config
        }, f, indent=2)
    
    # Save national-level results
    if national_result:
        national_metrics_path = results_dir / 'arima_national_metrics.json'
        with open(national_metrics_path, 'w') as f:
            json.dump({
                'metrics': {k: float(v) for k, v in national_result['metrics'].items()},
                'predictions': national_result['predictions'].tolist(),
                'actuals': national_result['actuals'].tolist(),
                'config': config
            }, f, indent=2)
    
    # Save detailed predictions for a sample
    sample_facilities = list(facility_results['series_results'].items())[:10]
    predictions_path = results_dir / 'arima_sample_predictions.json'
    with open(predictions_path, 'w') as f:
        sample_data = {}
        for facility_id, result in sample_facilities:
            sample_data[str(facility_id)] = {
                'predictions': result['predictions'].tolist(),
                'actuals': result['actuals'].tolist(),
                'metrics': {k: float(v) for k, v in result['metrics'].items()}
            }
        json.dump(sample_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Facility metrics: {facility_metrics_path}")
    print(f"  - Sector metrics:   {sector_metrics_path}")
    print(f"  - National metrics: {national_metrics_path}")
    print(f"  - Sample predictions: {predictions_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train ARIMA Baseline (Simple)')
    parser.add_argument('--config', type=str, default='configs/arima_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train_arima(args.config)


if __name__ == "__main__":
    main()

