"""
Detailed evaluation of trained LSTM model.

This script loads the best LSTM checkpoint and evaluates it on:
- Facility level (individual facilities)
- Sector level (aggregated by sector)
- National level (total emissions)

This matches the evaluation structure of Prophet and ARIMA baselines.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
from typing import Dict

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.lstm_model import LSTMLightningModule
from utils.data_loader import load_processed_data
from utils.metrics import compute_all_metrics


def evaluate_lstm_on_series(model: LSTMLightningModule,
                            series: pd.Series,
                            sequence_length: int = 5,
                            test_years: tuple = (2021, 2023),
                            device: str = 'cpu') -> Dict:
    """Evaluate LSTM on a single time series."""
    
    # Get test years
    test_series = series[(series.index >= test_years[0]) & (series.index <= test_years[1])]
    
    # Need enough history for sequence
    min_year = test_years[0] - sequence_length
    available_series = series[series.index >= min_year]
    
    if len(test_series) < 1 or len(available_series) < sequence_length + 1:
        return None
    
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for test_year in test_series.index:
            # Get input sequence
            input_years = range(test_year - sequence_length, test_year)
            
            # Check if all input years are available
            if not all(year in series.index for year in input_years):
                continue
            
            input_seq = [series[year] for year in input_years]
            
            # Prepare input tensor [1, seq_len, 1] and move to same device as model
            x = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
            
            # Predict
            y_pred = model(x)
            pred_value = y_pred.squeeze().cpu().item()
            
            predictions.append(pred_value)
            actuals.append(series[test_year])
    
    if len(predictions) == 0:
        return None
    
    # Compute metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    metrics = compute_all_metrics(actuals, predictions)
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'num_predictions': len(predictions)
    }


def main():
    """Run detailed LSTM evaluation."""
    
    print("="*60)
    print("LSTM Detailed Evaluation")
    print("="*60)
    
    # Load config
    import yaml
    with open('configs/lstm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Find best checkpoint
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoints = list(checkpoint_dir.glob('lstm-*.ckpt'))
    
    if not checkpoints:
        print("No LSTM checkpoints found!")
        print(f"Looked in: {checkpoint_dir}")
        return
    
    # Get the best one (lowest val_loss in filename)
    best_checkpoint = sorted(checkpoints, key=lambda x: float(str(x).split('val_loss=')[1].split('-')[0].split('.ckpt')[0]))[0]
    
    print(f"\nLoading checkpoint: {best_checkpoint}")
    
    # Load model
    model = LSTMLightningModule.load_from_checkpoint(
        best_checkpoint,
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        forecast_horizon=config['data']['forecast_horizon'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Determine device and move model
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading data...")
    data_dir = config['data']['data_dir']
    facility_series, sector_series, national_series, facility_metadata = load_processed_data(data_dir)
    
    print(f"Loaded {len(facility_series)} facility time series")
    print(f"Loaded {len(sector_series)} sector time series")
    
    # Test configuration
    sequence_length = config['data']['sequence_length']
    test_years = tuple(config['data']['test_years'])
    
    print(f"\nTest configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Test years: {test_years[0]}-{test_years[1]}")
    
    # Evaluate on facility level
    print("\n" + "="*60)
    print("Evaluating on Facility Level")
    print("="*60)
    
    facility_results = {}
    all_predictions = []
    all_actuals = []
    
    for i, (facility_id, series) in enumerate(facility_series.items()):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(facility_series)}")
        
        result = evaluate_lstm_on_series(model, series, sequence_length, test_years, device)
        
        if result:
            facility_results[facility_id] = result
            all_predictions.extend(result['predictions'])
            all_actuals.extend(result['actuals'])
    
    # Compute overall facility metrics
    if all_predictions:
        overall_metrics = compute_all_metrics(
            np.array(all_actuals),
            np.array(all_predictions)
        )
        
        print(f"\nEvaluated {len(facility_results)} facilities")
        print("\nOverall Facility-Level Metrics:")
        for name, value in overall_metrics.items():
            print(f"  {name}: {value:.4f}")
    else:
        overall_metrics = {}
        print("\nNo facility results!")
    
    # Evaluate on sector level
    print("\n" + "="*60)
    print("Evaluating on Sector Level")
    print("="*60)
    
    sector_results = {}
    sector_predictions = []
    sector_actuals = []
    
    for sector, series in sector_series.items():
        result = evaluate_lstm_on_series(model, series, sequence_length, test_years, device)
        
        if result:
            sector_results[sector] = result
            sector_predictions.extend(result['predictions'])
            sector_actuals.extend(result['actuals'])
    
    if sector_predictions:
        sector_overall_metrics = compute_all_metrics(
            np.array(sector_actuals),
            np.array(sector_predictions)
        )
        
        print(f"\nEvaluated {len(sector_results)} sectors")
        print("\nOverall Sector-Level Metrics:")
        for name, value in sector_overall_metrics.items():
            print(f"  {name}: {value:.4f}")
    else:
        sector_overall_metrics = {}
    
    # Evaluate on national level
    print("\n" + "="*60)
    print("Evaluating on National Level")
    print("="*60)
    
    national_result = evaluate_lstm_on_series(model, national_series, sequence_length, test_years, device)
    
    if national_result:
        print("\nNational-Level Metrics:")
        for name, value in national_result['metrics'].items():
            print(f"  {name}: {value:.4f}")
    
    # Save results
    results_dir = Path(config['logging']['checkpoint_dir']).parent / 'metrics'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save facility-level results
    facility_metrics_path = results_dir / 'lstm_facility_metrics.json'
    with open(facility_metrics_path, 'w') as f:
        json.dump({
            'overall_metrics': {k: float(v) for k, v in overall_metrics.items()},
            'num_series': len(facility_results),
            'config': config
        }, f, indent=2)
    
    # Save sector-level results
    sector_metrics_path = results_dir / 'lstm_sector_metrics.json'
    with open(sector_metrics_path, 'w') as f:
        json.dump({
            'overall_metrics': {k: float(v) for k, v in sector_overall_metrics.items()},
            'num_series': len(sector_results),
            'config': config
        }, f, indent=2)
    
    # Save national-level results
    if national_result:
        national_metrics_path = results_dir / 'lstm_national_metrics.json'
        with open(national_metrics_path, 'w') as f:
            json.dump({
                'metrics': {k: float(v) for k, v in national_result['metrics'].items()},
                'predictions': national_result['predictions'].tolist(),
                'actuals': national_result['actuals'].tolist(),
                'config': config
            }, f, indent=2)
    
    # Save sample predictions
    sample_facilities = list(facility_results.items())[:10]
    predictions_path = results_dir / 'lstm_sample_predictions.json'
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
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Facility metrics: {facility_metrics_path}")
    print(f"  - Sector metrics:   {sector_metrics_path}")
    print(f"  - National metrics: {national_metrics_path}")
    print(f"  - Sample predictions: {predictions_path}")
    print("="*60)


if __name__ == "__main__":
    main()

