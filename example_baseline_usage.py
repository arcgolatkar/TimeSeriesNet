"""
Example script showing how to use each baseline model independently.

This demonstrates the API for each model without the full training pipeline.
Useful for understanding the models or quick prototyping.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path('src').absolute()))


def example_prophet():
    """Example: Train and predict with Prophet."""
    print("\n" + "="*60)
    print("Example: Prophet Forecasting")
    print("="*60)
    
    from baselines.prophet_model import ProphetForecaster
    
    # Create synthetic time series
    years = np.arange(2010, 2021)
    values = 1000 + np.cumsum(np.random.randn(len(years)) * 50)
    series = pd.Series(values, index=years)
    
    print(f"Training data: {len(series)} years ({series.index[0]}-{series.index[-1]})")
    
    # Train Prophet
    forecaster = ProphetForecaster(
        growth='linear',
        changepoint_prior_scale=0.05,
        yearly_seasonality=False
    )
    
    forecaster.fit(series, verbose=False)
    
    # Forecast next 3 years
    last_date = pd.to_datetime(str(series.index[-1]), format='%Y')
    predictions = forecaster.predict(steps=3, last_date=last_date)
    
    print(f"\nForecasts for {series.index[-1]+1} to {series.index[-1]+3}:")
    for i, pred in enumerate(predictions):
        year = series.index[-1] + i + 1
        print(f"  {year}: {pred:.2f}")
    
    print("\n✓ Prophet example complete")


def example_arima():
    """Example: Train and predict with ARIMA."""
    print("\n" + "="*60)
    print("Example: ARIMA Forecasting")
    print("="*60)
    
    from baselines.arima_model import ARIMAForecaster
    
    # Create synthetic time series
    years = np.arange(2010, 2021)
    values = 1000 + np.cumsum(np.random.randn(len(years)) * 50)
    series = pd.Series(values, index=years)
    
    print(f"Training data: {len(series)} years ({series.index[0]}-{series.index[-1]})")
    
    # Train ARIMA with auto search
    forecaster = ARIMAForecaster(
        auto_arima_search=True,
        seasonal=False,
        max_p=3,
        max_d=2,
        max_q=3
    )
    
    print("Searching for best ARIMA parameters...")
    forecaster.fit(series, verbose=False)
    
    # Forecast next 3 years
    predictions = forecaster.predict(steps=3)
    
    print(f"\nForecasts for {series.index[-1]+1} to {series.index[-1]+3}:")
    for i, pred in enumerate(predictions):
        year = series.index[-1] + i + 1
        print(f"  {year}: {pred:.2f}")
    
    print("\n✓ ARIMA example complete")


def example_lstm():
    """Example: Create and use LSTM model."""
    print("\n" + "="*60)
    print("Example: LSTM Forecasting")
    print("="*60)
    
    import torch
    from baselines.lstm_model import LSTMForecaster
    
    # Create synthetic data
    sequence_length = 10
    forecast_horizon = 1
    batch_size = 4
    
    # Random input sequences [batch, seq_len, features]
    x = torch.randn(batch_size, sequence_length, 1)
    
    print(f"Input shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features: 1 (emissions)")
    
    # Create model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=forecast_horizon
    )
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x)
    
    print(f"\nOutput shape: {predictions.shape}")
    print(f"  Predictions: {predictions.shape[0]} sequences")
    print(f"  Horizon: {predictions.shape[1]} steps ahead")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {num_params:,} parameters")
    
    print("\n✓ LSTM example complete")


def example_metrics():
    """Example: Compute forecasting metrics."""
    print("\n" + "="*60)
    print("Example: Computing Metrics")
    print("="*60)
    
    from utils.metrics import compute_all_metrics
    import torch
    
    # Synthetic predictions vs actuals
    y_true = torch.tensor([100., 120., 110., 130., 125.])
    y_pred = torch.tensor([95., 125., 105., 128., 130.])
    
    print(f"Actuals:     {y_true.numpy()}")
    print(f"Predictions: {y_pred.numpy()}")
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name:10s}: {value:.4f}")
    
    print("\n✓ Metrics example complete")


def example_data_loading():
    """Example: Load processed EPA data."""
    print("\n" + "="*60)
    print("Example: Loading EPA Data")
    print("="*60)
    
    from utils.data_loader import load_processed_data
    
    data_dir = 'src/models/data/processed'
    
    print(f"Loading data from: {data_dir}")
    
    try:
        facility_series, sector_series, national_series, facility_metadata = \
            load_processed_data(data_dir)
        
        print(f"\n✓ Loaded data successfully:")
        print(f"  Facilities: {len(facility_series)} time series")
        print(f"  Sectors: {len(sector_series)} time series")
        print(f"  National: {len(national_series)} years")
        print(f"  Metadata: {len(facility_metadata)} facilities")
        
        # Show sample facility
        if facility_series:
            sample_id = list(facility_series.keys())[0]
            sample_ts = facility_series[sample_id]
            print(f"\nSample facility: {sample_id}")
            print(f"  Years: {sample_ts.index.min()} - {sample_ts.index.max()}")
            print(f"  Mean emissions: {sample_ts.mean():.2f}")
            print(f"  Std emissions: {sample_ts.std():.2f}")
        
        # Show sectors
        if sector_series:
            print(f"\nSectors: {', '.join(sector_series.keys())}")
        
        print("\n✓ Data loading example complete")
        
    except FileNotFoundError as e:
        print(f"\n✗ Data not found: {e}")
        print("Run preprocessing first to generate data files.")
        return False
    
    return True


def main():
    """Run all examples."""
    print("="*60)
    print("BASELINE MODELS - USAGE EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate how to use each baseline model.")
    print("They use synthetic data and don't require preprocessed files.")
    
    # Run examples with synthetic data
    try:
        example_prophet()
        example_arima()
        example_lstm()
        example_metrics()
    except Exception as e:
        print(f"\n✗ Error in examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Try to load real data if available
    print("\n" + "="*60)
    print("Attempting to load real EPA data...")
    print("="*60)
    
    try:
        example_data_loading()
    except Exception as e:
        print(f"\n✗ Could not load EPA data: {e}")
        print("This is expected if data hasn't been preprocessed yet.")
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run full training: python run_baselines.py --all")
    print("  2. See guide: BASELINE_MODELS_GUIDE.md")
    print("  3. Quick start: BASELINE_QUICKSTART.md")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

