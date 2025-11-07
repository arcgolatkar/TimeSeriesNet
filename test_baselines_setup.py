"""
Quick test script to verify baseline models setup.

This script checks:
1. Required packages are installed
2. Data files exist
3. Config files are valid
4. Models can be instantiated
"""

import sys
from pathlib import Path
import yaml


def check_packages():
    """Check if required packages are installed."""
    print("\n" + "="*60)
    print("Checking Required Packages")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'pytorch_lightning': 'PyTorch Lightning',
        'prophet': 'Prophet',
        'pmdarima': 'pmdarima (auto_arima)',
        'statsmodels': 'statsmodels',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll packages installed ✓")
    return True


def check_data_files():
    """Check if processed data files exist."""
    print("\n" + "="*60)
    print("Checking Data Files")
    print("="*60)
    
    data_dir = Path("src/models/data/processed")
    
    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print("\nData needs to be preprocessed first.")
        return False
    
    required_files = [
        'facility_series.pkl',
        'sector_series.pkl',
        'national_series.pkl',
        'facility_metadata.csv'
    ]
    
    missing = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {filename} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        print("Run preprocessing first.")
        return False
    
    print("\nAll data files found ✓")
    return True


def check_configs():
    """Check if config files are valid."""
    print("\n" + "="*60)
    print("Checking Configuration Files")
    print("="*60)
    
    configs = {
        'LSTM': 'configs/lstm_config.yaml',
        'Prophet': 'configs/prophet_config.yaml',
        'ARIMA': 'configs/arima_config.yaml'
    }
    
    all_valid = True
    for model, config_path in configs.items():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required = ['model', 'data', 'logging', 'reproducibility']
            missing = [s for s in required if s not in config]
            
            if missing:
                print(f"  ✗ {model}: Missing sections: {missing}")
                all_valid = False
            else:
                data_dir = config['data'].get('data_dir', 'N/A')
                print(f"  ✓ {model} (data_dir: {data_dir})")
        
        except FileNotFoundError:
            print(f"  ✗ {model}: Config file not found")
            all_valid = False
        except Exception as e:
            print(f"  ✗ {model}: Error loading config - {e}")
            all_valid = False
    
    if all_valid:
        print("\nAll configs valid ✓")
    
    return all_valid


def check_model_imports():
    """Check if model modules can be imported."""
    print("\n" + "="*60)
    print("Checking Model Imports")
    print("="*60)
    
    # Add src to path
    sys.path.insert(0, str(Path('src').absolute()))
    
    models = {
        'LSTM': ('baselines.lstm_model', 'LSTMForecaster'),
        'Prophet': ('baselines.prophet_model', 'ProphetForecaster'),
        'ARIMA': ('baselines.arima_model', 'ARIMAForecaster'),
    }
    
    all_good = True
    for name, (module, cls) in models.items():
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"  ✓ {name} model")
        except Exception as e:
            print(f"  ✗ {name} model: {e}")
            all_good = False
    
    if all_good:
        print("\nAll models can be imported ✓")
    
    return all_good


def test_data_loading():
    """Test loading a small sample of data."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path('src').absolute()))
        from utils.data_loader import load_processed_data
        
        facility_series, sector_series, national_series, facility_metadata = \
            load_processed_data('src/models/data/processed')
        
        print(f"  ✓ Loaded {len(facility_series)} facility time series")
        print(f"  ✓ Loaded {len(sector_series)} sector time series")
        print(f"  ✓ Loaded national time series ({len(national_series)} years)")
        print(f"  ✓ Loaded facility metadata ({len(facility_metadata)} rows)")
        
        # Check a sample
        if facility_series:
            sample_id = next(iter(facility_series))
            sample_ts = facility_series[sample_id]
            print(f"\n  Sample facility: {sample_id}")
            print(f"    Years: {sample_ts.index.min()} - {sample_ts.index.max()}")
            print(f"    Data points: {len(sample_ts)}")
        
        print("\nData loading successful ✓")
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("BASELINE MODELS SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'Packages': check_packages(),
        'Data Files': check_data_files(),
        'Config Files': check_configs(),
        'Model Imports': check_model_imports(),
        'Data Loading': test_data_loading()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("All checks passed! ✓")
        print("\nYou're ready to train baseline models:")
        print("  python run_baselines.py --all")
    else:
        print("Some checks failed. ✗")
        print("\nPlease fix the issues above before training.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

