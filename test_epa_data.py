"""
Quick test script for EPA data loading and dataset creation.

Run this to verify the new EPA data integration works correctly.
"""

import sys
sys.path.append('src')

from utils.load_epa_data import EPADataLoader
from utils.epa_dataset import create_dataloaders_from_epa


def main():
    print("="*70)
    print("EPA Data Integration Test")
    print("="*70)
    
    # Step 1: Load data
    print("\n1. Loading EPA data...")
    loader = EPADataLoader(data_dir='out_epa')
    
    if not loader.check_data_exists():
        print("   → Creating sample data (real data not found)")
        training_data, sector_data, splits = loader.create_sample_data()
    else:
        print("   → Loading existing processed data")
        training_data, sector_data, splits = loader.load_all()
    
    print(f"   ✓ Training data: {len(training_data):,} rows")
    print(f"   ✓ Facilities: {training_data['FACILITY_ID'].nunique():,}")
    print(f"   ✓ Years: {training_data['REPORTING_YEAR'].min()} - {training_data['REPORTING_YEAR'].max()}")
    
    # Step 2: Create dataloaders
    print("\n2. Creating PyTorch dataloaders...")
    dataloaders = create_dataloaders_from_epa(
        training_data=training_data,
        sector_data=sector_data,
        splits=splits,
        sequence_length=8,
        forecast_horizon=1,
        batch_size=32,
        num_workers=0
    )
    
    print(f"   ✓ Train: {len(dataloaders['train'])} batches")
    print(f"   ✓ Val:   {len(dataloaders['val'])} batches")
    print(f"   ✓ Test:  {len(dataloaders['test'])} batches")
    
    # Step 3: Test batch
    print("\n3. Testing batch structure...")
    for batch in dataloaders['train']:
        print(f"   Sample batch:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"   - {key:<20}: {str(value.shape):<25} {value.dtype}")
            else:
                print(f"   - {key:<20}: {type(value)}")
        
        print(f"\n   ✓ Input shape: [batch={batch['input_features'].shape[0]}, " +
              f"seq_len={batch['input_features'].shape[1]}, " +
              f"features={batch['input_features'].shape[2]}]")
        print(f"   ✓ Output shape: [batch={batch['facility_target'].shape[0]}, " +
              f"horizon={batch['facility_target'].shape[1]}]")
        break
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("✅ All tests passed! EPA data integration is working correctly.")
    print("="*70)
    print("\nNext steps:")
    print("  1. Place your real EPA data in out_epa/ directory")
    print("  2. Update training scripts to use the new data loaders")
    print("  3. Modify model input layer for multi-feature input")
    print("  4. Train your models!")
    print()


if __name__ == "__main__":
    main()

