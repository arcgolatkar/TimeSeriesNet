"""
PyTorch Dataset for EPA GHGRP training data.

Works with training_facility_year.csv format with engineered features.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EPAFacilityDataset(Dataset):
    """PyTorch Dataset for facility-level time series forecasting."""
    
    def __init__(self,
                 training_data: pd.DataFrame,
                 sector_data: pd.DataFrame,
                 years: List[int],
                 sequence_length: int = 8,
                 forecast_horizon: int = 1,
                 target_col: str = 'co2e_total',
                 feature_cols: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            training_data: DataFrame from training_facility_year.csv
            sector_data: DataFrame from sector_year_totals.csv
            years: List of years to include in this split
            sequence_length: Number of historical years (input window)
            forecast_horizon: Number of years to predict
            target_col: Column to predict (default: co2e_total)
            feature_cols: List of feature columns (if None, uses all numeric)
        """
        self.training_data = training_data
        self.sector_data = sector_data
        self.years = years
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Use co2e and qty columns as features
            self.feature_cols = [col for col in training_data.columns 
                                if col.startswith(('co2e_', 'qty_')) and 
                                col != target_col]
        else:
            self.feature_cols = feature_cols
        
        logger.info(f"Using {len(self.feature_cols)} feature columns")
        
        # Filter data for specified years
        self.data = training_data[training_data['REPORTING_YEAR'].isin(years)].copy()
        
        # Create samples
        self.samples = []
        self._prepare_samples()
        
        logger.info(f"Created {len(self.samples)} samples from {len(years)} years")
    
    def _prepare_samples(self):
        """Create sliding window samples from time series."""
        # Group by facility
        for facility_id, group in self.data.groupby('FACILITY_ID'):
            # Sort by year
            group = group.sort_values('REPORTING_YEAR')
            
            # Skip if not enough data
            if len(group) < self.sequence_length + self.forecast_horizon:
                continue
            
            years = group['REPORTING_YEAR'].values
            features = group[self.feature_cols].values
            targets = group[self.target_col].values
            
            # Get sector for this facility
            sector = group['sector'].iloc[0] if 'sector' in group.columns else None
            
            # Create sliding windows
            for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                input_features = features[i:i + self.sequence_length]
                input_targets = targets[i:i + self.sequence_length]
                output_targets = targets[i + self.sequence_length:
                                       i + self.sequence_length + self.forecast_horizon]
                
                input_years = years[i:i + self.sequence_length]
                output_years = years[i + self.sequence_length:
                                   i + self.sequence_length + self.forecast_horizon]
                
                # Get corresponding sector targets
                sector_targets = []
                if sector:
                    for year in output_years:
                        sector_val = self.sector_data[
                            (self.sector_data['sector'] == sector) &
                            (self.sector_data['REPORTING_YEAR'] == year)
                        ]['co2e_total'].values
                        sector_targets.append(sector_val[0] if len(sector_val) > 0 else 0)
                else:
                    sector_targets = [0] * len(output_years)
                
                # Get national targets
                national_targets = []
                for year in output_years:
                    nat_val = self.sector_data[
                        (self.sector_data['sector'] == 'NATIONAL_TOTAL') &
                        (self.sector_data['REPORTING_YEAR'] == year)
                    ]['co2e_total'].values
                    national_targets.append(nat_val[0] if len(nat_val) > 0 else 0)
                
                self.samples.append({
                    'facility_id': facility_id,
                    'sector': sector,
                    'input_features': input_features,
                    'input_targets': input_targets,
                    'facility_target': output_targets,
                    'sector_target': np.array(sector_targets),
                    'national_target': np.array(national_targets),
                    'input_years': input_years,
                    'output_years': output_years
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'facility_id': sample['facility_id'],
            'sector': sample['sector'],
            # Input: [seq_len, n_features]
            'input_features': torch.FloatTensor(sample['input_features']),
            # Historical targets: [seq_len, 1]
            'input_targets': torch.FloatTensor(sample['input_targets']).unsqueeze(-1),
            # Predictions at each level: [horizon, 1]
            'facility_target': torch.FloatTensor(sample['facility_target']).unsqueeze(-1),
            'sector_target': torch.FloatTensor(sample['sector_target']).unsqueeze(-1),
            'national_target': torch.FloatTensor(sample['national_target']).unsqueeze(-1),
            'input_years': torch.LongTensor(sample['input_years']),
            'output_years': torch.LongTensor(sample['output_years'])
        }


def create_dataloaders_from_epa(
    training_data: pd.DataFrame,
    sector_data: pd.DataFrame,
    splits: Dict[str, List[int]],
    sequence_length: int = 8,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders from EPA data.
    
    Args:
        training_data: DataFrame from training_facility_year.csv
        sector_data: DataFrame from sector_year_totals.csv
        splits: Dict with 'train', 'val', 'test' year lists
        sequence_length: Input window size
        forecast_horizon: Prediction horizon
        batch_size: Batch size
        num_workers: Number of data loading workers
        feature_cols: Feature columns to use
        
    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    # Create datasets
    train_dataset = EPAFacilityDataset(
        training_data, sector_data, splits['train'],
        sequence_length, forecast_horizon, feature_cols=feature_cols
    )
    
    val_dataset = EPAFacilityDataset(
        training_data, sector_data, splits['val'],
        sequence_length, forecast_horizon, feature_cols=feature_cols
    )
    
    test_dataset = EPAFacilityDataset(
        training_data, sector_data, splits['test'],
        sequence_length, forecast_horizon, feature_cols=feature_cols
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test data loading
    from load_epa_data import EPADataLoader
    
    loader = EPADataLoader()
    
    if not loader.check_data_exists():
        logger.info("Creating sample data for testing...")
        loader.create_sample_data()
    
    training_data, sector_data, splits = loader.load_all()
    
    # Create dataloaders
    dataloaders = create_dataloaders_from_epa(
        training_data, sector_data, splits,
        sequence_length=8,
        forecast_horizon=1,
        batch_size=32
    )
    
    # Test a batch
    for batch in dataloaders['train']:
        print("\nSample batch:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        break

