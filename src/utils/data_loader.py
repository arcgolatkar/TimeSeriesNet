"""
PyTorch data loaders for time series forecasting.

Loads preprocessed EPA GHGRP data and creates datasets for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting."""
    
    def __init__(self, 
                 series_dict: Dict[str, pd.Series],
                 sequence_length: int = 10,
                 forecast_horizon: int = 1,
                 start_year: int = 2010,
                 end_year: int = 2023):
        """
        Initialize time series dataset.
        
        Args:
            series_dict: Dictionary of time series {id: pd.Series}
            sequence_length: Number of historical timesteps to use
            forecast_horizon: Number of future timesteps to predict
            start_year: Start year for data split
            end_year: End year for data split
        """
        self.series_dict = series_dict
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.start_year = start_year
        self.end_year = end_year
        
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Create sliding window samples from time series."""
        for series_id, series in self.series_dict.items():
            # Filter by year range
            series = series[(series.index >= self.start_year) & 
                          (series.index <= self.end_year)]
            
            # Skip if not enough data
            if len(series) < self.sequence_length + self.forecast_horizon:
                continue
            
            # Create sliding windows
            values = series.values
            years = series.index.values
            
            for i in range(len(values) - self.sequence_length - self.forecast_horizon + 1):
                input_seq = values[i:i + self.sequence_length]
                target_seq = values[i + self.sequence_length:
                                   i + self.sequence_length + self.forecast_horizon]
                input_years = years[i:i + self.sequence_length]
                target_years = years[i + self.sequence_length:
                                    i + self.sequence_length + self.forecast_horizon]
                
                self.samples.append({
                    'series_id': series_id,
                    'input': input_seq,
                    'target': target_seq,
                    'input_years': input_years,
                    'target_years': target_years
                })
        
        logger.info(f"Created {len(self.samples)} samples from {len(self.series_dict)} series")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'series_id': sample['series_id'],
            'input': torch.FloatTensor(sample['input']).unsqueeze(-1),  # [seq_len, 1]
            'target': torch.FloatTensor(sample['target']).unsqueeze(-1),  # [horizon, 1]
            'input_years': torch.LongTensor(sample['input_years']),
            'target_years': torch.LongTensor(sample['target_years'])
        }


class HierarchicalTimeSeriesDataset(Dataset):
    """Dataset that includes hierarchical information (facility, sector, national)."""
    
    def __init__(self,
                 facility_series: Dict[str, pd.Series],
                 sector_series: Dict[str, pd.Series],
                 national_series: pd.Series,
                 facility_metadata: pd.DataFrame,
                 sequence_length: int = 10,
                 forecast_horizon: int = 1,
                 start_year: int = 2010,
                 end_year: int = 2023):
        """
        Initialize hierarchical time series dataset.
        
        Args:
            facility_series: Facility-level time series
            sector_series: Sector-level time series
            national_series: National-level time series
            facility_metadata: Metadata linking facilities to sectors
            sequence_length: Number of historical timesteps
            forecast_horizon: Number of future timesteps
            start_year: Start year for data split
            end_year: End year for data split
        """
        self.facility_series = facility_series
        self.sector_series = sector_series
        self.national_series = national_series
        self.facility_metadata = facility_metadata
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.start_year = start_year
        self.end_year = end_year
        
        self.samples = []
        self._prepare_hierarchical_samples()
    
    def _prepare_hierarchical_samples(self):
        """Create samples with hierarchical context."""
        for facility_id, facility_ts in self.facility_series.items():
            # Get facility metadata
            metadata = self.facility_metadata[
                self.facility_metadata['FACILITY_ID'] == facility_id
            ]
            if metadata.empty:
                continue
            
            sector = metadata.iloc[0]['INDUSTRY_TYPE']
            
            # Get corresponding sector and national series
            sector_ts = self.sector_series.get(sector)
            if sector_ts is None:
                continue
            
            # Filter by year range
            facility_ts = facility_ts[(facility_ts.index >= self.start_year) & 
                                     (facility_ts.index <= self.end_year)]
            sector_ts = sector_ts[(sector_ts.index >= self.start_year) & 
                                 (sector_ts.index <= self.end_year)]
            national_ts = self.national_series[(self.national_series.index >= self.start_year) & 
                                              (self.national_series.index <= self.end_year)]
            
            # Align time series (use intersection of years)
            common_years = set(facility_ts.index) & set(sector_ts.index) & set(national_ts.index)
            common_years = sorted(list(common_years))
            
            if len(common_years) < self.sequence_length + self.forecast_horizon:
                continue
            
            facility_values = [facility_ts[year] for year in common_years]
            sector_values = [sector_ts[year] for year in common_years]
            national_values = [national_ts[year] for year in common_years]
            
            # Create sliding windows
            for i in range(len(common_years) - self.sequence_length - self.forecast_horizon + 1):
                self.samples.append({
                    'facility_id': facility_id,
                    'sector': sector,
                    'facility_input': np.array(facility_values[i:i + self.sequence_length]),
                    'facility_target': np.array(facility_values[
                        i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
                    ]),
                    'sector_input': np.array(sector_values[i:i + self.sequence_length]),
                    'sector_target': np.array(sector_values[
                        i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
                    ]),
                    'national_input': np.array(national_values[i:i + self.sequence_length]),
                    'national_target': np.array(national_values[
                        i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
                    ]),
                    'years': common_years[i:i + self.sequence_length + self.forecast_horizon]
                })
        
        logger.info(f"Created {len(self.samples)} hierarchical samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'facility_id': sample['facility_id'],
            'sector': sample['sector'],
            'facility_input': torch.FloatTensor(sample['facility_input']).unsqueeze(-1),
            'facility_target': torch.FloatTensor(sample['facility_target']).unsqueeze(-1),
            'sector_input': torch.FloatTensor(sample['sector_input']).unsqueeze(-1),
            'sector_target': torch.FloatTensor(sample['sector_target']).unsqueeze(-1),
            'national_input': torch.FloatTensor(sample['national_input']).unsqueeze(-1),
            'national_target': torch.FloatTensor(sample['national_target']).unsqueeze(-1),
        }


def load_processed_data(data_dir: str = "data/processed") -> Tuple:
    """
    Load all processed data from disk.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (facility_series, sector_series, national_series, facility_metadata)
    """
    data_dir = Path(data_dir)
    
    with open(data_dir / 'facility_series.pkl', 'rb') as f:
        facility_series = pickle.load(f)
    
    with open(data_dir / 'sector_series.pkl', 'rb') as f:
        sector_series = pickle.load(f)
    
    with open(data_dir / 'national_series.pkl', 'rb') as f:
        national_series = pickle.load(f)
    
    facility_metadata = pd.read_csv(data_dir / 'facility_metadata.csv')
    
    return facility_series, sector_series, national_series, facility_metadata


def create_dataloaders(
    data_dir: str = "data/processed",
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    train_years: Tuple[int, int] = (2010, 2019),
    val_years: Tuple[int, int] = (2020, 2021),
    test_years: Tuple[int, int] = (2022, 2023),
    hierarchical: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing processed data
        sequence_length: Length of input sequences
        forecast_horizon: Length of forecast
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_years: (start, end) years for training
        val_years: (start, end) years for validation
        test_years: (start, end) years for testing
        hierarchical: Whether to use hierarchical dataset
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Load data
    facility_series, sector_series, national_series, facility_metadata = load_processed_data(data_dir)
    
    # Create datasets
    if hierarchical:
        train_dataset = HierarchicalTimeSeriesDataset(
            facility_series, sector_series, national_series, facility_metadata,
            sequence_length, forecast_horizon, train_years[0], train_years[1]
        )
        val_dataset = HierarchicalTimeSeriesDataset(
            facility_series, sector_series, national_series, facility_metadata,
            sequence_length, forecast_horizon, val_years[0], val_years[1]
        )
        test_dataset = HierarchicalTimeSeriesDataset(
            facility_series, sector_series, national_series, facility_metadata,
            sequence_length, forecast_horizon, test_years[0], test_years[1]
        )
    else:
        train_dataset = TimeSeriesDataset(
            facility_series, sequence_length, forecast_horizon, 
            train_years[0], train_years[1]
        )
        val_dataset = TimeSeriesDataset(
            facility_series, sequence_length, forecast_horizon,
            val_years[0], val_years[1]
        )
        test_dataset = TimeSeriesDataset(
            facility_series, sequence_length, forecast_horizon,
            test_years[0], test_years[1]
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
    dataloaders = create_dataloaders(
        batch_size=32,
        hierarchical=True
    )
    
    # Print sample batch
    for batch in dataloaders['train']:
        print("\nSample batch:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        break

