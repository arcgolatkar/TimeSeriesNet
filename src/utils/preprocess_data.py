"""
Preprocess EPA GHGRP emissions data for time series forecasting.

This script:
1. Loads raw EPA data
2. Cleans and handles missing values
3. Aggregates emissions to CO2e equivalent
4. Creates hierarchical levels (facility, sector, national)
5. Splits data into train/validation/test sets
6. Saves processed datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GHGDataPreprocessor:
    """Preprocess EPA GHGRP emissions data for forecasting."""
    
    # GWP (Global Warming Potential) values for converting to CO2e
    GWP_VALUES = {
        'Carbon Dioxide': 1.0,
        'CO2': 1.0,
        'Methane': 25.0,
        'CH4': 25.0,
        'Nitrous Oxide': 298.0,
        'N2O': 298.0,
    }
    
    def __init__(self, raw_data_dir: str = "out_epa/output", 
                 processed_data_dir: str = "data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.facilities_df = None
        self.emissions_df = None
        self.processed_data = {}
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw facilities and emissions data."""
        logger.info("Loading raw data...")
        
        facilities_path = self.raw_data_dir / 'facilities.csv'
        emissions_path = self.raw_data_dir / 'emissions.csv'
        
        if not facilities_path.exists() or not emissions_path.exists():
            raise FileNotFoundError(
                f"Raw data files not found in {self.raw_data_dir}. "
                f"Please run download_data.py first."
            )
        
        self.facilities_df = pd.read_csv(facilities_path)
        self.emissions_df = pd.read_csv(emissions_path)
        
        logger.info(f"Loaded {len(self.facilities_df)} facilities")
        logger.info(f"Loaded {len(self.emissions_df)} emission records")
        
        return self.facilities_df, self.emissions_df
    
    def clean_data(self):
        """Clean and prepare the data."""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        self.emissions_df = self.emissions_df.drop_duplicates(
            subset=['FACILITY_ID', 'REPORTING_YEAR', 'GHG_NAME']
        )
        
        # Handle missing values in emissions
        self.emissions_df['GHG_QUANTITY'] = self.emissions_df['GHG_QUANTITY'].fillna(0)
        
        # Ensure positive emissions
        self.emissions_df['GHG_QUANTITY'] = self.emissions_df['GHG_QUANTITY'].abs()
        
        # Filter valid years (2010-2023)
        self.emissions_df = self.emissions_df[
            (self.emissions_df['REPORTING_YEAR'] >= 2010) &
            (self.emissions_df['REPORTING_YEAR'] <= 2023)
        ]
        
        logger.info(f"After cleaning: {len(self.emissions_df)} emission records")
    
    def convert_to_co2e(self):
        """Convert all emissions to CO2 equivalent."""
        logger.info("Converting emissions to CO2e...")
        
        # Map GHG names to GWP values
        def get_gwp(ghg_name):
            for key, value in self.GWP_VALUES.items():
                if key.lower() in str(ghg_name).lower():
                    return value
            return 1.0  # Default to CO2 if unknown
        
        self.emissions_df['GWP'] = self.emissions_df['GHG_NAME'].apply(get_gwp)
        self.emissions_df['CO2E'] = self.emissions_df['GHG_QUANTITY'] * self.emissions_df['GWP']
        
        logger.info("Conversion complete")
    
    def aggregate_facility_level(self) -> pd.DataFrame:
        """Aggregate emissions at facility level (sum all GHGs per facility per year)."""
        logger.info("Aggregating to facility level...")
        
        facility_data = self.emissions_df.groupby(['FACILITY_ID', 'REPORTING_YEAR']).agg({
            'CO2E': 'sum'
        }).reset_index()
        
        # Merge with facility metadata
        facility_data = facility_data.merge(
            self.facilities_df[['FACILITY_ID', 'INDUSTRY_TYPE', 'STATE']],
            on='FACILITY_ID',
            how='left'
        )
        
        facility_data = facility_data.rename(columns={
            'CO2E': 'EMISSIONS',
            'REPORTING_YEAR': 'YEAR'
        })
        
        logger.info(f"Facility-level data: {len(facility_data)} records")
        return facility_data
    
    def aggregate_sector_level(self, facility_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate emissions at sector level."""
        logger.info("Aggregating to sector level...")
        
        sector_data = facility_data.groupby(['INDUSTRY_TYPE', 'YEAR']).agg({
            'EMISSIONS': 'sum',
            'FACILITY_ID': 'count'
        }).reset_index()
        
        sector_data = sector_data.rename(columns={
            'FACILITY_ID': 'NUM_FACILITIES'
        })
        
        logger.info(f"Sector-level data: {len(sector_data)} records")
        return sector_data
    
    def aggregate_national_level(self, facility_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate emissions at national level."""
        logger.info("Aggregating to national level...")
        
        national_data = facility_data.groupby('YEAR').agg({
            'EMISSIONS': 'sum',
            'FACILITY_ID': 'count'
        }).reset_index()
        
        national_data = national_data.rename(columns={
            'FACILITY_ID': 'NUM_FACILITIES'
        })
        
        logger.info(f"National-level data: {len(national_data)} records")
        return national_data
    
    def create_time_series_datasets(self, facility_data: pd.DataFrame,
                                   sector_data: pd.DataFrame,
                                   national_data: pd.DataFrame):
        """Create time series datasets for each hierarchical level."""
        logger.info("Creating time series datasets...")
        
        # Facility-level time series (one per facility)
        facility_series = {}
        for facility_id, group in facility_data.groupby('FACILITY_ID'):
            series = group.sort_values('YEAR').set_index('YEAR')['EMISSIONS']
            # Only keep facilities with sufficient data (at least 8 years)
            if len(series) >= 8:
                facility_series[facility_id] = series
        
        logger.info(f"Created {len(facility_series)} facility time series")
        
        # Sector-level time series (one per sector)
        sector_series = {}
        for sector, group in sector_data.groupby('INDUSTRY_TYPE'):
            series = group.sort_values('YEAR').set_index('YEAR')['EMISSIONS']
            sector_series[sector] = series
        
        logger.info(f"Created {len(sector_series)} sector time series")
        
        # National-level time series
        national_series = national_data.sort_values('YEAR').set_index('YEAR')['EMISSIONS']
        
        logger.info(f"Created national time series with {len(national_series)} years")
        
        return facility_series, sector_series, national_series
    
    def normalize_data(self, facility_series: Dict, sector_series: Dict, 
                      national_series: pd.Series):
        """Normalize the time series data using log transformation."""
        logger.info("Normalizing data...")
        
        # Log transform (adding 1 to avoid log(0))
        facility_series_norm = {
            k: np.log1p(v) for k, v in facility_series.items()
        }
        sector_series_norm = {
            k: np.log1p(v) for k, v in sector_series.items()
        }
        national_series_norm = np.log1p(national_series)
        
        # Store normalization statistics
        self.norm_stats = {
            'method': 'log1p',
            'facility_ranges': {
                k: (v.min(), v.max()) for k, v in facility_series.items()
            },
            'sector_ranges': {
                k: (v.min(), v.max()) for k, v in sector_series.items()
            },
            'national_range': (national_series.min(), national_series.max())
        }
        
        return facility_series_norm, sector_series_norm, national_series_norm
    
    def save_processed_data(self, facility_series: Dict, sector_series: Dict,
                           national_series: pd.Series, facility_data: pd.DataFrame):
        """Save all processed datasets."""
        logger.info("Saving processed data...")
        
        # Save time series as pickle files
        with open(self.processed_data_dir / 'facility_series.pkl', 'wb') as f:
            pickle.dump(facility_series, f)
        
        with open(self.processed_data_dir / 'sector_series.pkl', 'wb') as f:
            pickle.dump(sector_series, f)
        
        with open(self.processed_data_dir / 'national_series.pkl', 'wb') as f:
            pickle.dump(national_series, f)
        
        # Save metadata
        with open(self.processed_data_dir / 'norm_stats.pkl', 'wb') as f:
            pickle.dump(self.norm_stats, f)
        
        # Save facility metadata for reference
        facility_metadata = facility_data[['FACILITY_ID', 'INDUSTRY_TYPE', 'STATE']].drop_duplicates()
        facility_metadata.to_csv(self.processed_data_dir / 'facility_metadata.csv', index=False)
        
        logger.info(f"Saved processed data to {self.processed_data_dir}")
    
    def process_all(self):
        """Run the complete preprocessing pipeline."""
        logger.info("="*50)
        logger.info("Starting EPA GHGRP Data Preprocessing")
        logger.info("="*50)
        
        # Load data
        self.load_raw_data()
        
        # Clean and convert
        self.clean_data()
        self.convert_to_co2e()
        
        # Aggregate to different levels
        facility_data = self.aggregate_facility_level()
        sector_data = self.aggregate_sector_level(facility_data)
        national_data = self.aggregate_national_level(facility_data)
        
        # Create time series
        facility_series, sector_series, national_series = self.create_time_series_datasets(
            facility_data, sector_data, national_data
        )
        
        # Normalize
        facility_series_norm, sector_series_norm, national_series_norm = self.normalize_data(
            facility_series, sector_series, national_series
        )
        
        # Save
        self.save_processed_data(
            facility_series_norm, sector_series_norm, national_series_norm, facility_data
        )
        
        # Print summary statistics
        self._print_summary(facility_series, sector_series, national_series)
        
        logger.info("="*50)
        logger.info("Preprocessing Complete!")
        logger.info("="*50)
    
    def _print_summary(self, facility_series, sector_series, national_series):
        """Print summary statistics of processed data."""
        logger.info("\n" + "="*50)
        logger.info("Data Summary")
        logger.info("="*50)
        logger.info(f"Facility-level: {len(facility_series)} facilities")
        logger.info(f"Sector-level: {len(sector_series)} sectors")
        logger.info(f"National-level: {len(national_series)} years")
        logger.info(f"Year range: {national_series.index.min()} - {national_series.index.max()}")
        logger.info(f"\nSectors: {list(sector_series.keys())}")
        logger.info(f"\nNational total emissions range: "
                   f"{national_series.min():.2e} - {national_series.max():.2e} metric tons CO2e")
        logger.info("="*50)


def main():
    """Main function to preprocess EPA GHGRP data."""
    preprocessor = GHGDataPreprocessor()
    preprocessor.process_all()


if __name__ == "__main__":
    main()

