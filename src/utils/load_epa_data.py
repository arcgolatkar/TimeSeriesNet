"""
Load EPA GHGRP emissions data from pre-processed pipeline output.

Expected directory structure:
out_epa/
├─ raw/                                   # Raw EPA downloads
│  ├─ pub_dim_facility.csv
│  ├─ pub_facts_sector_ghg_emission.csv
│  └─ pub_facts_subp_ghg_emission_YYYY.csv
├─ training_facility_year.csv             # Main training data
├─ sector_year_totals.csv                 # Sector/national rollups
├─ facility_year_wide.csv                 # Wide format
├─ facility_year_gas_long.csv             # Long format
└─ splits_years.json                      # Train/val/test splits
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EPADataLoader:
    """Load pre-processed EPA GHGRP data for MST training."""
    
    def __init__(self, data_dir: str = "out_epa"):
        """
        Initialize EPA data loader.
        
        Args:
            data_dir: Root directory containing processed EPA files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
    
    def check_data_exists(self) -> bool:
        """Check if required processed files exist."""
        required = [
            "training_facility_year.csv",
            "sector_year_totals.csv",
            "splits_years.json"
        ]
        
        missing = [f for f in required if not (self.data_dir / f).exists()]
        
        if missing:
            logger.warning(f"Missing files: {missing}")
            return False
        
        return True
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load main training data with engineered features.
        
        Returns:
            DataFrame with columns:
            - FACILITY_ID: Facility identifier
            - REPORTING_YEAR: Year
            - co2e_total: Target variable (total CO2e)
            - qty_<gas>: Mass quantities per gas
            - co2e_<gas>: CO2e per gas
            - Lag features: co2e_total_lag1, co2e_total_lag3, etc.
            - Rolling stats: roll3_mean_co2e, roll3_std_co2e
            - Facility attributes: name, state, county, city, zip, lat, lon
        """
        path = self.data_dir / "training_facility_year.csv"
        logger.info(f"Loading training data from {path}")
        
        df = pd.read_csv(path)
        # rename year column to REPORTING_YEAR
        if 'year' in df.columns:
            df = df.rename(columns={'year': 'REPORTING_YEAR'})
        if 'fac_id' in df.columns:
            df = df.rename(columns={'fac_id': 'FACILITY_ID'})
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Years: {df['REPORTING_YEAR'].min()} - {df['REPORTING_YEAR'].max()}")
        logger.info(f"Facilities: {df['FACILITY_ID'].nunique()}")
        
        return df
    
    def load_sector_totals(self) -> pd.DataFrame:
        """
        Load sector and national level aggregates.
        
        Returns:
            DataFrame with sector-level emissions by year
        """
        path = self.data_dir / "sector_year_totals.csv"
        logger.info(f"Loading sector totals from {path}")
        
        df = pd.read_csv(path)
        # rename sector_name to sector if needed
        if 'sector_name' in df.columns:
            df = df.rename(columns={'sector_name': 'sector'})
        logger.info(f"Loaded {len(df)} sector-year records")
        
        return df
    
    def load_splits(self) -> Dict:
        """
        Load train/val/test year splits.
        
        Returns:
            Dict with keys 'train', 'val', 'test' containing year lists
        """
        path = self.data_dir / "splits_years.json"
        logger.info(f"Loading splits from {path}")
        
        with open(path, 'r') as f:
            splits = json.load(f)
        
        logger.info(f"Train years: {splits['train']}")
        logger.info(f"Val years: {splits['val']}")
        logger.info(f"Test years: {splits['test']}")
        
        return splits
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load all required data files.
        
        Returns:
            Tuple of (training_data, sector_totals, splits)
        """
        training_data = self.load_training_data()
        sector_totals = self.load_sector_totals()
        splits = self.load_splits()
        
        return training_data, sector_totals, splits
    
    def create_sample_data(self):
        """
        Create sample data in the expected format for development/testing.
        
        Generates:
        - training_facility_year.csv
        - sector_year_totals.csv  
        - splits_years.json
        """
        logger.info("Creating sample data...")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample parameters
        n_facilities = 200
        years = list(range(2010, 2024))
        sectors = ['Power Plants', 'Petroleum and Natural Gas', 'Chemicals', 
                   'Waste', 'Metals']
        states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH']
        gases = ['co2', 'ch4', 'n2o']
        
        # Create training data
        training_records = []
        
        for fid in range(1000000, 1000000 + n_facilities):
            sector = np.random.choice(sectors)
            state = np.random.choice(states)
            base_emission = 10000 + np.random.randint(0, 50000)
            
            for year_idx, year in enumerate(years):
                # Generate emission with trend and noise
                trend = year_idx * (100 if fid % 2 == 0 else -80)
                seasonal = 500 * np.sin(year_idx * 2 * np.pi / 7)  # 7-year cycle
                noise = np.random.normal(0, 500)
                total_co2e = max(1000, base_emission + trend + seasonal + noise)
                
                # Per-gas breakdown
                co2_fraction = 0.85 + np.random.uniform(-0.1, 0.1)
                ch4_fraction = 0.10 + np.random.uniform(-0.05, 0.05)
                n2o_fraction = 1 - co2_fraction - ch4_fraction
                
                co2e_co2 = total_co2e * co2_fraction
                co2e_ch4 = total_co2e * ch4_fraction
                co2e_n2o = total_co2e * n2o_fraction
                
                # Lags (using previous years from this loop)
                lag1 = training_records[-1]['co2e_total'] if training_records and \
                       training_records[-1]['FACILITY_ID'] == fid else np.nan
                lag3 = training_records[-3]['co2e_total'] if len(training_records) >= 3 and \
                       training_records[-3]['FACILITY_ID'] == fid else np.nan
                
                # Rolling stats (simplified)
                roll3_mean = lag1 if not np.isnan(lag1) else total_co2e
                roll3_std = abs(np.random.normal(0, total_co2e * 0.05))
                
                record = {
                    'FACILITY_ID': fid,
                    'REPORTING_YEAR': year,
                    'co2e_total': total_co2e,
                    'qty_co2': co2e_co2 / 1.0,  # CO2 mass (GWP=1)
                    'qty_ch4': co2e_ch4 / 25.0,  # CH4 mass (GWP=25)
                    'qty_n2o': co2e_n2o / 298.0,  # N2O mass (GWP=298)
                    'co2e_co2': co2e_co2,
                    'co2e_ch4': co2e_ch4,
                    'co2e_n2o': co2e_n2o,
                    'co2e_total_lag1': lag1,
                    'co2e_total_lag3': lag3,
                    'roll3_mean_co2e': roll3_mean,
                    'roll3_std_co2e': roll3_std,
                    'n_gases_reporting': 3,
                    'facility_name': f'Facility {fid}',
                    'state': state,
                    'sector': sector,
                    'city': f'City{fid % 100}',
                    'zip': f'{10000 + (fid % 90000)}',
                    'latitude': 30.0 + (fid % 20),
                    'longitude': -120.0 + (fid % 60),
                }
                
                training_records.append(record)
        
        training_df = pd.DataFrame(training_records)
        training_df.to_csv(self.data_dir / 'training_facility_year.csv', index=False)
        logger.info(f"Created training_facility_year.csv: {len(training_df)} rows")
        
        # Create sector totals
        sector_records = []
        for year in years:
            for sector in sectors:
                sector_total = training_df[
                    (training_df['REPORTING_YEAR'] == year) &
                    (training_df['sector'] == sector)
                ]['co2e_total'].sum()
                
                sector_records.append({
                    'REPORTING_YEAR': year,
                    'sector': sector,
                    'co2e_total': sector_total
                })
            
            # National total
            national_total = training_df[
                training_df['REPORTING_YEAR'] == year
            ]['co2e_total'].sum()
            
            sector_records.append({
                'REPORTING_YEAR': year,
                'sector': 'NATIONAL_TOTAL',
                'co2e_total': national_total
            })
        
        sector_df = pd.DataFrame(sector_records)
        sector_df.to_csv(self.data_dir / 'sector_year_totals.csv', index=False)
        logger.info(f"Created sector_year_totals.csv: {len(sector_df)} rows")
        
        # Create splits
        splits = {
            'train': years[:10],  # 2010-2019
            'val': years[10:12],  # 2020-2021
            'test': years[12:]    # 2022-2023
        }
        
        with open(self.data_dir / 'splits_years.json', 'w') as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Created splits_years.json")
        
        logger.info("Sample data creation complete!")
        
        return training_df, sector_df, splits


def main():
    """Main function to load or create EPA data."""
    loader = EPADataLoader()
    
    if not loader.check_data_exists():
        logger.warning("Processed data files not found. Creating sample data...")
        loader.create_sample_data()
    
    # Load all data
    training_data, sector_totals, splits = loader.load_all()
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("EPA Data Summary")
    logger.info("="*60)
    logger.info(f"Training data: {len(training_data)} rows")
    logger.info(f"  Facilities: {training_data['FACILITY_ID'].nunique()}")
    logger.info(f"  Years: {training_data['REPORTING_YEAR'].min()} - {training_data['REPORTING_YEAR'].max()}")
    logger.info(f"  Features: {len(training_data.columns)} columns")
    logger.info(f"\nSector totals: {len(sector_totals)} rows")
    logger.info(f"  Sectors: {sector_totals['sector'].nunique()}")
    logger.info(f"\nSplits:")
    logger.info(f"  Train: {len(splits['train'])} years")
    logger.info(f"  Val: {len(splits['val'])} years")
    logger.info(f"  Test: {len(splits['test'])} years")
    logger.info("="*60)


if __name__ == "__main__":
    main()

