"""
Load EPA GHGRP emissions data from pre-processed files.

This script works with the EPA data pipeline output structure:
- out_epa/raw/: Raw EPA downloads
- out_epa/: Processed files including training_facility_year.csv

If processed files don't exist, creates sample data for development.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EPADataLoader:
    """Load pre-processed EPA GHGRP emissions data."""
    
    def __init__(self, data_dir: str = "out_epa"):
        """
        Initialize the EPA data loader.
        
        Args:
            data_dir: Directory containing processed EPA data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    
    def check_processed_files_exist(self) -> bool:
        """Check if all required processed files exist."""
        required_files = [
            "training_facility_year.csv",
            "sector_year_totals.csv",
            "splits_years.json"
        ]
        
        for file in required_files:
            if not (self.data_dir / file).exists():
                return False
        return True
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Download a file from EPA and save it.
        
        Args:
            url: URL to download from
            filename: Name to save the file as
            
        Returns:
            Downloaded data as DataFrame or None if failed
        """
        output_path = self.output_dir / filename
        
        try:
            logger.info(f"Downloading data from {url}...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Save raw data
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Saved data to {output_path}")
            
            # Load as DataFrame
            df = pd.read_csv(output_path)
            logger.info(f"Loaded {len(df)} records")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def download_all(self) -> dict:
        """
        Download all EPA GHGRP datasets.
        
        Returns:
            Dictionary of DataFrames for each dataset
        """
        data = {}
        
        for name, endpoint in self.data_files.items():
            url = self.base_url + endpoint
            filename = f"{name}.csv"
            
            df = self.download_file(url, filename)
            if df is not None:
                data[name] = df
        
        return data
    
    def create_sample_data(self):
        """
        Create sample/synthetic data for development and testing.
        
        This is useful when EPA API is unavailable or for initial development.
        """
        logger.info("Creating sample data for development...")
        
        # Sample facility data
        facilities = []
        sectors = ['Power Plants', 'Petroleum and Natural Gas Systems', 
                   'Chemicals', 'Waste', 'Metals']
        states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH']
        
        facility_id = 1000000
        for sector in sectors:
            for state in states:
                # Create 5-10 facilities per sector per state
                num_facilities = 7
                for _ in range(num_facilities):
                    facilities.append({
                        'FACILITY_ID': facility_id,
                        'FACILITY_NAME': f'Facility {facility_id}',
                        'STATE': state,
                        'CITY': f'City{facility_id % 100}',
                        'ZIP': f'{10000 + (facility_id % 90000)}',
                        'LATITUDE': 30.0 + (facility_id % 20),
                        'LONGITUDE': -120.0 + (facility_id % 60),
                        'INDUSTRY_TYPE': sector,
                    })
                    facility_id += 1
        
        facilities_df = pd.DataFrame(facilities)
        
        # Sample emissions data (2010-2023)
        emissions = []
        years = list(range(2010, 2024))
        
        for _, facility in facilities_df.iterrows():
            base_emission = 10000 + (facility['FACILITY_ID'] % 50000)
            
            for year in years:
                # Add trend and random variation
                trend = (year - 2010) * (100 if facility['FACILITY_ID'] % 2 == 0 else -80)
                variation = (facility['FACILITY_ID'] * year) % 5000 - 2500
                co2_emission = max(1000, base_emission + trend + variation)
                
                emissions.append({
                    'FACILITY_ID': facility['FACILITY_ID'],
                    'REPORTING_YEAR': year,
                    'GHG_NAME': 'Carbon Dioxide',
                    'GHG_QUANTITY': co2_emission,
                    'UNIT': 'Metric Tons CO2e',
                })
                
                # Add CH4 and N2O for some facilities
                if facility['FACILITY_ID'] % 3 == 0:
                    emissions.append({
                        'FACILITY_ID': facility['FACILITY_ID'],
                        'REPORTING_YEAR': year,
                        'GHG_NAME': 'Methane',
                        'GHG_QUANTITY': co2_emission * 0.05,
                        'UNIT': 'Metric Tons CO2e',
                    })
        
        emissions_df = pd.DataFrame(emissions)
        
        # Save sample data
        facilities_df.to_csv(self.output_dir / 'facilities.csv', index=False)
        emissions_df.to_csv(self.output_dir / 'emissions.csv', index=False)
        
        logger.info(f"Created {len(facilities_df)} facilities and {len(emissions_df)} emission records")
        logger.info(f"Years: {emissions_df['REPORTING_YEAR'].min()} - {emissions_df['REPORTING_YEAR'].max()}")
        logger.info(f"Sectors: {facilities_df['INDUSTRY_TYPE'].unique()}")
        
        return {'facilities': facilities_df, 'emissions': emissions_df}


def main():
    """Main function to download EPA GHGRP data."""
    downloader = EPADataDownloader()
    
    # Try to download real data
    logger.info("Attempting to download EPA GHGRP data...")
    data = downloader.download_all()
    
    # If download fails, create sample data
    if not data or len(data) == 0:
        logger.warning("Could not download EPA data. Creating sample data instead...")
        data = downloader.create_sample_data()
    
    # Display summary
    if data:
        logger.info("\n" + "="*50)
        logger.info("Data Download Summary")
        logger.info("="*50)
        for name, df in data.items():
            logger.info(f"{name}: {len(df)} records")
            logger.info(f"Columns: {', '.join(df.columns[:5])}...")
        logger.info("="*50)
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    main()

