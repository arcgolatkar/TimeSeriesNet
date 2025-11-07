import pickle
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")   # or the path you used in config

# --- Facility series
with open(data_dir / "facility_series.pkl", "rb") as f:
    facility_series = pickle.load(f)
print("Facility series:", len(facility_series), "entries")
first_key = next(iter(facility_series))
print("Example facility ID:", first_key)
print(facility_series[first_key].head())
print(facility_series[first_key].tail())

# --- Sector series
with open(data_dir / "sector_series.pkl", "rb") as f:
    sector_series = pickle.load(f)
print("Sector series:", len(sector_series))
first_sec = next(iter(sector_series))
print("Example sector:", first_sec)
print(sector_series[first_sec].head())

# --- National series
with open(data_dir / "national_series.pkl", "rb") as f:
    national_series = pickle.load(f)
print("National series length:", len(national_series))
print(national_series.head())

# --- Metadata
meta = pd.read_csv(data_dir / "facility_metadata.csv")
print(meta.head())