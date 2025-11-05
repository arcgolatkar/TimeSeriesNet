# EPA Data Format Integration

## ✅ What's Been Updated

I've created new modules to work with your EPA data pipeline output format:

### 1. New Data Loading Module
**File:** `src/utils/load_epa_data.py`

**Key Features:**
- Loads `training_facility_year.csv` (main training data)
- Loads `sector_year_totals.csv` (sector/national aggregates)
- Loads `splits_years.json` (train/val/test year lists)
- Creates sample data if processed files don't exist

**Usage:**
```python
from utils.load_epa_data import EPADataLoader

loader = EPADataLoader(data_dir='out_epa')
training_data, sector_data, splits = loader.load_all()
```

### 2. New PyTorch Dataset
**File:** `src/utils/epa_dataset.py`

**Key Features:**
- Works with the engineered features from `training_facility_year.csv`
- Creates sliding windows (e.g., 8-year input → 1-year prediction)
- Includes hierarchical targets (facility, sector, national)
- Auto-detects feature columns (qty_*, co2e_*)

**Usage:**
```python
from utils.epa_dataset import create_dataloaders_from_epa

dataloaders = create_dataloaders_from_epa(
    training_data=training_data,
    sector_data=sector_data,
    splits=splits,
    sequence_length=8,
    forecast_horizon=1,
    batch_size=64
)
```

## 📊 Data Structure Expected

```
out_epa/
├─ raw/
│  ├─ pub_dim_facility.csv
│  ├─ pub_facts_sector_ghg_emission.csv
│  └─ pub_facts_subp_ghg_emission_YYYY.csv
├─ training_facility_year.csv          ← Main training file
├─ sector_year_totals.csv              ← Sector/national aggregates
├─ facility_year_wide.csv              ← Optional
├─ facility_year_gas_long.csv          ← Optional
└─ splits_years.json                   ← Train/val/test years
```

### Required Columns in `training_facility_year.csv`:
- `FACILITY_ID`: Facility identifier
- `REPORTING_YEAR`: Year
- `co2e_total`: Target variable (total CO2e emissions)
- `qty_<gas>`: Mass quantities per gas (e.g., qty_co2, qty_ch4)
- `co2e_<gas>`: CO2e per gas (e.g., co2e_co2, co2e_ch4)
- `co2e_total_lag1`, `co2e_total_lag3`: Lag features
- `roll3_mean_co2e`, `roll3_std_co2e`: Rolling stats
- Optional: `facility_name`, `state`, `sector`, `city`, `zip`, `latitude`, `longitude`

### Required Columns in `sector_year_totals.csv`:
- `REPORTING_YEAR`: Year
- `sector`: Sector name (or 'NATIONAL_TOTAL' for national)
- `co2e_total`: Total emissions for that sector/year

### Required Format for `splits_years.json`:
```json
{
  "train": [2010, 2011, ..., 2019],
  "val": [2020, 2021],
  "test": [2022, 2023]
}
```

## 🚀 Quick Start

### Option 1: Use Your Pre-Processed Data

If you already have the EPA pipeline output:

```bash
# Place your files in out_epa/
out_epa/
├─ training_facility_year.csv
├─ sector_year_totals.csv
└─ splits_years.json

# Test loading
python src/utils/load_epa_data.py
```

### Option 2: Create Sample Data for Testing

```bash
cd /Users/architgolatkar/arc/Fall2025/DL/TimeSeries
source venv/bin/activate
python src/utils/load_epa_data.py
```

This will create sample data in `out_epa/` with the correct format.

## 📝 Update Your Workflow

### Old Workflow:
```python
from utils.download_data import EPADataDownloader
from utils.preprocess_data import GHGDataPreprocessor

# Download and preprocess
downloader = EPADataDownloader()
data = downloader.create_sample_data()

preprocessor = GHGDataPreprocessor()
preprocessor.process_all()
```

### New Workflow:
```python
from utils.load_epa_data import EPADataLoader
from utils.epa_dataset import create_dataloaders_from_epa

# Load pre-processed data
loader = EPADataLoader(data_dir='out_epa')
training_data, sector_data, splits = loader.load_all()

# Create PyTorch dataloaders
dataloaders = create_dataloaders_from_epa(
    training_data, sector_data, splits,
    sequence_length=8,
    forecast_horizon=1,
    batch_size=64
)

# Ready for training!
for batch in dataloaders['train']:
    # batch['input_features']: [64, 8, n_features]
    # batch['facility_target']: [64, 1, 1]
    # batch['sector_target']: [64, 1, 1]
    # batch['national_target']: [64, 1, 1]
    pass
```

## 🔧 Updating Model Training Scripts

### Update MST Training

In `src/models/train_mst.py`, replace the data loading section:

```python
# OLD:
from utils.data_loader import create_dataloaders

dataloaders = create_dataloaders(...)

# NEW:
from utils.load_epa_data import EPADataLoader
from utils.epa_dataset import create_dataloaders_from_epa

loader = EPADataLoader(data_dir='out_epa')
training_data, sector_data, splits = loader.load_all()

dataloaders = create_dataloaders_from_epa(
    training_data, sector_data, splits,
    sequence_length=config['data']['sequence_length'],
    forecast_horizon=config['data']['forecast_horizon'],
    batch_size=config['data']['batch_size'],
    num_workers=config['data']['num_workers']
)
```

### Update Model Architecture

The MST model needs to handle the new feature dimensions:

```python
# In multi_scale_transformer.py
# Update input_projection to handle n_features instead of 1

class FineScaleEncoder(nn.Module):
    def __init__(self, input_size=6, d_model=256, ...):  # input_size = n_features
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)  # Changed from 1
        ...
```

## ✅ Benefits of New Format

1. **Pre-engineered Features**: Lag variables and rolling stats already computed
2. **Hierarchical Targets**: Facility, sector, and national predictions in one batch
3. **Consistent Splits**: Train/val/test defined in JSON (reproducible)
4. **Per-Gas Information**: Can extend to multi-task learning (predict each gas)
5. **Rich Metadata**: Facility attributes available for analysis
6. **Efficient**: No need to recompute aggregations during training

## 📊 Batch Structure

Each batch contains:

```python
{
    'facility_id': List of facility IDs
    'sector': List of sectors
    'input_features': torch.Tensor([batch, 8, n_features])  # Historical features
    'input_targets': torch.Tensor([batch, 8, 1])           # Historical targets  
    'facility_target': torch.Tensor([batch, 1, 1])         # Facility prediction
    'sector_target': torch.Tensor([batch, 1, 1])           # Sector aggregate
    'national_target': torch.Tensor([batch, 1, 1])         # National aggregate
    'input_years': torch.LongTensor([batch, 8])            # Input year labels
    'output_years': torch.LongTensor([batch, 1])           # Output year labels
}
```

## 🎯 Next Steps

1. **Place your EPA data** in `out_epa/` directory
2. **Test loading**: `python src/utils/load_epa_data.py`
3. **Test dataset**: `python src/utils/epa_dataset.py`
4. **Update training configs** to use new data paths
5. **Modify model** input layer for multi-feature input
6. **Train models** with the new data pipeline

## 💡 Feature Selection

You can control which features to use:

```python
# Use only CO2e features
feature_cols = ['co2e_co2', 'co2e_ch4', 'co2e_n2o']

# Or use all quantity + CO2e features
feature_cols = None  # Auto-detects all qty_* and co2e_* columns

dataloaders = create_dataloaders_from_epa(
    ...,
    feature_cols=feature_cols
)
```

## 📞 Support

The old modules (`download_data.py`, `preprocess_data.py`, `data_loader.py`) are still available for backward compatibility, but the new modules are recommended for working with your EPA pipeline output.

---

**Status**: ✅ Ready to integrate your EPA data pipeline output!

