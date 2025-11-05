# Updated Workflow for EPA Data Format

## 🎯 Summary of Changes

I've updated the codebase to work with your EPA data pipeline format:

### ✅ New Files Created

1. **`src/utils/load_epa_data.py`** - EPA data loader
   - Loads `training_facility_year.csv`, `sector_year_totals.csv`, `splits_years.json`
   - Creates sample data if needed

2. **`src/utils/epa_dataset.py`** - PyTorch dataset for EPA data
   - Handles multi-feature input (qty_*, co2e_*)
   - Creates sliding windows (e.g., 8 years → 1 year)
   - Includes hierarchical targets (facility, sector, national)

3. **`EPA_DATA_INTEGRATION.md`** - Complete integration guide

4. **`test_epa_data.py`** - Test script to verify setup

### 📁 Expected Data Structure

```
out_epa/
├─ training_facility_year.csv    ← Main training data with features
├─ sector_year_totals.csv        ← Sector/national aggregates
└─ splits_years.json             ← Train/val/test year splits
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/architgolatkar/arc/Fall2025/DL/TimeSeries
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test with Sample Data

```bash
# This creates sample EPA data in out_epa/
python test_epa_data.py
```

Expected output:
```
EPA Data Integration Test
==================================================================
1. Loading EPA data...
   → Creating sample data (real data not found)
   ✓ Training data: 2,800 rows
   ✓ Facilities: 200
   ✓ Years: 2010 - 2023

2. Creating PyTorch dataloaders...
   ✓ Train: 43 batches
   ✓ Val:   8 batches
   ✓ Test:  8 batches

3. Testing batch structure...
   ✓ Input shape: [batch=32, seq_len=8, features=6]
   ✓ Output shape: [batch=32, horizon=1]

✅ All tests passed!
```

### 3. Add Your Real EPA Data

Once you have your pre-processed EPA data:

```bash
# Copy your files to out_epa/
cp /path/to/your/training_facility_year.csv out_epa/
cp /path/to/your/sector_year_totals.csv out_epa/
cp /path/to/your/splits_years.json out_epa/

# Test loading
python test_epa_data.py
```

## 📝 Code Examples

### Quick Start

```python
from utils.load_epa_data import EPADataLoader
from utils.epa_dataset import create_dataloaders_from_epa

# Load data
loader = EPADataLoader(data_dir='out_epa')
training_data, sector_data, splits = loader.load_all()

# Create dataloaders
dataloaders = create_dataloaders_from_epa(
    training_data=training_data,
    sector_data=sector_data,
    splits=splits,
    sequence_length=8,
    forecast_horizon=1,
    batch_size=64,
    num_workers=4
)

# Use in training
for batch in dataloaders['train']:
    # batch['input_features']: [64, 8, 6]  # 6 features: co2e_co2, co2e_ch4, etc.
    # batch['facility_target']: [64, 1, 1]
    # batch['sector_target']: [64, 1, 1]
    # batch['national_target']: [64, 1, 1]
    pass
```

### Batch Structure

Each batch contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_features` | [B, L, F] | Historical features (L=8 years, F≈6-12 features) |
| `input_targets` | [B, L, 1] | Historical targets |
| `facility_target` | [B, H, 1] | Facility-level target (H=1 forecast horizon) |
| `sector_target` | [B, H, 1] | Sector-level aggregate |
| `national_target` | [B, H, 1] | National-level aggregate |
| `input_years` | [B, L] | Year labels for input |
| `output_years` | [B, H] | Year labels for output |
| `facility_id` | List | Facility IDs |
| `sector` | List | Sector names |

where B=batch_size, L=sequence_length, H=forecast_horizon, F=n_features

## 🔧 Model Updates Needed

### 1. Update Input Layer

The MST model currently expects 1 feature per timestep. Update it for multi-feature input:

```python
# In src/models/multi_scale_transformer.py

class FineScaleEncoder(nn.Module):
    def __init__(self, 
                 input_size=6,  # Changed from 1
                 d_model=256, 
                 ...):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)  # Changed from 1
        # rest stays the same
```

### 2. Update Training Script

In `src/models/train_mst.py`:

```python
# Replace data loading section
from utils.load_epa_data import EPADataLoader
from utils.epa_dataset import create_dataloaders_from_epa

# Load EPA data
loader = EPADataLoader(data_dir='out_epa')
training_data, sector_data, splits = loader.load_all()

# Get feature columns
feature_cols = [c for c in training_data.columns 
                if c.startswith(('co2e_', 'qty_')) and c != 'co2e_total']
n_features = len(feature_cols)

print(f"Using {n_features} features: {feature_cols}")

# Create dataloaders
dataloaders = create_dataloaders_from_epa(
    training_data, sector_data, splits,
    sequence_length=config['data']['sequence_length'],
    forecast_horizon=config['data']['forecast_horizon'],
    batch_size=config['data']['batch_size'],
    num_workers=config['data']['num_workers'],
    feature_cols=feature_cols
)

# Create model with correct input size
model = MSTLightningModule(
    input_size=n_features,  # Add this parameter
    d_model=config['model']['fine_encoder']['d_model'],
    ...
)
```

### 3. Update Config

In `configs/mst_config.yaml`:

```yaml
model:
  name: "MultiScaleTransformer"
  input_size: 6  # Add this: number of input features
  
  fine_encoder:
    d_model: 256
    ...
```

## 📊 Data Features

### Available Features in `training_facility_year.csv`:

**Target:**
- `co2e_total` - Total CO2e emissions (primary target)

**Per-Gas Features:**
- `qty_co2`, `qty_ch4`, `qty_n2o` - Mass quantities
- `co2e_co2`, `co2e_ch4`, `co2e_n2o` - CO2e equivalents

**Engineered Features:**
- `co2e_total_lag1`, `co2e_total_lag3`, `co2e_total_lag5` - Lag features
- `roll3_mean_co2e`, `roll3_std_co2e` - Rolling statistics

**Metadata:**
- `FACILITY_ID`, `REPORTING_YEAR`
- `facility_name`, `state`, `sector`, `city`, `zip`
- `latitude`, `longitude`

### Feature Selection

You can control which features to use:

```python
# Option 1: Only CO2e features (default)
feature_cols = ['co2e_co2', 'co2e_ch4', 'co2e_n2o']

# Option 2: CO2e + lags
feature_cols = ['co2e_co2', 'co2e_ch4', 'co2e_n2o', 
                'co2e_total_lag1', 'roll3_mean_co2e']

# Option 3: All engineered features
feature_cols = None  # Auto-detects all qty_* and co2e_* columns

dataloaders = create_dataloaders_from_epa(
    ...,
    feature_cols=feature_cols
)
```

## ✅ Migration Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test sample data: `python test_epa_data.py`
- [ ] Place real EPA data in `out_epa/` directory
- [ ] Update model input layer for multi-feature support
- [ ] Update training scripts to use new data loaders
- [ ] Update configs with `input_size` parameter
- [ ] Test training with small epoch to verify setup
- [ ] Run full training

## 🎯 Benefits

✅ **Pre-engineered features** - Lags, rolling stats computed  
✅ **Hierarchical targets** - Facility, sector, national in one batch  
✅ **Consistent splits** - Defined in JSON (reproducible)  
✅ **Per-gas breakdown** - Can extend to multi-task learning  
✅ **Efficient** - No recomputing aggregations during training  

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/utils/load_epa_data.py` | Load EPA pre-processed files |
| `src/utils/epa_dataset.py` | PyTorch dataset creation |
| `test_epa_data.py` | Test script |
| `EPA_DATA_INTEGRATION.md` | Detailed integration guide |
| `UPDATED_WORKFLOW.md` | This file |

## 🆘 Troubleshooting

**Error: "Processed data files not found"**
- Run `python test_epa_data.py` to create sample data
- Or place your real EPA files in `out_epa/`

**Error: "ModuleNotFoundError: No module named 'torch'"**
- Install dependencies: `pip install -r requirements.txt`

**Error: "Input shape mismatch"**
- Update model `input_size` parameter to match number of features
- Check `n_features` in dataloaders creation

**Low performance**
- Experiment with different feature combinations
- Try including lag features and rolling stats
- Adjust sequence length (try 10 or 12 years)

## 🎓 Next Steps

1. **Test with sample data** to verify everything works
2. **Integrate your real EPA data** into `out_epa/`
3. **Update model architecture** for multi-feature input
4. **Run initial training** to get baseline results
5. **Iterate on features** to improve performance
6. **Complete interim report** with results

---

**Status**: ✅ Code updated and ready for EPA data format!

**Need help?** Check `EPA_DATA_INTEGRATION.md` for detailed examples.

