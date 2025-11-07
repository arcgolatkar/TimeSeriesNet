# Notebooks Directory

This directory contains Jupyter notebooks for data analysis and experimentation.

## Available Notebooks

### 01_data_preprocessing_and_experiments.ipynb
Initial data preprocessing and experimental analysis.

### 02_epa_data_eda.ipynb
**Comprehensive Exploratory Data Analysis (EDA) for EPA GHG Emissions Dataset**

This notebook provides detailed analysis for the interim report, including:
- Dataset overview and statistics
- Temporal trend analysis
- Geographic distribution analysis
- Gas type composition analysis
- Facility-level patterns
- Missing data assessment
- Feature correlation analysis

## Running the EDA Notebook

### Prerequisites

Ensure you have the required packages installed:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Launch Jupyter

From the project root directory:

```bash
jupyter notebook notebooks/02_epa_data_eda.ipynb
```

Or from within VS Code/Cursor:
1. Open `02_epa_data_eda.ipynb`
2. Select Python kernel
3. Run all cells

### Expected Outputs

The notebook will generate:
- Statistical summaries printed to console
- Visualizations displayed inline
- Figure files saved to `results/figures/`:
  - `temporal_analysis.png`
  - `spatial_analysis.png`
  - `geographic_distribution.png`
  - `gas_type_analysis.png`
  - `facility_analysis.png`
  - `correlation_matrix.png`
  - `emissions_distribution.png`

### Runtime

- **Data loading**: ~30-60 seconds
- **Analysis and visualization**: ~2-5 minutes
- **Total runtime**: ~5-10 minutes (depending on system)

## Data Requirements

The notebook expects data in the following structure:

```
TimeSeries/
├── out_epa/
│   ├── training_facility_year.csv
│   ├── facility_year_wide.csv
│   ├── sector_year_totals.csv
│   └── splits_years.json
└── results/
    └── figures/  (will be created if doesn't exist)
```

## Troubleshooting

### Issue: "File not found"
**Solution**: Ensure you're running from the `notebooks/` directory or that relative paths are correct.

### Issue: "Module not found"
**Solution**: Install missing packages with `pip install <package_name>`

### Issue: "Memory error"
**Solution**: The dataset is large (~843K rows). Ensure you have at least 4GB RAM available. Consider loading subsets for initial exploration.

### Issue: "seaborn style warning"
**Solution**: If you get style warnings, replace `plt.style.use('seaborn-v0_8-darkgrid')` with `plt.style.use('seaborn-darkgrid')` or simply remove the line.

## Using Results for Report

After running the notebook:

1. **Figures**: All visualizations are saved in `results/figures/`
2. **Statistics**: Copy relevant statistics from notebook output
3. **Summary**: Refer to `report/eda_report_summary.md` for formatted content
4. **Key Findings**: Located in the final markdown cell of the notebook

## Customization

You can customize the analysis by modifying:
- **Year filter**: Change `df_train[df_train['year'] == 2023]` to any year
- **Top N**: Modify `top_states.head(15)` to show more/fewer items
- **Visualization style**: Adjust colors, sizes, and layout parameters
- **Statistical thresholds**: Change filtering criteria for outliers

## Notes

- The notebook uses the pre-processed training data with engineered features
- All visualizations use appropriate scales (log for skewed distributions)
- Missing data is handled through mask features
- Correlation analysis focuses on temporal features

## Contact

For questions or issues with the notebooks, please refer to the main project documentation.

