# Multi-Scale Transformer for U.S. Greenhouse Gas Emission Forecasting

## Project Overview

This project develops a Multi-Scale Transformer (MST) to forecast U.S. greenhouse gas emissions using the EPA GHGRP dataset. The model learns both short-term variations and long-term trends across facility, sector, and national levels through hierarchical attention and cross-scale fusion.

## Team Members

- Shyam Solanke (UID: 121127761)
- Ninad Wadode (UID: 121317674)
- Saanika Patil (UID: 120414893)
- Archit Golatkar (UID: 121305282)
- Sriniketh Shankar (UID: 121113580)

**Course**: MSML/DATA 612 Deep Learning  
**Professor**: Samyet Ayhan

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TimeSeries
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
TimeSeries/
├── data/
│   ├── raw/              # Raw EPA GHGRP data
│   └── processed/        # Preprocessed datasets
├── src/
│   ├── models/           # Multi-Scale Transformer implementation
│   ├── baselines/        # ARIMA, Prophet, LSTM models
│   └── utils/            # Data loaders, metrics, visualization
├── notebooks/            # Exploratory analysis and experiments
├── configs/              # Configuration files
├── results/
│   ├── figures/          # Generated plots and visualizations
│   ├── metrics/          # Evaluation metrics
│   └── checkpoints/      # Model checkpoints
├── report/               # Interim report and presentation
├── requirements.txt      # Python dependencies
└── README.md
```

## Usage

### 1. Data Preparation

Download and preprocess the EPA GHGRP data:

```bash
python src/utils/download_data.py
python src/utils/preprocess_data.py
```

### 2. Training Baseline Models

Train classical and deep learning baselines:

```bash
# ARIMA/SARIMA
python src/baselines/train_arima.py --config configs/arima_config.yaml

# Prophet
python src/baselines/train_prophet.py --config configs/prophet_config.yaml

# LSTM
python src/baselines/train_lstm.py --config configs/lstm_config.yaml
```

### 3. Training Multi-Scale Transformer

Train the Multi-Scale Transformer:

```bash
python src/models/train_mst.py --config configs/mst_config.yaml
```

### 4. Evaluation

Evaluate all models and generate comparison results:

```bash
python src/utils/evaluate_models.py --output results/metrics/
```

### 5. Generate Visualizations

Create plots and figures for the report:

```bash
python src/utils/generate_plots.py --output results/figures/
```

## Model Architecture

The Multi-Scale Transformer consists of four key components:

1. **Fine-Scale Encoder**: Captures short-term, year-to-year emission changes using local attention
2. **Coarse-Scale Encoder**: Models long-term sectoral trends via downsampled sequences
3. **Cross-Scale Fusion Layer**: Combines information between fine and coarse representations
4. **Hierarchical Decoder**: Ensures aggregation consistency across facility, sector, and national levels

## Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **sMAPE** (Symmetric Mean Absolute Percentage Error): Scale-independent percentage error
- **MASE** (Mean Absolute Scaled Error): Scaled error metric relative to naive baseline

## Data Source

U.S. EPA Greenhouse Gas Reporting Program (GHGRP) - Emissions by Location  
https://www.epa.gov/ghgreporting/ghgrp-emissions-location

## Reproducibility

All random seeds are fixed for reproducibility:
- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`

## References

1. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
2. Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Forecasting. AAAI.
3. Li, S. et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS.

## License

This project is for academic purposes as part of MSML/DATA 612 Deep Learning course.
