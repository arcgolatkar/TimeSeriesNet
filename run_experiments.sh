#!/bin/bash

# Quick-start script to run all experiments
# Usage: bash run_experiments.sh

echo "=========================================="
echo "Multi-Scale Transformer for GHG Forecasting"
echo "Complete Experiment Pipeline"
echo "=========================================="

# Setup
echo ""
echo "Step 1: Setting up environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Data
echo ""
echo "Step 2: Downloading and preprocessing data..."
python src/utils/download_data.py

echo "Preprocessing data..."
python src/utils/preprocess_data.py

# Baselines
echo ""
echo "Step 3: Training baseline models..."

echo "  - Training ARIMA..."
python src/baselines/train_arima.py --config configs/arima_config.yaml

echo "  - Training Prophet..."
python src/baselines/train_prophet.py --config configs/prophet_config.yaml

echo "  - Training LSTM..."
python src/baselines/train_lstm.py --config configs/lstm_config.yaml

# Multi-Scale Transformer
echo ""
echo "Step 4: Training Multi-Scale Transformer..."
python src/models/train_mst.py --config configs/mst_config.yaml

# Evaluation
echo ""
echo "Step 5: Evaluating all models..."
python src/utils/evaluate_models.py

# Summary
echo ""
echo "=========================================="
echo "Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Metrics: results/metrics/"
echo "  - Figures: results/figures/"
echo "  - Checkpoints: results/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Review results in results/metrics/model_comparison.csv"
echo "  2. Check visualizations in results/figures/"
echo "  3. Open notebook: notebooks/01_data_preprocessing_and_experiments.ipynb"
echo "  4. Fill in interim report template: report/interim_report_template.md"
echo ""
echo "=========================================="

