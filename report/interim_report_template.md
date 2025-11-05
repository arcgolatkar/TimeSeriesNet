# Multi-Scale Transformer for U.S. Greenhouse Gas Emission Forecasting
## Interim Report

**Course:** MSML/DATA 612 Deep Learning  
**Professor:** Samyet Ayhan  
**Date:** [Insert Date]

**Team Members:**
- Shyam Solanke (UID: 121127761)
- Ninad Wadode (UID: 121317674)
- Saanika Patil (UID: 120414893)
- Archit Golatkar (UID: 121305282)
- Sriniketh Shankar (UID: 121113580)

---

## Abstract

This project develops a Multi-Scale Transformer (MST) to forecast U.S. greenhouse gas emissions using the EPA GHGRP dataset. The model learns both short-term variations and long-term trends across facility, sector, and national levels through hierarchical attention and cross-scale fusion. Our interim results show [INSERT RESULTS] compared to baseline models including ARIMA, Prophet, and LSTM.

---

## 1. Introduction

### 1.1 Problem Statement and Motivation

Accurate greenhouse gas (GHG) emission forecasting is critical for:
- Environmental policy planning and compliance monitoring
- Emission trading systems and carbon markets
- Progress evaluation toward reduction targets
- Industrial production planning and mitigation strategies

Traditional forecasting methods (ARIMA, regression) fail to capture the multi-scale dependencies inherent in emission patterns, which vary:
- **Short-term**: Facility-level fluctuations due to operational changes
- **Long-term**: Sector-wide trends driven by technological shifts and regulations
- **Hierarchical**: Dependencies between facility, sector, and national levels

### 1.2 Objectives

Our project aims to:
1. Develop a Multi-Scale Transformer architecture that jointly models fine-grained and coarse-grained emission patterns
2. Ensure hierarchical consistency where facility predictions aggregate to sector and national totals
3. Outperform classical (ARIMA, Prophet) and deep learning (LSTM) baselines
4. Provide interpretable attention patterns for understanding emission dynamics

---

## 2. Related Work and Literature Review

### 2.1 Transformer Architectures

**Attention Is All You Need** (Vaswani et al., 2017) [1]:
- Introduced the Transformer architecture with self-attention mechanisms
- Enables parallel processing and long-range dependencies
- Foundation for our multi-scale approach

### 2.2 Time Series Transformers

**Informer** (Zhou et al., 2021) [2]:
- Efficient Transformer for long sequence forecasting
- ProbSparse self-attention reduces complexity from O(L²) to O(L log L)
- Inspired our encoder design choices

**Autoformer** (Li et al., 2021) [3]:
- Decomposition architecture with auto-correlation mechanism
- Captures seasonal patterns and trends separately
- Influenced our multi-scale decomposition strategy

### 2.3 Hierarchical Forecasting

**Hierarchical Time Series Forecasting** (Wickramasuriya et al., 2019) [4]:
- Methods for ensuring forecast consistency across levels
- Reconciliation techniques (bottom-up, top-down, optimal)
- Basis for our hierarchical consistency loss

---

## 3. Dataset and Preprocessing

### 3.1 Data Source

**EPA Greenhouse Gas Reporting Program (GHGRP)**:
- Source: https://www.epa.gov/ghgreporting/ghgrp-emissions-location
- Facility-level emissions from large emitters (≥25,000 metric tons CO₂e/year)
- Annual data from 2010 to 2023 (14 years)
- Coverage: [INSERT NUMBER] facilities across [INSERT NUMBER] industry sectors

### 3.2 Data Statistics

**Coverage:**
| Level | Count | Description |
|-------|-------|-------------|
| Facilities | [INSERT] | Individual emission sources |
| Sectors | [INSERT] | Industry categories (Power Plants, Petroleum, Chemicals, etc.) |
| Years | 14 | 2010-2023 temporal coverage |
| Total Records | [INSERT] | Facility-year combinations |

**Emissions by Gas Type:**
- CO₂ (Carbon Dioxide): [INSERT]%
- CH₄ (Methane): [INSERT]%  
- N₂O (Nitrous Oxide): [INSERT]%
- Others (HFCs, PFCs, SF₆): [INSERT]%

### 3.3 Preprocessing Pipeline

**Step 1: Data Cleaning**
- Remove duplicates and invalid entries
- Handle missing values using [INSERT METHOD]
- Filter outliers beyond [INSERT] standard deviations

**Step 2: GHG to CO₂e Conversion**
- Convert all gases to CO₂ equivalent using Global Warming Potential (GWP):
  - CO₂: GWP = 1
  - CH₄: GWP = 25
  - N₂O: GWP = 298

**Step 3: Hierarchical Aggregation**
- **Facility Level**: Sum all GHGs per facility per year
- **Sector Level**: Aggregate facilities by industry type
- **National Level**: Sum across all sectors

**Step 4: Normalization**
- Apply log transformation: log(1 + emissions)
- Reduces skewness and stabilizes variance
- Facilitates neural network training

**Step 5: Train/Val/Test Split**
- Training: 2010-2019 (10 years)
- Validation: 2020-2021 (2 years)
- Test: 2022-2023 (2 years)
- Temporal split ensures no data leakage

---

## 4. Model Architecture and Implementation

### 4.1 Multi-Scale Transformer Overview

Our architecture consists of four key components:

```
Input → [Fine-Scale Encoder] → ┐
                               ├→ [Cross-Scale Fusion] → [Hierarchical Decoder] → Predictions
Input → [Coarse-Scale Encoder] → ┘
```

### 4.2 Fine-Scale Encoder

**Purpose:** Capture short-term, year-to-year emission changes

**Architecture:**
- Input projection: 1D → d_model (256 dimensions)
- Positional encoding: Sinusoidal embeddings
- Transformer encoder: 4 layers
- Multi-head attention: 8 heads
- Feed-forward dimension: 1024
- Dropout: 0.1

**Key Features:**
- Full sequence attention for capturing local dependencies
- Models annual fluctuations due to operational changes

### 4.3 Coarse-Scale Encoder

**Purpose:** Capture long-term sectoral trends

**Architecture:**
- Temporal downsampling: Average pooling (factor = 3)
- Input projection: 1D → d_model (256 dimensions)
- Positional encoding: Sinusoidal embeddings
- Transformer encoder: 4 layers (same as fine-scale)
- Multi-head attention: 8 heads

**Key Features:**
- Downsampling reduces sequence length (e.g., 12 years → 4 points)
- Focuses on structural trends and long-term patterns

### 4.4 Cross-Scale Fusion Layer

**Purpose:** Integrate information across temporal scales

**Mechanism:**
- Bidirectional cross-attention:
  - Fine features query coarse features
  - Coarse features query fine features
- Residual connections preserve original information
- Layer normalization for stability

**Formula:**
```
fine_fused = fine + CrossAttn(fine, coarse, coarse)
coarse_fused = coarse + CrossAttn(coarse, fine, fine)
```

### 4.5 Hierarchical Decoder

**Purpose:** Generate predictions at facility, sector, and national levels

**Architecture:**
- Transformer decoder: 3 layers
- Learnable query embeddings for forecast horizon
- Separate prediction heads for each hierarchical level:
  - Facility head: MLP(d_model → d_model/2 → forecast_horizon)
  - Sector head: MLP(d_model → d_model/2 → forecast_horizon)
  - National head: MLP(d_model → d_model/2 → forecast_horizon)

**Hierarchical Consistency:**
- Soft constraint via loss function (see Section 4.7)

### 4.6 Model Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 256 | Balance between capacity and efficiency |
| num_heads | 8 | Standard choice for 256-dim embeddings |
| num_encoder_layers | 4 | Sufficient for temporal patterns |
| num_decoder_layers | 3 | Decoder complexity |
| dim_feedforward | 1024 | 4x hidden dimension (standard) |
| dropout | 0.1 | Regularization |
| downsample_factor | 3 | Coarse scale = ~3-year aggregates |
| sequence_length | 10 | 10 years of history |
| forecast_horizon | 1 | Predict next year |

**Total Parameters:** [INSERT NUMBER]

### 4.7 Loss Function

Multi-scale loss combining three components:

```
L_total = α₁·L_fine + α₂·L_coarse + α₃·L_consistency

where:
  L_fine = MSE(y_facility, ŷ_facility)
  L_coarse = [MSE(y_sector, ŷ_sector) + MSE(y_national, ŷ_national)] / 2
  L_consistency = |mean(ŷ_facility) - mean(ŷ_sector)| + |mean(ŷ_sector) - ŷ_national|
```

**Loss Weights:**
- α₁ = 1.0 (fine-level emphasis)
- α₂ = 0.5 (coarse-level regularization)
- α₃ = 0.3 (consistency constraint)

### 4.8 Training Details

**Optimizer:** AdamW
- Learning rate: 1e-4
- Weight decay: 0.01 (L2 regularization)
- β₁ = 0.9, β₂ = 0.999

**Learning Rate Schedule:** Cosine annealing
- T_max = 200 epochs
- η_min = 1e-6

**Other Training Settings:**
- Batch size: 64
- Max epochs: 200
- Early stopping: Patience = 20 epochs
- Gradient clipping: max_norm = 1.0
- Random seed: 42 (reproducibility)

**Hardware:**
- [INSERT: e.g., NVIDIA RTX 3090, 24GB VRAM]
- Training time: [INSERT] hours

### 4.9 Implementation Framework

- **Language:** Python 3.9+
- **Deep Learning:** PyTorch 2.0, PyTorch Lightning 2.0
- **Data Processing:** Pandas, NumPy
- **Classical Models:** Statsmodels (ARIMA), Prophet
- **Experiment Tracking:** TensorBoard / MLflow
- **Version Control:** Git

---

## 5. Baseline Models

### 5.1 ARIMA/SARIMA

**Implementation:**
- Auto ARIMA with pmdarima library
- Searches (p,d,q) space: p,q ∈ [0,5], d ∈ [0,2]
- Seasonal order: (P,D,Q,m) with m=1 (annual data)
- Rolling window forecasting for evaluation

**Configuration:**
- Automatic parameter selection
- AIC (Akaike Information Criterion) for model selection

### 5.2 Prophet

**Implementation:**
- Facebook Prophet with default settings
- Linear growth model
- No sub-annual seasonality (yearly data)
- Rolling window forecasting

**Configuration:**
- changepoint_prior_scale: 0.05
- seasonality_prior_scale: 10.0
- Additive seasonality mode

### 5.3 LSTM

**Architecture:**
- 3-layer LSTM
- Hidden size: 128
- Dropout: 0.2
- Unidirectional

**Training:**
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Max epochs: 150

---

## 6. Experiments and Results

### 6.1 Evaluation Metrics

We use multiple metrics to assess forecast accuracy:

**MAE (Mean Absolute Error):**
```
MAE = (1/n) Σ |y_true - y_pred|
```
- Measures average absolute deviation
- Interpretable in original units

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt((1/n) Σ (y_true - y_pred)²)
```
- Penalizes large errors more heavily

**sMAPE (Symmetric Mean Absolute Percentage Error):**
```
sMAPE = 100% · (1/n) Σ |y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
```
- Scale-independent percentage metric
- Range: 0-100%

**MASE (Mean Absolute Scaled Error):**
```
MASE = MAE(forecast) / MAE(naive_forecast)
```
- Compares to naive (previous value) baseline
- MASE < 1 means better than naive

### 6.2 Model Comparison Results

**Overall Performance on Test Set (2022-2023):**

| Model | MAE ↓ | RMSE ↓ | sMAPE ↓ | MASE ↓ | Training Time |
|-------|-------|--------|---------|---------|---------------|
| ARIMA | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Prophet | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| LSTM | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| **MST (Ours)** | **[INSERT]** | **[INSERT]** | **[INSERT]** | **[INSERT]** | [INSERT] |

**Key Findings:**
- [INSERT ANALYSIS: e.g., "MST achieves X% lower MAE than best baseline"]
- [INSERT: Discuss which baselines performed well/poorly and why]

### 6.3 Hierarchical Level Performance

**MST Performance by Hierarchical Level:**

| Level | MAE | RMSE | sMAPE |
|-------|-----|------|-------|
| Facility | [INSERT] | [INSERT] | [INSERT] |
| Sector | [INSERT] | [INSERT] | [INSERT] |
| National | [INSERT] | [INSERT] | [INSERT] |

**Analysis:**
- [INSERT: Discuss performance across levels]
- [INSERT: Comment on hierarchical consistency]

### 6.4 Ablation Study

**Impact of Model Components:**

| Configuration | MAE | Change |
|--------------|-----|--------|
| Full MST | [INSERT] | Baseline |
| w/o Cross-Scale Fusion | [INSERT] | +[INSERT]% |
| w/o Coarse Encoder | [INSERT] | +[INSERT]% |
| w/o Hierarchical Consistency Loss | [INSERT] | +[INSERT]% |

**Findings:**
- [INSERT: Which components are most critical?]

### 6.5 Visualization of Results

**Figure 1: National-Level Predictions**
[INSERT FIGURE: Time series plot showing actual vs predicted national emissions]

**Figure 2: Sector-Level Predictions**
[INSERT FIGURE: Multiple time series for different sectors]

**Figure 3: Model Comparison**
[INSERT FIGURE: Bar chart comparing metrics across models]

**Figure 4: Error Distribution**
[INSERT FIGURE: Histogram of prediction errors for MST]

**Figure 5: Training Curves**
[INSERT FIGURE: Training and validation loss over epochs]

---

## 7. Discussion

### 7.1 Key Findings

[INSERT: Summarize 3-5 main findings from experiments]

1. **Multi-scale learning improves accuracy:** [EXPLAIN]
2. **Hierarchical consistency matters:** [EXPLAIN]
3. **Cross-scale fusion captures complex patterns:** [EXPLAIN]

### 7.2 Model Strengths

- **Multi-temporal modeling:** Captures both short and long-term patterns
- **Hierarchical consistency:** Predictions aggregate correctly across levels
- **Attention mechanism:** Provides interpretability via attention weights
- **Scalability:** Handles hundreds of facilities and multiple sectors

### 7.3 Limitations and Challenges

**Data Limitations:**
- Limited temporal data (14 years)
- Annual granularity (no monthly/quarterly patterns)
- Missing data for some facilities

**Model Limitations:**
- Computational cost higher than classical baselines
- Requires sufficient training data
- [INSERT: Any other challenges encountered]

**Implementation Challenges:**
- [INSERT: E.g., tuning hierarchical loss weights]
- [INSERT: E.g., handling variable-length sequences]

### 7.4 Lessons Learned

[INSERT: What worked well? What didn't? What would you do differently?]

---

## 8. Future Work and Next Steps

### 8.1 For Final Report

**Model Improvements:**
- [ ] Incorporate external features (economic indicators, policy changes)
- [ ] Multi-horizon forecasting (predict 2-5 years ahead)
- [ ] Uncertainty quantification (prediction intervals)

**Additional Experiments:**
- [ ] Spatial generalization (train on certain regions, test on others)
- [ ] Few-shot learning for new facilities
- [ ] Attention visualization and interpretation

**Analysis:**
- [ ] Feature importance analysis
- [ ] Error analysis by sector and facility characteristics
- [ ] Comparison with ensemble methods

### 8.2 Extensions

**Technical Extensions:**
- Incorporate graph neural networks for facility relationships
- Add exogenous variables (temperature, production data, policy indicators)
- Implement probabilistic forecasting with quantile regression

**Application Extensions:**
- Real-time forecasting dashboard
- What-if scenario analysis for policy makers
- Integration with emission trading systems

---

## 9. Code Quality and Reproducibility

### 9.1 Repository Structure

```
TimeSeries/
├── data/
│   ├── raw/              # Original EPA data
│   └── processed/        # Preprocessed datasets
├── src/
│   ├── models/           # MST implementation
│   ├── baselines/        # Baseline models
│   └── utils/            # Data loaders, metrics, visualization
├── notebooks/            # Jupyter notebooks
├── configs/              # YAML configuration files
├── results/              # Figures and metrics
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

### 9.2 Reproducibility

**All experiments are reproducible:**
- Fixed random seeds (seed=42)
- Version-controlled configurations (YAML files)
- Documented dependencies (requirements.txt)
- Step-by-step instructions in README

**To reproduce results:**
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download and preprocess data
python src/utils/download_data.py
python src/utils/preprocess_data.py

# 3. Train models
python src/baselines/train_arima.py --config configs/arima_config.yaml
python src/baselines/train_prophet.py --config configs/prophet_config.yaml
python src/baselines/train_lstm.py --config configs/lstm_config.yaml
python src/models/train_mst.py --config configs/mst_config.yaml

# 4. Evaluate and compare
python src/utils/evaluate_models.py
```

### 9.3 Code Quality

**Best Practices Followed:**
- Type hints for function signatures
- Comprehensive docstrings (Google style)
- PEP8 compliant (checked with flake8)
- Modular design with clear separation of concerns
- Unit tests for critical functions (in progress)

---

## 10. References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

[2] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

[3] Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y. X., & Yan, X. (2021). Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. *Advances in Neural Information Processing Systems*, 34, 5243-5254.

[4] Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804-819.

[5] U.S. Environmental Protection Agency. (2024). Greenhouse Gas Reporting Program (GHGRP). Retrieved from https://www.epa.gov/ghgreporting

[6] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

[7] Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.

[8] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control*. John Wiley & Sons.

---

## Appendix A: Additional Figures

[INSERT: Any additional plots, attention visualizations, etc.]

## Appendix B: Hyperparameter Tuning

[INSERT: If applicable, show results of hyperparameter search]

## Appendix C: Detailed Results Tables

[INSERT: Per-sector results, per-facility results for select facilities, etc.]

---

**Word Count:** [INSERT]  
**Figure Count:** [INSERT]  
**Table Count:** [INSERT]

---

*This report represents interim progress on the Multi-Scale Transformer for GHG Emission Forecasting project for MSML/DATA 612 Deep Learning course.*

