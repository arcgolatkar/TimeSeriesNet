"""
Evaluation metrics for time series forecasting.

Implements MAE, sMAPE, MASE, and other metrics.
"""

import torch
import numpy as np
from typing import Union, Optional


def mae(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return np.sqrt(mse(y_true, y_pred))


def smape(y_true: Union[np.ndarray, torch.Tensor], 
          y_pred: Union[np.ndarray, torch.Tensor],
          epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    sMAPE = 100% * mean(|y_true - y_pred| / (|y_true| + |y_pred|))
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        sMAPE score (0-100)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    
    return 100.0 * np.mean(numerator / denominator)


def mape(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor],
         epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE score (0-100)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return 100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))


def mase(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor],
         y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
         seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error.
    
    MASE = MAE(forecast) / MAE(naive_forecast)
    
    The naive forecast is the seasonal naive method (previous value at same season).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_train: Training data to compute scaling factor (if None, uses y_true)
        seasonality: Seasonal period (1 for non-seasonal data)
        
    Returns:
        MASE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # MAE of forecast
    forecast_mae = np.mean(np.abs(y_true - y_pred))
    
    # MAE of naive forecast (use training data if provided)
    if y_train is not None:
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    else:
        # Use in-sample naive forecast
        if len(y_true) <= seasonality:
            return forecast_mae  # Cannot compute MASE
        naive_mae = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    
    # Avoid division by zero
    if naive_mae < 1e-8:
        return float('inf') if forecast_mae > 1e-8 else 0.0
    
    return forecast_mae / naive_mae


def r2_score(y_true: Union[np.ndarray, torch.Tensor], 
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    R-squared (coefficient of determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R2 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-8:
        return 0.0
    
    return 1.0 - (ss_res / ss_tot)


def compute_all_metrics(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor],
                       y_train: Optional[Union[np.ndarray, torch.Tensor]] = None) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_train: Training data for MASE computation
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }
    
    # Add MASE if training data provided
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train)
    
    return metrics


def hierarchical_consistency_error(facility_preds: dict,
                                   sector_preds: dict,
                                   national_pred: float,
                                   facility_to_sector: dict) -> float:
    """
    Compute hierarchical consistency error.
    
    Checks if facility predictions aggregate to sector predictions,
    and sector predictions aggregate to national prediction.
    
    Args:
        facility_preds: {facility_id: prediction}
        sector_preds: {sector: prediction}
        national_pred: National level prediction
        facility_to_sector: {facility_id: sector}
        
    Returns:
        Mean absolute consistency error
    """
    errors = []
    
    # Check facility -> sector aggregation
    for sector, sector_pred in sector_preds.items():
        facilities_in_sector = [
            fid for fid, s in facility_to_sector.items() if s == sector
        ]
        facility_sum = sum(
            facility_preds.get(fid, 0.0) for fid in facilities_in_sector
        )
        errors.append(abs(facility_sum - sector_pred))
    
    # Check sector -> national aggregation
    sector_sum = sum(sector_preds.values())
    errors.append(abs(sector_sum - national_pred))
    
    return np.mean(errors) if errors else 0.0


class MetricsTracker:
    """Track metrics over multiple batches/epochs."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_dict: dict):
        """Update metrics with new values."""
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value
            self.counts[name] += 1
    
    def compute(self) -> dict:
        """Compute average metrics."""
        return {
            name: total / self.counts[name]
            for name, total in self.metrics.items()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([100, 110, 105, 115, 120])
    y_pred = np.array([98, 112, 103, 117, 118])
    
    print("Testing metrics:")
    print(f"MAE: {mae(y_true, y_pred):.2f}")
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"sMAPE: {smape(y_true, y_pred):.2f}%")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
    print(f"MASE: {mase(y_true, y_pred):.4f}")
    print(f"R2: {r2_score(y_true, y_pred):.4f}")
    
    print("\nAll metrics:")
    all_metrics = compute_all_metrics(y_true, y_pred)
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.4f}")

