"""
ARIMA baseline using statsmodels directly (no pmdarima dependency).

This version doesn't use auto_arima but provides a simpler, more reliable ARIMA implementation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict, Optional
import warnings
import sys
sys.path.append('..')
from utils.metrics import compute_all_metrics

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """Simple ARIMA forecaster using statsmodels."""
    
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize ARIMA forecaster with fixed parameters.
        
        Args:
            order: (p, d, q) for ARIMA
            seasonal_order: (P, D, Q, m) for SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, y: pd.Series, verbose: bool = False) -> 'ARIMAForecaster':
        """Fit ARIMA model."""
        try:
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=verbose, maxiter=100)
            return self
        except Exception as e:
            if verbose:
                print(f"Error fitting ARIMA: {e}")
            # Fallback to simpler model
            self.order = (1, 1, 0)
            self.model = SARIMAX(y, order=self.order, seasonal_order=(0, 0, 0, 0))
            self.fitted_model = self.model.fit(disp=False, maxiter=50)
            return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Forecast future values."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted")
        return self.fitted_model.forecast(steps=steps).values
    
    def forecast_rolling(self,
                        train: pd.Series,
                        test: pd.Series,
                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Perform rolling window forecasting."""
        predictions = []
        actuals = []
        history = list(train.values)
        
        for i, actual in enumerate(test.values):
            if verbose and i % 5 == 0:
                print(f"  Step {i+1}/{len(test)}")
            
            try:
                # Fit model on history
                model = SARIMAX(
                    history,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(disp=False, maxiter=100)
                forecast = fitted.forecast(steps=1)[0]
                predictions.append(forecast)
            except:
                # Fallback to naive forecast
                predictions.append(history[-1])
            
            actuals.append(actual)
            history.append(actual)
        
        return np.array(predictions), np.array(actuals)


def evaluate_arima_on_series(series: pd.Series,
                             train_years: Tuple[int, int],
                             test_years: Tuple[int, int],
                             order: Tuple[int, int, int] = (1, 1, 1),
                             seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Optional[Dict]:
    """Evaluate ARIMA on a single time series."""
    train = series[(series.index >= train_years[0]) & (series.index <= train_years[1])]
    test = series[(series.index >= test_years[0]) & (series.index <= test_years[1])]
    
    if len(train) < 3 or len(test) < 1:
        return None
    
    forecaster = ARIMAForecaster(order=order, seasonal_order=seasonal_order)
    
    try:
        predictions, actuals = forecaster.forecast_rolling(train, test, verbose=False)
        metrics = compute_all_metrics(actuals, predictions, y_train=train.values)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics,
            'train_size': len(train),
            'test_size': len(test)
        }
    except Exception as e:
        return None


def evaluate_arima_on_multiple_series(series_dict: Dict[str, pd.Series],
                                      train_years: Tuple[int, int],
                                      test_years: Tuple[int, int],
                                      max_series: Optional[int] = None,
                                      order: Tuple[int, int, int] = (1, 1, 1),
                                      seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Dict:
    """Evaluate ARIMA on multiple time series."""
    results = {}
    all_predictions = []
    all_actuals = []
    
    series_list = list(series_dict.items())
    if max_series:
        series_list = series_list[:max_series]
    
    print(f"Evaluating ARIMA on {len(series_list)} time series...")
    print(f"Using ARIMA{order}")
    
    for i, (series_id, series) in enumerate(series_list):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(series_list)}")
        
        result = evaluate_arima_on_series(
            series, train_years, test_years, order, seasonal_order
        )
        
        if result:
            results[series_id] = result
            all_predictions.extend(result['predictions'])
            all_actuals.extend(result['actuals'])
    
    if all_predictions and all_actuals:
        overall_metrics = compute_all_metrics(
            np.array(all_actuals),
            np.array(all_predictions)
        )
    else:
        overall_metrics = {}
    
    print(f"\nEvaluated {len(results)} series successfully")
    print("\nOverall Metrics:")
    for name, value in overall_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return {
        'series_results': results,
        'overall_metrics': overall_metrics,
        'num_series': len(results)
    }

