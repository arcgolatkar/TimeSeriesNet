"""
ARIMA/SARIMA baseline model for time series forecasting.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from typing import Tuple, Optional, Dict
import warnings
import sys
sys.path.append('..')
from utils.metrics import compute_all_metrics

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """ARIMA/SARIMA forecaster for time series."""
    
    def __init__(self,
                 order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 1),
                 auto_arima_search: bool = True,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5,
                 max_P: int = 2,
                 max_D: int = 1,
                 max_Q: int = 2,
                 m: int = 1,
                 seasonal: bool = False):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: (p, d, q) order for ARIMA
            seasonal_order: (P, D, Q, m) order for SARIMA
            auto_arima_search: Whether to use auto ARIMA to find best parameters
            max_p, max_d, max_q: Max values for auto ARIMA search
            max_P, max_D, max_Q: Max values for seasonal auto ARIMA
            m: Seasonal period
            seasonal: Whether to use SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order if seasonal else (0, 0, 0, 0)
        self.auto_arima_search = auto_arima_search
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.m = m
        self.seasonal = seasonal
        
        self.model = None
        self.fitted_model = None
    
    def fit(self, y: pd.Series, verbose: bool = False) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to time series.
        
        Args:
            y: Time series to fit
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        if self.auto_arima_search:
            if verbose:
                print("Searching for best ARIMA parameters...")
            
            # Use auto_arima to find best parameters
            self.model = auto_arima(
                y,
                start_p=0, max_p=self.max_p,
                start_q=0, max_q=self.max_q,
                max_d=self.max_d,
                seasonal=self.seasonal,
                m=self.m,
                start_P=0, max_P=self.max_P,
                start_Q=0, max_Q=self.max_Q,
                max_D=self.max_D,
                trace=verbose,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_fits=50
            )
            
            if verbose:
                print(f"Best model: ARIMA{self.model.order}")
                if self.seasonal:
                    print(f"Seasonal order: {self.model.seasonal_order}")
        else:
            # Use specified parameters
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False)
        
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.auto_arima_search:
            forecast = self.model.predict(n_periods=steps)
        else:
            forecast = self.fitted_model.forecast(steps=steps)
        
        return np.array(forecast)
    
    def forecast_rolling(self, 
                        train: pd.Series,
                        test: pd.Series,
                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform rolling window forecasting.
        
        Args:
            train: Training time series
            test: Test time series
            verbose: Whether to print progress
            
        Returns:
            Tuple of (predictions, actuals)
        """
        predictions = []
        actuals = []
        
        # Start with training data
        history = list(train)
        
        for i, actual in enumerate(test):
            if verbose and i % 5 == 0:
                print(f"Forecasting step {i+1}/{len(test)}")
            
            # Fit model on history
            try:
                model = auto_arima(
                    history,
                    start_p=0, max_p=min(3, self.max_p),
                    start_q=0, max_q=min(3, self.max_q),
                    max_d=self.max_d,
                    seasonal=self.seasonal,
                    m=self.m,
                    suppress_warnings=True,
                    error_action='ignore',
                    stepwise=True
                )
                
                # Forecast next step
                forecast = model.predict(n_periods=1)[0]
                predictions.append(forecast)
                actuals.append(actual)
                
                # Add actual value to history
                history.append(actual)
                
            except Exception as e:
                if verbose:
                    print(f"Error at step {i}: {e}")
                # Use naive forecast (last value)
                predictions.append(history[-1])
                actuals.append(actual)
                history.append(actual)
        
        return np.array(predictions), np.array(actuals)


def evaluate_arima_on_series(series: pd.Series,
                             train_years: Tuple[int, int],
                             test_years: Tuple[int, int],
                             **arima_kwargs) -> Dict:
    """
    Evaluate ARIMA on a single time series.
    
    Args:
        series: Time series indexed by year
        train_years: (start, end) for training
        test_years: (start, end) for testing
        **arima_kwargs: Arguments for ARIMAForecaster
        
    Returns:
        Dictionary with predictions and metrics
    """
    # Split data
    train = series[(series.index >= train_years[0]) & (series.index <= train_years[1])]
    test = series[(series.index >= test_years[0]) & (series.index <= test_years[1])]
    
    if len(train) < 3 or len(test) < 1:
        return None
    
    # Create forecaster
    forecaster = ARIMAForecaster(**arima_kwargs)
    
    # Rolling forecast
    try:
        predictions, actuals = forecaster.forecast_rolling(train, test, verbose=False)
        
        # Compute metrics
        metrics = compute_all_metrics(actuals, predictions, y_train=train.values)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics,
            'train_size': len(train),
            'test_size': len(test)
        }
    except Exception as e:
        print(f"Error evaluating series: {e}")
        return None


def evaluate_arima_on_multiple_series(series_dict: Dict[str, pd.Series],
                                      train_years: Tuple[int, int],
                                      test_years: Tuple[int, int],
                                      max_series: Optional[int] = None,
                                      **arima_kwargs) -> Dict:
    """
    Evaluate ARIMA on multiple time series.
    
    Args:
        series_dict: Dictionary of time series
        train_years: (start, end) for training
        test_years: (start, end) for testing
        max_series: Maximum number of series to evaluate
        **arima_kwargs: Arguments for ARIMAForecaster
        
    Returns:
        Dictionary with aggregated results
    """
    results = {}
    all_predictions = []
    all_actuals = []
    
    series_list = list(series_dict.items())
    if max_series:
        series_list = series_list[:max_series]
    
    print(f"Evaluating ARIMA on {len(series_list)} time series...")
    
    for i, (series_id, series) in enumerate(series_list):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(series_list)}")
        
        result = evaluate_arima_on_series(
            series, train_years, test_years, **arima_kwargs
        )
        
        if result:
            results[series_id] = result
            all_predictions.extend(result['predictions'])
            all_actuals.extend(result['actuals'])
    
    # Compute overall metrics
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


if __name__ == "__main__":
    # Test ARIMA forecaster
    np.random.seed(42)
    
    # Create synthetic time series
    years = np.arange(2010, 2024)
    trend = np.linspace(100, 150, len(years))
    noise = np.random.normal(0, 5, len(years))
    values = trend + noise
    
    series = pd.Series(values, index=years)
    
    print("Testing ARIMA Forecaster")
    print("="*50)
    
    # Split data
    train = series[series.index <= 2020]
    test = series[series.index > 2020]
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Fit and forecast
    forecaster = ARIMAForecaster(auto_arima_search=True)
    predictions, actuals = forecaster.forecast_rolling(train, test, verbose=True)
    
    print("\nPredictions vs Actuals:")
    for year, pred, actual in zip(test.index, predictions, actuals):
        print(f"  {year}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={abs(pred-actual):.2f}")
    
    # Compute metrics
    metrics = compute_all_metrics(actuals, predictions)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

