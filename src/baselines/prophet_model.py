"""
Prophet baseline model for time series forecasting.
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Tuple, Optional, Dict
import warnings
import sys
sys.path.append('..')
from utils.metrics import compute_all_metrics

warnings.filterwarnings('ignore')


class ProphetForecaster:
    """Prophet forecaster for time series."""
    
    def __init__(self,
                 growth: str = 'linear',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = False,
                 weekly_seasonality: bool = False,
                 daily_seasonality: bool = False):
        """
        Initialize Prophet forecaster.
        
        Args:
            growth: 'linear' or 'logistic'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Strength of seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = None
    
    def fit(self, series: pd.Series, verbose: bool = False) -> 'ProphetForecaster':
        """
        Fit Prophet model to time series.
        
        Args:
            series: Time series to fit (index should be dates or convertible to dates)
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        # Prepare data in Prophet format
        df = pd.DataFrame({
            'ds': pd.to_datetime(series.index.astype(str), format='%Y'),
            'y': series.values
        })
        
        # Initialize Prophet model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            uncertainty_samples=0  # Faster prediction
        )
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df, verbose=verbose)
        
        return self
    
    def predict(self, steps: int = 1, last_date: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            last_date: Last date in training data
            
        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        if last_date is None:
            last_date = pd.Timestamp.now()
        
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(years=1),
            periods=steps,
            freq='YS'  # Year start
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values
    
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
        history = train.copy()
        
        for i, (year, actual) in enumerate(test.items()):
            if verbose and i % 5 == 0:
                print(f"Forecasting year {year} ({i+1}/{len(test)})")
            
            try:
                # Fit model on history
                forecaster = ProphetForecaster(
                    growth=self.growth,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    seasonality_prior_scale=self.seasonality_prior_scale,
                    seasonality_mode=self.seasonality_mode,
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality
                )
                forecaster.fit(history, verbose=False)
                
                # Forecast next year
                last_date = pd.to_datetime(str(history.index[-1]), format='%Y')
                forecast = forecaster.predict(steps=1, last_date=last_date)
                
                predictions.append(forecast[0])
                actuals.append(actual)
                
                # Add actual value to history
                history = pd.concat([history, pd.Series([actual], index=[year])])
                
            except Exception as e:
                if verbose:
                    print(f"Error at year {year}: {e}")
                # Use naive forecast (last value)
                predictions.append(history.iloc[-1])
                actuals.append(actual)
                history = pd.concat([history, pd.Series([actual], index=[year])])
        
        return np.array(predictions), np.array(actuals)


def evaluate_prophet_on_series(series: pd.Series,
                               train_years: Tuple[int, int],
                               test_years: Tuple[int, int],
                               **prophet_kwargs) -> Dict:
    """
    Evaluate Prophet on a single time series.
    
    Args:
        series: Time series indexed by year
        train_years: (start, end) for training
        test_years: (start, end) for testing
        **prophet_kwargs: Arguments for ProphetForecaster
        
    Returns:
        Dictionary with predictions and metrics
    """
    # Split data
    train = series[(series.index >= train_years[0]) & (series.index <= train_years[1])]
    test = series[(series.index >= test_years[0]) & (series.index <= test_years[1])]
    
    if len(train) < 3 or len(test) < 1:
        return None
    
    # Create forecaster
    forecaster = ProphetForecaster(**prophet_kwargs)
    
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


def evaluate_prophet_on_multiple_series(series_dict: Dict[str, pd.Series],
                                       train_years: Tuple[int, int],
                                       test_years: Tuple[int, int],
                                       max_series: Optional[int] = None,
                                       **prophet_kwargs) -> Dict:
    """
    Evaluate Prophet on multiple time series.
    
    Args:
        series_dict: Dictionary of time series
        train_years: (start, end) for training
        test_years: (start, end) for testing
        max_series: Maximum number of series to evaluate
        **prophet_kwargs: Arguments for ProphetForecaster
        
    Returns:
        Dictionary with aggregated results
    """
    results = {}
    all_predictions = []
    all_actuals = []
    
    series_list = list(series_dict.items())
    if max_series:
        series_list = series_list[:max_series]
    
    print(f"Evaluating Prophet on {len(series_list)} time series...")
    
    for i, (series_id, series) in enumerate(series_list):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(series_list)}")
        
        result = evaluate_prophet_on_series(
            series, train_years, test_years, **prophet_kwargs
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
    # Test Prophet forecaster
    np.random.seed(42)
    
    # Create synthetic time series
    years = np.arange(2010, 2024)
    trend = np.linspace(100, 150, len(years))
    noise = np.random.normal(0, 5, len(years))
    values = trend + noise
    
    series = pd.Series(values, index=years)
    
    print("Testing Prophet Forecaster")
    print("="*50)
    
    # Split data
    train = series[series.index <= 2020]
    test = series[series.index > 2020]
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Fit and forecast
    forecaster = ProphetForecaster(growth='linear', yearly_seasonality=False)
    predictions, actuals = forecaster.forecast_rolling(train, test, verbose=True)
    
    print("\nPredictions vs Actuals:")
    for year, pred, actual in zip(test.index, predictions, actuals):
        print(f"  {year}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={abs(pred-actual):.2f}")
    
    # Compute metrics
    metrics = compute_all_metrics(actuals, predictions)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

