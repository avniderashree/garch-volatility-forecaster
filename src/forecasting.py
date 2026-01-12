"""
Forecasting Module
Generates volatility forecasts and evaluates prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class ForecastResult:
    """Container for forecast results."""
    model_name: str
    forecast_horizon: int
    point_forecast: np.ndarray      # Point estimates
    lower_bound: np.ndarray         # Lower confidence interval
    upper_bound: np.ndarray         # Upper confidence interval
    confidence_level: float
    forecast_dates: pd.DatetimeIndex
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame."""
        return pd.DataFrame({
            'Forecast': self.point_forecast,
            'Lower': self.lower_bound,
            'Upper': self.upper_bound
        }, index=self.forecast_dates)


def forecast_volatility(
    garch_result,
    horizon: int = 21,
    confidence: float = 0.95,
    annualize: bool = True
) -> ForecastResult:
    """
    Generate volatility forecasts from fitted GARCH model.
    
    Parameters:
    -----------
    garch_result : GARCHResult
        Fitted GARCH model result
    horizon : int
        Forecast horizon in days
    confidence : float
        Confidence level for prediction intervals
    annualize : bool
        Whether to annualize volatility forecasts
    
    Returns:
    --------
    ForecastResult
        Forecast results with confidence intervals
    """
    model = garch_result.fitted_model
    
    # Generate forecasts
    forecasts = model.forecast(horizon=horizon)
    
    # Extract variance forecasts
    variance_forecast = forecasts.variance.iloc[-1].values
    
    # Convert to volatility (remember model was fit on *100 scaled returns)
    vol_forecast = np.sqrt(variance_forecast) / 100
    
    if annualize:
        vol_forecast = vol_forecast * np.sqrt(252)
    
    # Calculate confidence intervals
    # Using chi-square distribution for variance
    alpha = 1 - confidence
    df = len(model.resid) - len(model.params)
    
    # Simplified CI using normal approximation for volatility
    z = stats.norm.ppf(1 - alpha / 2)
    std_error = vol_forecast * 0.15  # Approximate SE
    
    lower = vol_forecast - z * std_error
    upper = vol_forecast + z * std_error
    
    # Ensure non-negative
    lower = np.maximum(lower, 0)
    
    # Create forecast dates
    last_date = model.resid.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    return ForecastResult(
        model_name=garch_result.model_name,
        forecast_horizon=horizon,
        point_forecast=vol_forecast,
        lower_bound=lower,
        upper_bound=upper,
        confidence_level=confidence,
        forecast_dates=forecast_dates
    )


def rolling_forecast(
    returns: pd.Series,
    window: int = 252,
    horizon: int = 1,
    vol_model: str = 'GARCH',
    step: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate rolling out-of-sample volatility forecasts.
    
    Parameters:
    -----------
    returns : pd.Series
        Full return series
    window : int
        Estimation window size
    horizon : int
        Forecast horizon (typically 1 for next-day)
    vol_model : str
        Volatility model type
    step : int
        Step size for rolling window
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Forecasted volatility and realized volatility
    """
    from .garch_models import fit_garch
    
    forecasts = []
    realized = []
    dates = []
    
    n = len(returns)
    
    for i in range(window, n - horizon, step):
        # Training data
        train = returns.iloc[i-window:i]
        
        # Fit model
        try:
            result = fit_garch(train, vol=vol_model)
            
            # Forecast
            forecast = forecast_volatility(result, horizon=horizon, annualize=False)
            forecasted_vol = forecast.point_forecast[-1]
            
            # Realized volatility (using squared return as proxy)
            future_returns = returns.iloc[i:i+horizon]
            realized_vol = future_returns.std()
            
            forecasts.append(forecasted_vol * np.sqrt(252))  # Annualize
            realized.append(realized_vol * np.sqrt(252))     # Annualize
            dates.append(returns.index[i])
            
        except Exception as e:
            continue
    
    forecast_series = pd.Series(forecasts, index=dates, name='Forecast')
    realized_series = pd.Series(realized, index=dates, name='Realized')
    
    return forecast_series, realized_series


def forecast_evaluation_metrics(
    forecast: pd.Series,
    realized: pd.Series
) -> Dict[str, float]:
    """
    Calculate forecast evaluation metrics.
    
    Parameters:
    -----------
    forecast : pd.Series
        Forecasted volatility
    realized : pd.Series
        Realized volatility
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    # Align series
    common_idx = forecast.index.intersection(realized.index)
    f = forecast.loc[common_idx]
    r = realized.loc[common_idx]
    
    # Error
    error = f - r
    
    # Mean Error (bias)
    me = error.mean()
    
    # Mean Absolute Error
    mae = np.abs(error).mean()
    
    # Root Mean Square Error
    rmse = np.sqrt((error ** 2).mean())
    
    # Mean Absolute Percentage Error
    mape = (np.abs(error) / r).mean() * 100
    
    # Correlation
    corr = f.corr(r)
    
    # Theil's U (ratio of RMSE to benchmark)
    naive_rmse = np.sqrt(((r - r.shift(1).fillna(r.mean())) ** 2).mean())
    theil_u = rmse / naive_rmse if naive_rmse > 0 else np.nan
    
    # Mincer-Zarnowitz regression R^2
    # Realized = a + b * Forecast + error
    from scipy import stats as scipy_stats
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(f, r)
    mz_r2 = r_value ** 2
    
    return {
        'ME (Bias)': me,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Correlation': corr,
        "Theil's U": theil_u,
        'MZ R²': mz_r2
    }


def compare_with_vix(
    garch_vol: pd.Series,
    vix: pd.Series
) -> Dict[str, float]:
    """
    Compare GARCH volatility estimates with VIX.
    
    Parameters:
    -----------
    garch_vol : pd.Series
        GARCH conditional volatility (annualized)
    vix : pd.Series
        VIX levels (annualized, as decimal)
    
    Returns:
    --------
    Dict[str, float]
        Comparison metrics
    """
    # Align series
    common_idx = garch_vol.index.intersection(vix.index)
    g = garch_vol.loc[common_idx]
    v = vix.loc[common_idx]
    
    # Correlation
    corr = g.corr(v)
    
    # Mean difference
    mean_diff = (g - v).mean()
    
    # Tracking error
    tracking_error = (g - v).std()
    
    # Information ratio
    ir = mean_diff / tracking_error if tracking_error > 0 else np.nan
    
    return {
        'Correlation': corr,
        'Mean Difference': mean_diff,
        'Tracking Error': tracking_error,
        'Information Ratio': ir
    }


def term_structure_forecast(
    garch_result,
    horizons: List[int] = [1, 5, 10, 21, 63, 126, 252]
) -> pd.DataFrame:
    """
    Generate volatility term structure forecast.
    
    Parameters:
    -----------
    garch_result : GARCHResult
        Fitted GARCH model
    horizons : List[int]
        List of forecast horizons
    
    Returns:
    --------
    pd.DataFrame
        Term structure of volatility forecasts
    """
    model = garch_result.fitted_model
    
    # Get forecasts for maximum horizon
    max_horizon = max(horizons)
    forecasts = model.forecast(horizon=max_horizon)
    
    variance = forecasts.variance.iloc[-1].values
    vol = np.sqrt(variance) / 100 * np.sqrt(252)  # Annualize
    
    # Extract specific horizons
    data = []
    for h in horizons:
        data.append({
            'Horizon (days)': h,
            'Term': f"{h}D" if h < 21 else f"{h//21}M" if h < 252 else f"{h//252}Y",
            'Volatility': vol[h-1] if h <= len(vol) else vol[-1]
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Forecasting module loaded successfully.")
    print("Available functions:")
    print("  - forecast_volatility()")
    print("  - rolling_forecast()")
    print("  - forecast_evaluation_metrics()")
    print("  - compare_with_vix()")
    print("  - term_structure_forecast()")
