"""
Unit Tests for GARCH Volatility Forecaster
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.garch_models import fit_garch, fit_multiple_garch_models, get_conditional_volatility
from src.forecasting import forecast_volatility, forecast_evaluation_metrics


class TestGARCHModels:
    """Test GARCH model fitting."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return pd.Series(
            np.random.normal(0.0005, 0.015, 500),
            index=pd.date_range('2022-01-01', periods=500, freq='B')
        )
    
    def test_garch_fit_returns_correct_structure(self, sample_returns):
        """Test that GARCH fit returns correct result structure."""
        result = fit_garch(sample_returns, vol='GARCH')
        
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'beta')
        assert hasattr(result, 'persistence')
        assert hasattr(result, 'half_life')
        assert hasattr(result, 'unconditional_vol')
        assert hasattr(result, 'aic')
        assert hasattr(result, 'fitted_model')
    
    def test_garch_persistence_bounded(self, sample_returns):
        """Test that persistence is bounded [0, 1]."""
        result = fit_garch(sample_returns, vol='GARCH')
        
        assert 0 <= result.persistence <= 1.0
    
    def test_garch_positive_volatility(self, sample_returns):
        """Test that unconditional volatility is positive."""
        result = fit_garch(sample_returns, vol='GARCH')
        
        assert result.unconditional_vol > 0
    
    def test_egarch_fits(self, sample_returns):
        """Test that EGARCH model fits successfully."""
        result = fit_garch(sample_returns, vol='EGARCH')
        
        assert result.model_name == 'EGARCH(1,1)'
        assert result.fitted_model is not None
    
    def test_gjr_garch_fits(self, sample_returns):
        """Test that GJR-GARCH model fits successfully."""
        result = fit_garch(sample_returns, vol='GJR-GARCH')
        
        assert result.model_name == 'GJR-GARCH(1,1)'
        assert result.fitted_model is not None
    
    def test_fit_multiple_models(self, sample_returns):
        """Test fitting multiple GARCH models."""
        results = fit_multiple_garch_models(
            sample_returns, 
            models=['GARCH', 'EGARCH', 'GJR-GARCH']
        )
        
        assert 'GARCH' in results
        assert 'EGARCH' in results
        assert 'GJR-GARCH' in results
    
    def test_conditional_volatility_extraction(self, sample_returns):
        """Test extraction of conditional volatility."""
        result = fit_garch(sample_returns, vol='GARCH')
        cond_vol = get_conditional_volatility(result)
        
        assert len(cond_vol) == len(sample_returns)
        assert cond_vol.min() > 0
    
    def test_aic_bic_valid(self, sample_returns):
        """Test that AIC and BIC are valid numbers."""
        result = fit_garch(sample_returns, vol='GARCH')
        
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)


class TestForecasting:
    """Test volatility forecasting."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted GARCH model for testing."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.0005, 0.015, 500),
            index=pd.date_range('2022-01-01', periods=500, freq='B')
        )
        return fit_garch(returns, vol='GARCH')
    
    def test_forecast_correct_horizon(self, fitted_model):
        """Test forecast has correct horizon."""
        horizon = 21
        forecast = forecast_volatility(fitted_model, horizon=horizon)
        
        assert forecast.forecast_horizon == horizon
        assert len(forecast.point_forecast) == horizon
        assert len(forecast.lower_bound) == horizon
        assert len(forecast.upper_bound) == horizon
    
    def test_forecast_positive(self, fitted_model):
        """Test that forecasts are positive."""
        forecast = forecast_volatility(fitted_model, horizon=21)
        
        assert (forecast.point_forecast > 0).all()
        assert (forecast.lower_bound >= 0).all()
    
    def test_confidence_interval_order(self, fitted_model):
        """Test that lower < point < upper."""
        forecast = forecast_volatility(fitted_model, horizon=21)
        
        assert (forecast.lower_bound <= forecast.point_forecast).all()
        assert (forecast.point_forecast <= forecast.upper_bound).all()
    
    def test_forecast_to_dataframe(self, fitted_model):
        """Test forecast conversion to DataFrame."""
        forecast = forecast_volatility(fitted_model, horizon=21)
        df = forecast.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'Forecast' in df.columns
        assert 'Lower' in df.columns
        assert 'Upper' in df.columns
        assert len(df) == 21


class TestForecastEvaluation:
    """Test forecast evaluation metrics."""
    
    def test_evaluation_metrics_structure(self):
        """Test forecast evaluation returns correct metrics."""
        # Create sample forecast and realized
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        forecast = pd.Series(np.random.uniform(0.1, 0.3, 100), index=dates)
        realized = pd.Series(np.random.uniform(0.1, 0.3, 100), index=dates)
        
        metrics = forecast_evaluation_metrics(forecast, realized)
        
        assert 'ME (Bias)' in metrics
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'Correlation' in metrics
    
    def test_perfect_forecast_metrics(self):
        """Test metrics for perfect forecast."""
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        values = np.random.uniform(0.1, 0.3, 100)
        forecast = pd.Series(values, index=dates)
        realized = pd.Series(values, index=dates)
        
        metrics = forecast_evaluation_metrics(forecast, realized)
        
        assert abs(metrics['ME (Bias)']) < 1e-10
        assert metrics['Correlation'] > 0.999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
