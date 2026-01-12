"""
Visualization Module
Charts and plots for GARCH volatility analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_return_and_volatility(
    returns: pd.Series,
    conditional_vol: pd.Series,
    realized_vol: Optional[pd.Series] = None,
    vix: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot returns and volatility comparison.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    conditional_vol : pd.Series
        GARCH conditional volatility (annualized)
    realized_vol : pd.Series, optional
        Realized volatility
    vix : pd.Series, optional
        VIX levels
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Returns
    ax1 = axes[0]
    ax1.fill_between(returns.index, returns * 100, 0, 
                     where=returns >= 0, color='green', alpha=0.5, label='Positive')
    ax1.fill_between(returns.index, returns * 100, 0, 
                     where=returns < 0, color='red', alpha=0.5, label='Negative')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.set_title('Daily Returns', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility Comparison
    ax2 = axes[1]
    ax2.plot(conditional_vol.index, conditional_vol * 100, 
             color='navy', linewidth=1.5, label='GARCH Volatility')
    
    if realized_vol is not None:
        ax2.plot(realized_vol.index, realized_vol * 100, 
                 color='orange', linewidth=1, alpha=0.7, label='Realized Vol (21d)')
    
    if vix is not None:
        ax2.plot(vix.index, vix * 100, 
                 color='red', linewidth=1, alpha=0.7, label='VIX')
    
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.set_title('Volatility Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility Clustering
    ax3 = axes[2]
    sq_returns = (returns ** 2) * 10000  # Squared returns in bp^2
    ax3.bar(sq_returns.index, sq_returns, color='steelblue', alpha=0.7, width=1)
    ax3.set_ylabel('Squared Return (bp²)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Volatility Clustering (Squared Returns)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_garch_diagnostics(
    garch_result,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot GARCH model diagnostics.
    
    Parameters:
    -----------
    garch_result : GARCHResult
        Fitted GARCH model result
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    model = garch_result.fitted_model
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Standardized Residuals
    ax1 = axes[0, 0]
    std_resid = model.std_resid
    ax1.plot(std_resid.index, std_resid, color='steelblue', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title('Standardized Residuals', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Std Residual', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals Distribution
    ax2 = axes[0, 1]
    ax2.hist(std_resid, bins=50, density=True, color='steelblue', 
             edgecolor='white', alpha=0.7, label='Residuals')
    
    # Overlay normal distribution
    x = np.linspace(-4, 4, 100)
    from scipy.stats import norm, t
    ax2.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Normal')
    
    ax2.set_title('Distribution of Standardized Residuals', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Std Residual', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ACF of Squared Residuals
    ax3 = axes[1, 0]
    from statsmodels.graphics.tsaplots import plot_acf
    sq_resid = std_resid ** 2
    plot_acf(sq_resid.dropna(), ax=ax3, lags=30, alpha=0.05)
    ax3.set_title('ACF of Squared Residuals', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Lag', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: QQ Plot
    ax4 = axes[1, 1]
    from scipy.stats import probplot
    probplot(std_resid.dropna(), dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
    ax4.get_lines()[0].set_color('steelblue')
    ax4.get_lines()[1].set_color('red')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forecast(
    historical_vol: pd.Series,
    forecast_result,
    realized_vol: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot volatility forecast with confidence intervals.
    
    Parameters:
    -----------
    historical_vol : pd.Series
        Historical conditional volatility
    forecast_result : ForecastResult
        Forecast results
    realized_vol : pd.Series, optional
        Realized volatility for comparison
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical volatility (last 60 days)
    hist_plot = historical_vol.tail(60) * 100
    ax.plot(hist_plot.index, hist_plot, color='navy', linewidth=2, label='Historical')
    
    # Plot forecast
    forecast_dates = forecast_result.forecast_dates
    forecast_vol = forecast_result.point_forecast * 100
    lower = forecast_result.lower_bound * 100
    upper = forecast_result.upper_bound * 100
    
    ax.plot(forecast_dates, forecast_vol, color='red', linewidth=2, 
            linestyle='--', label='Forecast')
    ax.fill_between(forecast_dates, lower, upper, color='red', alpha=0.2,
                   label=f'{int(forecast_result.confidence_level*100)}% CI')
    
    if realized_vol is not None:
        realized_plot = realized_vol.tail(60) * 100
        ax.plot(realized_plot.index, realized_plot, color='green', 
                linewidth=1, alpha=0.7, label='Realized')
    
    # Add vertical line at forecast start
    ax.axvline(x=hist_plot.index[-1], color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Volatility (%)', fontsize=11)
    ax.set_title(f'{forecast_result.model_name} Volatility Forecast ({forecast_result.forecast_horizon}-day)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: Dict,
    realized_vol: pd.Series,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare conditional volatility from different GARCH models.
    
    Parameters:
    -----------
    results : Dict
        Dictionary of GARCHResult objects
    realized_vol : pd.Series
        Realized volatility for comparison
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from .garch_models import get_conditional_volatility
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    colors = {'GARCH': 'navy', 'EGARCH': 'green', 'GJR-GARCH': 'purple'}
    
    # Plot 1: Volatility estimates
    ax1 = axes[0]
    
    for name, result in results.items():
        cond_vol = get_conditional_volatility(result)
        ax1.plot(cond_vol.index, cond_vol * 100, 
                 color=colors.get(name, 'gray'), linewidth=1.5, 
                 label=f'{result.model_name}', alpha=0.8)
    
    # Add realized vol
    ax1.plot(realized_vol.index, realized_vol * 100, 
             color='red', linewidth=1, alpha=0.5, label='Realized (21d)')
    
    ax1.set_ylabel('Volatility (%)', fontsize=11)
    ax1.set_title('Model Comparison: Conditional Volatility', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model differences from GARCH
    ax2 = axes[1]
    
    if 'GARCH' in results:
        garch_vol = get_conditional_volatility(results['GARCH'])
        
        for name, result in results.items():
            if name != 'GARCH':
                other_vol = get_conditional_volatility(result)
                diff = (other_vol - garch_vol) * 100
                ax2.plot(diff.index, diff, 
                         color=colors.get(name, 'gray'), linewidth=1, 
                         label=f'{name} - GARCH', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Difference (%)', fontsize=11)
    ax2.set_title('Volatility Difference from GARCH(1,1)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_vix_comparison(
    garch_vol: pd.Series,
    vix: pd.Series,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot comparing GARCH volatility with VIX.
    
    Parameters:
    -----------
    garch_vol : pd.Series
        GARCH conditional volatility
    vix : pd.Series
        VIX levels
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Align series
    common_idx = garch_vol.index.intersection(vix.index)
    g = garch_vol.loc[common_idx] * 100
    v = vix.loc[common_idx] * 100
    
    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(v, g, alpha=0.3, s=10, color='steelblue')
    
    # Add 45-degree line
    min_val = min(g.min(), v.min())
    max_val = max(g.max(), v.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='45° Line')
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(v, g)
    ax1.plot(v.sort_values(), intercept + slope * v.sort_values(), 
             'g-', linewidth=2, label=f'Regression (R²={r_value**2:.2f})')
    
    ax1.set_xlabel('VIX (%)', fontsize=11)
    ax1.set_ylabel('GARCH Volatility (%)', fontsize=11)
    ax1.set_title('GARCH vs VIX', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series
    ax2 = axes[1]
    ax2.plot(g.index, g, color='navy', linewidth=1, label='GARCH', alpha=0.8)
    ax2.plot(v.index, v, color='red', linewidth=1, label='VIX', alpha=0.8)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.set_title('GARCH vs VIX Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_return_and_volatility()")
    print("  - plot_garch_diagnostics()")
    print("  - plot_forecast()")
    print("  - plot_model_comparison()")
    print("  - plot_vix_comparison()")
