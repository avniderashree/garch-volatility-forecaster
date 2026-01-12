#!/usr/bin/env python3
"""
GARCH Volatility Forecaster
============================
Main execution script demonstrating GARCH volatility modeling and forecasting.

Author: Avni Derashree
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import prepare_volatility_data, calculate_realized_volatility
from src.garch_models import (
    fit_garch, 
    fit_multiple_garch_models, 
    get_conditional_volatility,
    model_comparison_table,
    stationarity_tests,
    arch_effect_test
)
from src.forecasting import (
    forecast_volatility,
    forecast_evaluation_metrics,
    compare_with_vix,
    term_structure_forecast
)
from src.visualization import (
    plot_return_and_volatility,
    plot_garch_diagnostics,
    plot_forecast,
    plot_model_comparison,
    plot_vix_comparison
)


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}")


def main():
    """Main execution function."""
    
    print_header("GARCH VOLATILITY FORECASTER", "=")
    print("\nThis analysis models and forecasts market volatility using:")
    print("  1. GARCH(1,1) - Standard model")
    print("  2. EGARCH(1,1) - Exponential GARCH (asymmetric)")
    print("  3. GJR-GARCH(1,1) - Leverage effect model")
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_header("STEP 1: Loading Market Data", "-")
    
    ticker = "SPY"  # S&P 500 ETF
    prices, returns, realized_vol, vix = prepare_volatility_data(ticker, period="5y")
    
    print(f"\nAsset: {ticker}")
    print(f"\nReturn Statistics:")
    print(f"  • Mean daily return:     {returns.mean():.4%}")
    print(f"  • Daily volatility:      {returns.std():.4%}")
    print(f"  • Annualized volatility: {returns.std() * np.sqrt(252):.2%}")
    print(f"  • Skewness:              {returns.skew():.3f}")
    print(f"  • Kurtosis:              {returns.kurtosis():.3f}")
    
    # =========================================================================
    # STEP 2: Pre-Modeling Tests
    # =========================================================================
    print_header("STEP 2: Pre-Modeling Diagnostics", "-")
    
    # Stationarity tests
    print("\nStationarity Tests:")
    stat_tests = stationarity_tests(returns)
    
    adf_stat, adf_pval = stat_tests['ADF']
    kpss_stat, kpss_pval = stat_tests['KPSS']
    
    print(f"  • ADF Test:  stat={adf_stat:.3f}, p-value={adf_pval:.4f}")
    print(f"    → {'✓ Stationary' if adf_pval < 0.05 else '✗ Non-stationary'} (reject H0 if p<0.05)")
    
    print(f"  • KPSS Test: stat={kpss_stat:.3f}, p-value={kpss_pval:.4f}")
    print(f"    → {'✓ Stationary' if kpss_pval > 0.05 else '✗ Non-stationary'} (fail to reject H0 if p>0.05)")
    
    # ARCH effect test
    print("\nARCH Effect Test (Engle's LM test):")
    arch_stat, arch_pval = arch_effect_test(returns, lags=10)
    print(f"  • LM stat={arch_stat:.2f}, p-value={arch_pval:.4f}")
    print(f"    → {'✓ ARCH effects present' if arch_pval < 0.05 else '✗ No ARCH effects'}")
    
    if arch_pval > 0.05:
        print("  ⚠️  Warning: ARCH effects not significant. GARCH may not be appropriate.")
    
    # =========================================================================
    # STEP 3: Fit GARCH Models
    # =========================================================================
    print_header("STEP 3: Fitting GARCH Models", "-")
    
    print("\nFitting multiple volatility models...")
    results = fit_multiple_garch_models(returns, models=['GARCH', 'EGARCH', 'GJR-GARCH'])
    
    print("\n" + "-" * 60)
    for name, result in results.items():
        print(f"\n{result}")
    print("-" * 60)
    
    # Model comparison table
    print("\n📊 Model Comparison (sorted by AIC):")
    print(model_comparison_table(results).to_string(index=False))
    
    # Select best model
    best_model_name = min(results, key=lambda x: results[x].aic)
    best_result = results[best_model_name]
    print(f"\n✓ Best model by AIC: {best_result.model_name}")
    
    # Interpretation
    print(f"\n📈 Model Interpretation ({best_result.model_name}):")
    print(f"  • α = {best_result.alpha:.4f} → Shock impact coefficient")
    print(f"  • β = {best_result.beta:.4f} → Volatility persistence")
    print(f"  • α + β = {best_result.persistence:.4f} → Total persistence")
    print(f"  • Half-life = {best_result.half_life:.1f} days → Shock decay time")
    
    if best_result.persistence > 0.99:
        print("  ⚠️  High persistence indicates near-integrated variance (IGARCH-like)")
    
    # =========================================================================
    # STEP 4: Generate Forecasts
    # =========================================================================
    print_header("STEP 4: Volatility Forecasting", "-")
    
    print(f"\nGenerating 21-day volatility forecast using {best_result.model_name}...")
    
    forecast = forecast_volatility(best_result, horizon=21, confidence=0.95)
    forecast_df = forecast.to_dataframe()
    
    print("\nVolatility Forecast (Annualized):")
    print("-" * 45)
    print(f"{'Day':<8} {'Forecast':<12} {'Lower 95%':<12} {'Upper 95%':<12}")
    print("-" * 45)
    
    for i, (date, row) in enumerate(forecast_df.iterrows()):
        if i < 5 or i >= len(forecast_df) - 2:
            print(f"{i+1:<8} {row['Forecast']*100:>10.2f}% {row['Lower']*100:>10.2f}% {row['Upper']*100:>10.2f}%")
        elif i == 5:
            print("    ...")
    
    print("-" * 45)
    
    # Term structure
    print("\nVolatility Term Structure:")
    term_struct = term_structure_forecast(best_result, horizons=[1, 5, 10, 21, 63, 252])
    print(term_struct.to_string(index=False))
    
    # =========================================================================
    # STEP 5: Compare with VIX
    # =========================================================================
    print_header("STEP 5: VIX Comparison", "-")
    
    garch_vol = get_conditional_volatility(best_result)
    vix_comparison = compare_with_vix(garch_vol, vix)
    
    print(f"\n{best_result.model_name} vs VIX Comparison:")
    print(f"  • Correlation:      {vix_comparison['Correlation']:.3f}")
    print(f"  • Mean Difference:  {vix_comparison['Mean Difference']*100:.2f}%")
    print(f"  • Tracking Error:   {vix_comparison['Tracking Error']*100:.2f}%")
    
    if vix_comparison['Correlation'] > 0.7:
        print("  ✓ Strong correlation with VIX suggests model captures market fear")
    
    # Current volatility comparison
    current_garch = garch_vol.iloc[-1] * 100
    current_vix = vix.iloc[-1] * 100
    print(f"\nCurrent Levels:")
    print(f"  • GARCH Volatility: {current_garch:.2f}%")
    print(f"  • VIX:              {current_vix:.2f}%")
    print(f"  • Difference:       {current_garch - current_vix:.2f}%")
    
    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print_header("STEP 6: Generating Visualizations", "-")
    
    os.makedirs('output', exist_ok=True)
    
    print("\nSaving charts to ./output/ directory...")
    
    # Chart 1: Returns and Volatility
    fig1 = plot_return_and_volatility(returns, garch_vol, realized_vol, vix)
    fig1.savefig('output/returns_volatility.png', dpi=150, bbox_inches='tight')
    print("  ✓ returns_volatility.png")
    
    # Chart 2: GARCH Diagnostics
    fig2 = plot_garch_diagnostics(best_result)
    fig2.savefig('output/garch_diagnostics.png', dpi=150, bbox_inches='tight')
    print("  ✓ garch_diagnostics.png")
    
    # Chart 3: Forecast
    fig3 = plot_forecast(garch_vol, forecast, realized_vol)
    fig3.savefig('output/volatility_forecast.png', dpi=150, bbox_inches='tight')
    print("  ✓ volatility_forecast.png")
    
    # Chart 4: Model Comparison
    fig4 = plot_model_comparison(results, realized_vol)
    fig4.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ model_comparison.png")
    
    # Chart 5: VIX Comparison
    fig5 = plot_vix_comparison(garch_vol, vix)
    fig5.savefig('output/vix_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ vix_comparison.png")
    
    plt.close('all')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("ANALYSIS COMPLETE", "=")
    
    print("\n📊 Key Findings:")
    print(f"  • Asset analyzed: {ticker}")
    print(f"  • Best model: {best_result.model_name}")
    print(f"  • Volatility persistence: {best_result.persistence:.4f}")
    print(f"  • Current GARCH vol: {current_garch:.2f}%")
    print(f"  • 21-day forecast: {forecast.point_forecast[-1]*100:.2f}%")
    
    print(f"\n💡 Insights:")
    if best_result.persistence > 0.95:
        print("  • High persistence: volatility shocks decay slowly")
    if 'EGARCH' in best_model_name or 'GJR' in best_model_name:
        print("  • Asymmetric model selected: negative shocks have larger impact")
    if vix_comparison['Correlation'] > 0.8:
        print("  • Strong VIX correlation: model tracks market fear well")
    
    print("\n📁 Output files saved to ./output/")
    print("\nDone! ✅")


if __name__ == "__main__":
    main()
