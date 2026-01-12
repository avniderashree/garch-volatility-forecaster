"""
GARCH Models Module
Implements GARCH, EGARCH, GJR-GARCH volatility models.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, EGARCH, FIGARCH
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class GARCHResult:
    """Container for GARCH model results."""
    model_name: str
    omega: float          # Constant term
    alpha: float          # ARCH term (shock impact)
    beta: float           # GARCH term (persistence)
    persistence: float    # alpha + beta (volatility persistence)
    half_life: float      # Days for shock to decay 50%
    unconditional_vol: float  # Long-run annualized volatility
    aic: float
    bic: float
    log_likelihood: float
    fitted_model: object
    
    def __str__(self):
        return (
            f"{self.model_name} Results:\n"
            f"  ω (omega):     {self.omega:.6f}\n"
            f"  α (alpha):     {self.alpha:.4f}\n"
            f"  β (beta):      {self.beta:.4f}\n"
            f"  Persistence:   {self.persistence:.4f}\n"
            f"  Half-life:     {self.half_life:.1f} days\n"
            f"  Unconditional Vol: {self.unconditional_vol:.2%} (annualized)\n"
            f"  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}"
        )


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = 'Constant',
    vol: str = 'GARCH',
    dist: str = 't',
    rescale: bool = True
) -> GARCHResult:
    """
    Fit a GARCH model to return series.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series (in decimal, e.g., 0.01 for 1%)
    p : int
        Order of GARCH term (lagged variance)
    q : int
        Order of ARCH term (lagged squared returns)
    mean : str
        Mean model ('Constant', 'Zero', 'AR')
    vol : str
        Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH', 'FIGARCH')
    dist : str
        Error distribution ('normal', 't', 'skewt')
    rescale : bool
        Whether to rescale returns (*100) for numerical stability
    
    Returns:
    --------
    GARCHResult
        Fitted model results
    """
    # Rescale returns for numerical stability (work in percentage points)
    if rescale:
        returns_scaled = returns * 100
    else:
        returns_scaled = returns
    
    # Create model
    model = arch_model(
        returns_scaled,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist
    )
    
    # Fit model
    result = model.fit(disp='off', show_warning=False)
    
    # Extract parameters
    params = result.params
    
    # Get model-specific parameters
    if vol == 'GARCH':
        omega = params.get('omega', 0)
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
    elif vol == 'EGARCH':
        omega = params.get('omega', 0)
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
    elif vol == 'GJR-GARCH':
        omega = params.get('omega', 0)
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        gamma = params.get('gamma[1]', 0)
        # Effective alpha for persistence calculation
        alpha = alpha + gamma / 2
    else:
        omega = params.get('omega', 0)
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
    
    # Calculate derived statistics
    persistence = alpha + beta
    
    # Half-life of volatility shocks
    if persistence < 1 and persistence > 0:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.inf
    
    # Unconditional (long-run) variance
    if persistence < 1:
        unconditional_var = omega / (1 - persistence)
        # Convert back from percentage if rescaled
        if rescale:
            unconditional_var = unconditional_var / 10000
        unconditional_vol = np.sqrt(unconditional_var * 252)  # Annualize
    else:
        unconditional_vol = np.nan
    
    return GARCHResult(
        model_name=f"{vol}({p},{q})",
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        half_life=half_life,
        unconditional_vol=unconditional_vol,
        aic=result.aic,
        bic=result.bic,
        log_likelihood=result.loglikelihood,
        fitted_model=result
    )


def fit_multiple_garch_models(
    returns: pd.Series,
    models: List[str] = ['GARCH', 'EGARCH']
) -> Dict[str, GARCHResult]:
    """
    Fit multiple GARCH model variants and compare.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    models : List[str]
        List of volatility models to fit ('GARCH', 'EGARCH')
    
    Returns:
    --------
    Dict[str, GARCHResult]
        Dictionary of fitted models
    """
    results = {}
    
    for model_type in models:
        try:
            result = fit_garch(returns, vol=model_type)
            results[model_type] = result
        except Exception as e:
            print(f"Warning: Could not fit {model_type}: {e}")
    
    return results


def get_conditional_volatility(
    garch_result: GARCHResult,
    annualize: bool = True
) -> pd.Series:
    """
    Extract conditional volatility from fitted model.
    
    Parameters:
    -----------
    garch_result : GARCHResult
        Fitted GARCH model result
    annualize : bool
        Whether to annualize volatility
    
    Returns:
    --------
    pd.Series
        Conditional volatility series
    """
    # Get conditional variance
    cond_var = garch_result.fitted_model.conditional_volatility
    
    # Model was fitted on scaled returns (*100), so variance is in %^2
    # Convert back to decimal: divide by 100
    cond_vol = cond_var / 100
    
    if annualize:
        cond_vol = cond_vol * np.sqrt(252)
    
    return cond_vol


def model_comparison_table(results: Dict[str, GARCHResult]) -> pd.DataFrame:
    """
    Create comparison table of GARCH models.
    
    Parameters:
    -----------
    results : Dict[str, GARCHResult]
        Dictionary of fitted models
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    data = []
    for name, result in results.items():
        data.append({
            'Model': result.model_name,
            'α (ARCH)': f"{result.alpha:.4f}",
            'β (GARCH)': f"{result.beta:.4f}",
            'Persistence': f"{result.persistence:.4f}",
            'Half-Life': f"{result.half_life:.1f} days",
            'Long-Run Vol': f"{result.unconditional_vol:.2%}",
            'AIC': f"{result.aic:.1f}",
            'BIC': f"{result.bic:.1f}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('AIC')  # Sort by AIC (lower is better)
    
    return df


def stationarity_tests(returns: pd.Series) -> Dict[str, Tuple[float, float]]:
    """
    Perform stationarity tests on return series.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    
    Returns:
    --------
    Dict with test statistics and p-values
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    # ADF test (null: unit root / non-stationary)
    adf_stat, adf_pval, _, _, _, _ = adfuller(returns.dropna())
    
    # KPSS test (null: stationary)
    kpss_stat, kpss_pval, _, _ = kpss(returns.dropna(), regression='c')
    
    return {
        'ADF': (adf_stat, adf_pval),
        'KPSS': (kpss_stat, kpss_pval)
    }


def arch_effect_test(returns: pd.Series, lags: int = 10) -> Tuple[float, float]:
    """
    Test for ARCH effects using Engle's ARCH-LM test.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    lags : int
        Number of lags for the test
    
    Returns:
    --------
    Tuple[float, float]
        Test statistic and p-value
    """
    from statsmodels.stats.diagnostic import het_arch
    
    stat, pval, _, _ = het_arch(returns.dropna(), nlags=lags)
    
    return stat, pval


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Generate sample GARCH(1,1) returns
    n = 1000
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    returns = pd.Series(np.random.normal(0, 0.01, n))
    
    print("Testing GARCH Models")
    print("=" * 50)
    
    # Fit models
    results = fit_multiple_garch_models(returns)
    
    for name, result in results.items():
        print(f"\n{result}")
    
    print("\n" + "=" * 50)
    print("\nModel Comparison:")
    print(model_comparison_table(results).to_string(index=False))
