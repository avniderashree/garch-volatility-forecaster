# GARCH Volatility Forecaster

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready Python implementation of GARCH volatility modeling and forecasting, with VIX benchmarking and comprehensive model diagnostics.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Sample Results](#-sample-results)
- [Usage Examples](#-usage-examples)
- [Visualizations](#-visualizations)
- [Technical Skills Demonstrated](#-technical-skills-demonstrated)
- [References](#-references)

---

## 📊 Overview

Volatility is the cornerstone of financial risk management, options pricing, and portfolio optimization. This project implements **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models to:

1. **Model** time-varying volatility in financial returns
2. **Forecast** future volatility with confidence intervals
3. **Compare** GARCH estimates against VIX (market-implied volatility)

### Why GARCH?

Traditional volatility measures (like rolling standard deviation) assume constant volatility. GARCH captures two key stylized facts of financial markets:

- **Volatility Clustering**: Large price moves tend to be followed by large moves
- **Mean Reversion**: Volatility eventually returns to a long-run average

---

## ✨ Features

### GARCH Models Implemented

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **GARCH(1,1)** | Standard model | Symmetric response to shocks |
| **EGARCH(1,1)** | Exponential GARCH | Captures asymmetric effects |

### Additional Capabilities

- ✅ **VIX Benchmarking** — Compare GARCH estimates with market-implied volatility
- ✅ **Rolling Forecasts** — Out-of-sample forecast generation
- ✅ **Term Structure** — Volatility forecasts across multiple horizons
- ✅ **Model Diagnostics** — Residual analysis, Q-Q plots, ACF
- ✅ **Stationarity Tests** — ADF and KPSS tests
- ✅ **ARCH-LM Test** — Verify presence of ARCH effects
- ✅ **Professional Visualizations** — 5 publication-ready chart types

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/avniderashree/garch-volatility-forecaster.git
cd garch-volatility-forecaster

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
python main.py
```

This will:
1. Download 5 years of SPY (S&P 500 ETF) data and VIX
2. Perform pre-modeling diagnostics (stationarity, ARCH effects)
3. Fit GARCH and EGARCH models
4. Generate 21-day volatility forecasts
5. Compare with VIX
6. Save visualizations to `output/` directory

### Interactive Exploration

```bash
jupyter notebook notebooks/garch_analysis.ipynb
```

---

## 📁 Project Structure

```
garch-volatility-forecaster/
│
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Core modules
│   ├── __init__.py
│   ├── data_loader.py          # Market data fetching
│   ├── garch_models.py         # GARCH model fitting
│   ├── forecasting.py          # Volatility forecasting
│   └── visualization.py        # Charts and plots
│
├── notebooks/
│   └── garch_analysis.ipynb    # Interactive walkthrough
│
├── tests/
│   └── test_garch.py           # Unit tests
│
└── output/                     # Generated visualizations
    ├── returns_volatility.png
    ├── garch_diagnostics.png
    ├── volatility_forecast.png
    ├── model_comparison.png
    └── vix_comparison.png
```

---

## 🧮 Methodology

### GARCH(1,1) Model

The standard GARCH(1,1) model specifies conditional variance as:

```
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
```

Where:
- **ω (omega)**: Base variance level
- **α (alpha)**: Impact of yesterday's shock (ARCH term)
- **β (beta)**: Persistence of yesterday's variance (GARCH term)
- **α + β**: Total persistence (should be < 1 for stationarity)

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Persistence** | α + β | How long shocks persist |
| **Half-Life** | log(0.5) / log(α+β) | Days for shock to decay 50% |
| **Unconditional Vol** | √(ω / (1-α-β)) | Long-run volatility level |

### EGARCH Model

EGARCH models asymmetric volatility (leverage effect):

```
log(σ²ₜ) = ω + α·g(zₜ₋₁) + β·log(σ²ₜ₋₁)
```

This captures the empirical observation that negative returns increase volatility more than positive returns of the same magnitude.

---

## 📈 Sample Results

### Model Comparison (SPY, 5 Years)

| Model | α (ARCH) | β (GARCH) | Persistence | Half-Life | AIC |
|-------|----------|-----------|-------------|-----------|-----|
| GARCH(1,1) | 0.1251 | 0.8546 | 0.9797 | 33.8 days | 3375.2 |
| EGARCH(1,1) | 0.2403 | 0.9710 | 1.2113 | ∞ | 3379.9 |

### VIX Comparison

| Metric | Value |
|--------|-------|
| Correlation | 0.761 |
| Mean Difference | -3.27% |
| Tracking Error | 4.30% |

### Key Insights

1. **High Persistence (0.98)**: Volatility shocks take ~34 days to decay 50%
2. **Strong VIX Correlation (0.76)**: GARCH captures market fear effectively
3. **GARCH Underestimates VIX**: GARCH vol typically 3% below VIX (VIX includes risk premium)

---

## 💻 Usage Examples

### Basic GARCH Fitting

```python
from src.data_loader import prepare_volatility_data
from src.garch_models import fit_garch, get_conditional_volatility

# Load data
prices, returns, realized_vol, vix = prepare_volatility_data("SPY", "5y")

# Fit GARCH(1,1)
result = fit_garch(returns, vol='GARCH')

print(f"Persistence: {result.persistence:.4f}")
print(f"Half-life: {result.half_life:.1f} days")
print(f"Long-run vol: {result.unconditional_vol:.2%}")

# Get conditional volatility
cond_vol = get_conditional_volatility(result)
```

### Volatility Forecasting

```python
from src.forecasting import forecast_volatility, term_structure_forecast

# 21-day forecast
forecast = forecast_volatility(result, horizon=21, confidence=0.95)
forecast_df = forecast.to_dataframe()
print(forecast_df.head())

# Term structure
term_struct = term_structure_forecast(result, horizons=[1, 5, 21, 63, 252])
print(term_struct)
```

### Compare with VIX

```python
from src.forecasting import compare_with_vix

metrics = compare_with_vix(cond_vol, vix)
print(f"Correlation with VIX: {metrics['Correlation']:.3f}")
```

---

## 📊 Visualizations

The project generates five professional charts:

### 1. Returns and Volatility
Three-panel view: daily returns, volatility comparison (GARCH vs Realized vs VIX), and volatility clustering.

### 2. GARCH Diagnostics
Four-panel model diagnostics: standardized residuals, distribution, ACF of squared residuals, Q-Q plot.

### 3. Volatility Forecast
Historical volatility with 21-day forecast and 95% confidence interval.

### 4. Model Comparison
Compare conditional volatility from GARCH and EGARCH models against realized volatility.

### 5. VIX Comparison
Scatter plot and time series comparing GARCH volatility with VIX.

---

## 🎓 Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Time Series** | GARCH, EGARCH, stationarity testing, volatility modeling |
| **Econometrics** | MLE estimation, ARCH-LM test, Kupiec test |
| **Python** | arch, statsmodels, pandas, numpy, scipy, yfinance |
| **ML/Stats** | Hypothesis testing, confidence intervals, model selection (AIC/BIC) |
| **Visualization** | matplotlib, seaborn, publication-quality charts |
| **Software Eng** | Modular design, type hints, docstrings, unit testing |

---

## 📚 References

1. Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
2. Nelson, D. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach*. Econometrica.
3. Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of UK Inflation*. Econometrica.
4. Hull, J.C. (2018). *Options, Futures, and Other Derivatives*. Pearson.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Avni Derashree**  
Quantitative Risk Analyst | Python | Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/avniderashree/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/avniderashree)

---

*This project is part of a quantitative finance portfolio. See my other projects on [GitHub](https://github.com/avniderashree).*