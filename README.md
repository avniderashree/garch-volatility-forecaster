# GARCH Volatility Forecaster

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready Python implementation of GARCH volatility modeling and forecasting, with VIX benchmarking and comprehensive model diagnostics.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [What You'll See](#-what-youll-see)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Sample Results](#-sample-results)
- [Usage Examples](#-usage-examples)
- [Visualizations](#-visualizations)
- [Troubleshooting](#-troubleshooting)
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

### Real-World Applications

| Application | How GARCH is Used |
|-------------|-------------------|
| **Options Pricing** | More accurate implied volatility estimation |
| **Risk Management** | Time-varying VaR and Expected Shortfall |
| **Portfolio Optimization** | Dynamic volatility inputs for mean-variance |
| **Algorithmic Trading** | Volatility regime detection |

---

## ✨ Features

### GARCH Models Implemented

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **GARCH(1,1)** | Standard model | Symmetric response to shocks |
| **EGARCH(1,1)** | Exponential GARCH | Captures asymmetric effects (leverage) |

### Additional Capabilities

- ✅ **VIX Benchmarking** — Compare GARCH estimates with market-implied volatility
- ✅ **Rolling Forecasts** — Out-of-sample forecast generation
- ✅ **Term Structure** — Volatility forecasts across multiple horizons (1D to 1Y)
- ✅ **Model Diagnostics** — Residual analysis, Q-Q plots, ACF
- ✅ **Stationarity Tests** — ADF and KPSS tests
- ✅ **ARCH-LM Test** — Verify presence of ARCH effects
- ✅ **Professional Visualizations** — 5 publication-ready chart types

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/avniderashree/garch-volatility-forecaster.git
cd garch-volatility-forecaster
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `arch` — GARCH model estimation
- `statsmodels` — Stationarity tests
- `yfinance` — Market data
- `pandas`, `numpy`, `scipy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization

### Step 4: Run the Analysis

```bash
python main.py
```

### Step 5: Explore Interactively (Optional)

```bash
jupyter notebook notebooks/garch_analysis.ipynb
```

---

## 🖥️ What You'll See

When you run `python main.py`, the script produces this output:

```
============================================================
 GARCH VOLATILITY FORECASTER
============================================================

This analysis models and forecasts market volatility using:
  1. GARCH(1,1) - Standard model
  2. EGARCH(1,1) - Exponential GARCH (asymmetric)

------------------------------------------------------------
 STEP 1: Loading Market Data
------------------------------------------------------------
Fetching data for SPY...
Calculating returns...
Data loaded: 1255 trading days
Date range: 2021-01-12 to 2026-01-09

Asset: SPY

Return Statistics:
  • Mean daily return:     0.0538%
  • Daily volatility:      1.0749%
  • Annualized volatility: 17.06%
  • Skewness:              0.127
  • Kurtosis:              8.136

------------------------------------------------------------
 STEP 2: Pre-Modeling Diagnostics
------------------------------------------------------------

Stationarity Tests:
  • ADF Test:  stat=-22.046, p-value=0.0000
    → ✓ Stationary (reject H0 if p<0.05)
  • KPSS Test: stat=0.122, p-value=0.1000
    → ✓ Stationary (fail to reject H0 if p>0.05)

ARCH Effect Test (Engle's LM test):
  • LM stat=159.46, p-value=0.0000
    → ✓ ARCH effects present

------------------------------------------------------------
 STEP 3: Fitting GARCH Models
------------------------------------------------------------

📊 Model Comparison (sorted by AIC):
      Model α (ARCH) β (GARCH) Persistence Half-Life Long-Run Vol    AIC
 GARCH(1,1)   0.1251    0.8546      0.9797 33.8 days       18.75% 3375.2
EGARCH(1,1)   0.2403    0.9710      1.2113  inf days         nan% 3379.9

✓ Best model by AIC: GARCH(1,1)

📈 Model Interpretation (GARCH(1,1)):
  • α = 0.1251 → Shock impact coefficient
  • β = 0.8546 → Volatility persistence
  • α + β = 0.9797 → Total persistence
  • Half-life = 33.8 days → Shock decay time

------------------------------------------------------------
 STEP 4: Volatility Forecasting
------------------------------------------------------------

Volatility Forecast (Annualized):
---------------------------------------------
Day      Forecast     Lower 95%    Upper 95%
---------------------------------------------
1             10.08%       7.12%      13.05%
5             11.02%       7.78%      14.26%
21            13.63%       9.62%      17.64%
---------------------------------------------

Volatility Term Structure:
 Horizon (days) Term  Volatility
              1   1D    10.08%
              5   5D    11.02%
             21   1M    13.63%
            252   1Y    18.72%

------------------------------------------------------------
 STEP 5: VIX Comparison
------------------------------------------------------------

GARCH(1,1) vs VIX Comparison:
  • Correlation:      0.761
  • Mean Difference:  -3.27%
  • Tracking Error:   4.30%
  ✓ Strong correlation with VIX suggests model captures market fear

Current Levels:
  • GARCH Volatility: 9.98%
  • VIX:              14.49%
  • Difference:       -4.51%

------------------------------------------------------------
 STEP 6: Generating Visualizations
------------------------------------------------------------

Saving charts to ./output/ directory...
  ✓ returns_volatility.png
  ✓ garch_diagnostics.png
  ✓ volatility_forecast.png
  ✓ model_comparison.png
  ✓ vix_comparison.png

============================================================
 ANALYSIS COMPLETE
============================================================

📊 Key Findings:
  • Asset analyzed: SPY
  • Best model: GARCH(1,1)
  • Volatility persistence: 0.9797
  • Current GARCH vol: 9.98%
  • 21-day forecast: 13.63%

📁 Output files saved to ./output/

Done! ✅
```

---

## 📁 Project Structure

```
garch-volatility-forecaster/
│
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── src/                        # Core modules
│   ├── __init__.py
│   ├── data_loader.py          # Market data fetching & preprocessing
│   ├── garch_models.py         # GARCH model fitting & diagnostics
│   ├── forecasting.py          # Volatility forecasting & evaluation
│   └── visualization.py        # Charts and plots
│
├── notebooks/
│   └── garch_analysis.ipynb    # Interactive Jupyter walkthrough
│
├── tests/
│   └── test_garch.py           # Unit tests (pytest)
│
└── output/                     # Generated visualizations
    ├── returns_volatility.png  # Returns & volatility overview
    ├── garch_diagnostics.png   # Model diagnostics
    ├── volatility_forecast.png # 21-day forecast
    ├── model_comparison.png    # GARCH vs EGARCH
    └── vix_comparison.png      # GARCH vs VIX
```

### Module Descriptions

| Module | Functions | Purpose |
|--------|-----------|---------|
| `data_loader.py` | `fetch_stock_data()`, `fetch_vix_data()`, `calculate_returns()` | Load market data from Yahoo Finance |
| `garch_models.py` | `fit_garch()`, `fit_multiple_garch_models()`, `stationarity_tests()` | Fit GARCH models, extract parameters |
| `forecasting.py` | `forecast_volatility()`, `compare_with_vix()`, `term_structure_forecast()` | Generate forecasts, evaluate accuracy |
| `visualization.py` | `plot_return_and_volatility()`, `plot_garch_diagnostics()`, etc. | Create publication-ready charts |

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

### Parameter Interpretation

| Parameter | Typical Range | Interpretation |
|-----------|---------------|----------------|
| α (alpha) | 0.05 - 0.15 | Higher = shocks have bigger immediate impact |
| β (beta) | 0.80 - 0.95 | Higher = volatility persists longer |
| α + β | 0.95 - 0.99 | Higher = slower mean reversion |

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

This captures the empirical observation that negative returns increase volatility more than positive returns of the same magnitude ("fear is stronger than greed").

### Pre-Modeling Tests

Before fitting GARCH, we verify:

1. **Stationarity (ADF/KPSS)**: Returns must be stationary
2. **ARCH Effects (Engle's LM)**: Volatility clustering must be present

---

## 📈 Sample Results

### Model Comparison (SPY, 5 Years)

| Model | α (ARCH) | β (GARCH) | Persistence | Half-Life | AIC |
|-------|----------|-----------|-------------|-----------|-----|
| GARCH(1,1) | 0.1251 | 0.8546 | 0.9797 | 33.8 days | 3375.2 |
| EGARCH(1,1) | 0.2403 | 0.9710 | 1.2113 | ∞ | 3379.9 |

**Winner: GARCH(1,1)** (lower AIC)

### VIX Comparison

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Correlation | 0.761 | Strong relationship |
| Mean Difference | -3.27% | GARCH underestimates VIX |
| Tracking Error | 4.30% | Day-to-day deviation |

### 21-Day Forecast

| Day | Forecast | 95% CI |
|-----|----------|--------|
| 1 | 10.08% | [7.12%, 13.05%] |
| 5 | 11.02% | [7.78%, 14.26%] |
| 21 | 13.63% | [9.62%, 17.64%] |

### Key Insights

1. **High Persistence (0.98)**: Volatility shocks take ~34 days to decay 50%
2. **Strong VIX Correlation (0.76)**: GARCH captures market fear effectively
3. **GARCH Underestimates VIX**: GARCH vol typically 3% below VIX (VIX includes risk premium)
4. **Mean Reversion**: Forecasts converge to long-run vol of ~18.75%

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

# 21-day forecast with 95% confidence
forecast = forecast_volatility(result, horizon=21, confidence=0.95)
forecast_df = forecast.to_dataframe()
print(forecast_df.head())

# Volatility term structure
term_struct = term_structure_forecast(result, horizons=[1, 5, 21, 63, 252])
print(term_struct)
```

### Compare with VIX

```python
from src.forecasting import compare_with_vix

metrics = compare_with_vix(cond_vol, vix)
print(f"Correlation with VIX: {metrics['Correlation']:.3f}")
print(f"Tracking Error: {metrics['Tracking Error']*100:.2f}%")
```

### Custom Ticker Analysis

```python
# Analyze a different stock/ETF
prices, returns, realized_vol, vix = prepare_volatility_data("QQQ", "3y")
result = fit_garch(returns, vol='GARCH')
print(f"QQQ Volatility: {result.unconditional_vol:.2%}")
```

---

## 📊 Visualizations

The project generates five professional charts saved to `output/`:

### 1. Returns and Volatility (`returns_volatility.png`)
Three-panel view showing:
- Daily returns (positive/negative colored)
- Volatility comparison (GARCH vs Realized vs VIX)
- Volatility clustering (squared returns)

### 2. GARCH Diagnostics (`garch_diagnostics.png`)
Four-panel model diagnostics:
- Standardized residuals time series
- Residual distribution vs normal
- ACF of squared residuals (should be insignificant)
- Q-Q plot (should follow 45° line)

### 3. Volatility Forecast (`volatility_forecast.png`)
Historical volatility with 21-day forecast and 95% confidence interval.

### 4. Model Comparison (`model_comparison.png`)
Compare conditional volatility from GARCH and EGARCH against realized volatility.

### 5. VIX Comparison (`vix_comparison.png`)
Scatter plot (with R² regression) and time series comparing GARCH with VIX.

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'arch'` | Run `pip install arch` |
| `yfinance not downloading data` | Check internet connection; try `pip install --upgrade yfinance` |
| `matplotlib backend error` | Run `pip install pyqt5` or use `%matplotlib inline` in Jupyter |
| `EGARCH persistence > 1` | This is expected for EGARCH; it uses log variance so persistence interpretation differs |

### Running Tests

```bash
pytest tests/test_garch.py -v
```

---

## 🎓 Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Time Series** | GARCH, EGARCH, stationarity testing, volatility modeling |
| **Econometrics** | Maximum Likelihood Estimation, ARCH-LM test, AIC/BIC model selection |
| **Python** | arch, statsmodels, pandas, numpy, scipy, yfinance |
| **ML/Stats** | Hypothesis testing, confidence intervals, forecast evaluation |
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

*This project is part of a quantitative finance portfolio. See my other projects:*

- [Portfolio VaR Calculator](https://github.com/avniderashree/portfolio-var-calculator) — Value at Risk with Historical, Parametric, and Monte Carlo methods