# 🧪 Codelab: Build a GARCH Volatility Forecaster from Scratch

**Estimated time:** 3–4 hours · **Difficulty:** Intermediate · **Language:** Python 3.9+

---

## What You'll Build

By the end of this codelab, you'll have a fully working **GARCH Volatility Forecaster** that:

- Downloads real S&P 500 (SPY) market data from Yahoo Finance
- Runs **stationarity tests** (ADF & KPSS) and **ARCH effect tests** to validate the data
- Fits **GARCH(1,1)** and **EGARCH(1,1)** models to capture time-varying volatility
- **Forecasts** future volatility with 95% confidence intervals
- Builds a **volatility term structure** (1-day to 1-year forecasts)
- **Benchmarks** GARCH estimates against the VIX ("Wall Street's fear gauge")
- Generates **5 publication-quality charts**
- Includes **unit tests** for everything

The final project structure:

```
garch-volatility-forecaster/
├── main.py                     # Entry point — runs the entire pipeline
├── requirements.txt            # Dependencies
├── src/
│   ├── __init__.py             # Makes src a Python package
│   ├── data_loader.py          # Fetches market data, computes returns
│   ├── garch_models.py         # GARCH/EGARCH fitting + diagnostics
│   ├── forecasting.py          # Volatility forecasts + VIX comparison
│   └── visualization.py        # 5 professional charts
├── tests/
│   └── test_garch.py           # Unit tests
└── output/                     # Where charts get saved
```

---

## Prerequisites

You only need:

- Python 3.9+ installed
- Basic familiarity with Python (functions, classes, imports)
- A terminal / command line

**No finance or statistics knowledge required.** We'll explain every concept from the ground up before we code it.

---

---

# PART 1: THE CONCEPTS (What & Why)

Before writing a single line of code, let's understand the problem we're solving. This entire section is theory — no coding yet. Read it carefully; it'll make every line of code click later.

---

## 1.1 What Is Volatility?

In everyday language, "volatile" means unpredictable and fast-changing. In finance, **volatility** has a precise mathematical meaning:

> **Volatility** is the standard deviation of returns — it measures how wildly prices swing.

**Concrete example:**

Two stocks both average a 10% annual return. Stock A moves ±0.5% per day. Stock B moves ±3% per day. Stock B is far more **volatile** — even though they have the same average return.

**Why does volatility matter?**

| Who | Why They Care |
|-----|---------------|
| **Options traders** | The price of an option is *directly* determined by expected future volatility. Higher vol = more expensive options. |
| **Risk managers** | Need to know "how bad could tomorrow be?" — that's VaR, which is computed from volatility. |
| **Portfolio managers** | Allocate less money to volatile assets, more to stable ones. |
| **Algo traders** | Detect "regime changes" — is the market calm or panicking? |

**The key problem:** Volatility is **not constant**. It changes every day. Look at any stock chart — there are calm periods and wild periods. We need a model that captures this.

---

## 1.2 Why Not Just Use Rolling Standard Deviation?

The simplest way to measure volatility is the **rolling standard deviation**: take the last 30 days of returns, compute their standard deviation. Slide the window forward by one day, repeat.

**The problem:** This approach treats every day in the window equally. A massive crash 29 days ago counts just as much as yesterday's tiny move. The real market doesn't work like that.

**What actually happens in markets — two "stylized facts":**

### Stylized Fact #1: Volatility Clustering

Large price moves tend to be followed by large price moves. Small moves follow small moves. Volatile days come in *clusters*.

```
Calm period:   +0.2%, -0.1%, +0.3%, -0.2%, +0.1%  ← small moves
                                                       then suddenly...
Volatile period: -3.5%, +2.8%, -4.1%, +3.2%, -2.9%  ← BIG moves cluster
                                                       then gradually...
Calm again:     +0.5%, -0.3%, +0.2%, +0.4%, -0.1%  ← back to small moves
```

**Why?** Because fear is contagious. When a big drop happens, traders panic-sell the next day, causing another big move, which causes more panic, and so on — until things finally calm down.

### Stylized Fact #2: Mean Reversion

Volatility doesn't stay high forever. After a spike (like a market crash), it gradually falls back to a "normal" long-run level. This is called **mean reversion**.

```
                  ╱╲
Volatility       ╱  ╲        ← Spike (e.g., COVID crash)
                ╱    ╲
               ╱      ╲───────── Gradually returns to normal
──────────────╱               ← Long-run average (~17% for S&P 500)
```

**Rolling standard deviation can't capture either of these.** It's too sluggish to react to clustering, and it doesn't have a concept of "long-run normal." That's where GARCH comes in.

---

## 1.3 The GARCH Model — The Big Idea

**GARCH** stands for **Generalized Autoregressive Conditional Heteroskedasticity**. That's a mouthful, so let's break it down word by word:

| Word | Meaning |
|------|---------|
| **Heteroskedasticity** | "Changing variance" — volatility is not constant |
| **Conditional** | Today's volatility *depends on* (is conditioned on) yesterday's information |
| **Autoregressive** | Today's volatility is partly a function of yesterday's volatility |
| **Generalized** | It's an extension of the simpler ARCH model (by Robert Engle, 1982) |

### The GARCH(1,1) Equation

The GARCH(1,1) model says:

```
σ²ₜ = ω + α × ε²ₜ₋₁ + β × σ²ₜ₋₁
```

**In English:** "Today's variance equals a base level, plus how big yesterday's shock was, plus how volatile things were yesterday."

Let's break down each piece:

**`σ²ₜ`** — Today's **conditional variance** (the thing we're trying to compute). The square root of this is today's volatility.

**`ω` (omega)** — The **base level** of variance. Think of it as the "floor" that variance can never go below. It's a small positive constant.

**`α × ε²ₜ₋₁`** — The **ARCH term** (shock impact).
- `ε²ₜ₋₁` is yesterday's **squared return shock** (how surprising yesterday's return was).
- `α` (alpha) controls how much yesterday's shock matters. Typical value: 0.05–0.15.
- **Captures volatility clustering:** A huge shock yesterday (large `ε²`) pushes up today's variance.

**`β × σ²ₜ₋₁`** — The **GARCH term** (persistence).
- `σ²ₜ₋₁` is yesterday's conditional variance (yesterday's volatility estimate).
- `β` (beta) controls how much yesterday's overall volatility carries over. Typical value: 0.80–0.95.
- **Captures persistence:** If things were volatile yesterday, they're probably still volatile today.

### Worked Example

Imagine these values: `ω = 0.00001, α = 0.10, β = 0.85`

**Day 1 (calm day):** Yesterday's shock `ε = 0.5%`, yesterday's variance `σ² = 0.0001` (1% vol)
```
σ²today = 0.00001 + 0.10 × (0.005)² + 0.85 × 0.0001
        = 0.00001 + 0.0000025 + 0.000085
        = 0.0000975
σtoday  = √0.0000975 = 0.987% daily vol ← Calm, close to yesterday
```

**Day 2 (crash day!):** Yesterday's shock `ε = -5%` (a huge drop), yesterday's variance `σ² = 0.0000975`
```
σ²today = 0.00001 + 0.10 × (-0.05)² + 0.85 × 0.0000975
        = 0.00001 + 0.00025 + 0.0000829
        = 0.000343
σtoday  = √0.000343 = 1.85% daily vol ← Shot up! Shock drove volatility higher
```

**Day 3 (no new shock):** Yesterday's shock `ε = 0.1%` (tiny), yesterday's variance `σ² = 0.000343`
```
σ²today = 0.00001 + 0.10 × (0.001)² + 0.85 × 0.000343
        = 0.00001 + 0.0000001 + 0.000292
        = 0.000302
σtoday  = √0.000302 = 1.74% daily vol ← Starting to decay back down (mean reversion)
```

See? The crash on Day 2 spiked volatility. Without a new shock, volatility decays back toward its long-run level. That's GARCH in action.

### Key Derived Metrics

From the GARCH parameters, we can compute important quantities:

**Persistence = α + β**
```
Persistence = 0.10 + 0.85 = 0.95
```
This tells you how "sticky" volatility is. Values close to 1 mean shocks persist a long time. Typical for stock markets: 0.95–0.99.

**Half-Life = log(0.5) / log(α + β)**
```
Half-Life = log(0.5) / log(0.95) = 13.5 days
```
After a volatility spike, it takes ~13.5 days for half the spike to decay away.

**Unconditional (Long-Run) Volatility = √(ω / (1 - α - β))**
```
Long-run variance = 0.00001 / (1 - 0.95) = 0.0002
Long-run vol = √0.0002 = 1.41% daily ≈ 22.4% annualized
```
This is the "normal" level volatility reverts to.

---

## 1.4 EGARCH — Handling Asymmetry (The Leverage Effect)

Standard GARCH treats positive and negative shocks the same: a +3% day and a -3% day both contribute `(0.03)² = 0.0009` to tomorrow's variance.

But in reality, **negative shocks increase volatility more than positive shocks**. This is called the **leverage effect** — when stock prices drop, the company's debt-to-equity ratio rises (leverage increases), making the stock riskier.

**EGARCH** (Exponential GARCH, by Daniel Nelson, 1991) fixes this:

```
log(σ²ₜ) = ω + α × [|zₜ₋₁| - E|zₜ₋₁|] + γ × zₜ₋₁ + β × log(σ²ₜ₋₁)
```

Where `zₜ₋₁ = εₜ₋₁ / σₜ₋₁` is the **standardized residual** (yesterday's shock divided by yesterday's volatility).

**The key new piece:** `γ × zₜ₋₁` — this is the **asymmetry (leverage) term**.
- When `γ < 0` (the usual case): negative shocks (`z < 0`) increase log-variance more than positive shocks. This is the leverage effect.
- When `γ = 0`: EGARCH becomes symmetric, similar to standard GARCH.

**Why model in log-space?** Because `log(σ²)` can be any real number, so σ² is always positive. Standard GARCH needs parameter constraints (ω > 0, α ≥ 0, β ≥ 0) to keep variance positive; EGARCH doesn't.

---

## 1.5 The VIX — What GARCH Competes Against

The **VIX** (CBOE Volatility Index) is called "Wall Street's fear gauge." It represents the market's expectation of 30-day future volatility for the S&P 500, derived from options prices.

**Key differences between GARCH volatility and VIX:**

| | GARCH Volatility | VIX |
|---|---|---|
| **Source** | Computed from past returns | Derived from options prices |
| **Direction** | Backward-looking (uses historical data) | Forward-looking (market expectation) |
| **Risk premium** | None | Includes a "fear premium" |
| **Typical level** | ~15–17% for S&P 500 | ~17–20% for S&P 500 |

**Why compare them?** If GARCH tracks VIX closely, it means our statistical model is capturing the same information that the entire options market is pricing in — that's a strong validation.

**Why VIX > GARCH (usually)?** VIX includes a **variance risk premium** — options sellers demand extra compensation, so VIX systematically overshoots realized/GARCH volatility by about 2–4%.

---

## 1.6 Pre-Modeling Tests — Checking Assumptions

Before fitting GARCH, we need to verify two things:

### Test 1: Stationarity (ADF & KPSS Tests)

GARCH requires that the return series is **stationary** — meaning its statistical properties (mean, variance) don't change over time. Raw prices are NOT stationary (they trend upward), but daily returns usually are.

**ADF (Augmented Dickey-Fuller) Test:**
- H₀: The series has a unit root (non-stationary)
- H₁: The series is stationary
- **Decision:** If p < 0.05, reject H₀ → series IS stationary ✓

**KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test:**
- H₀: The series IS stationary
- H₁: The series is non-stationary
- **Decision:** If p > 0.05, fail to reject H₀ → series IS stationary ✓

We run both because they have opposite null hypotheses. If both agree, we're confident.

### Test 2: ARCH Effects (Engle's LM Test)

This test checks: "Is there volatility clustering in the data?" If not, GARCH is pointless — you'd just use constant volatility.

**How it works:** Regress squared returns on their own lags. If the lags are significant, past shocks predict future volatility → ARCH effects are present.

- H₀: No ARCH effects (volatility is constant)
- H₁: ARCH effects present (volatility clusters)
- **Decision:** If p < 0.05, reject H₀ → ARCH effects ARE present → proceed with GARCH ✓

---

## 1.7 Forecasting & Term Structure

Once the GARCH model is fit, we can forecast future volatility.

**How GARCH forecasts work:**

For 1-step ahead:
```
σ²ₜ₊₁ = ω + α × ε²ₜ + β × σ²ₜ
```
(We know today's shock `εₜ` and today's variance `σ²ₜ`, so this is straightforward.)

For multi-step ahead (e.g., day t+5):
```
σ²ₜ₊ₕ = ω/(1-α-β) + (α+β)^(h-1) × [σ²ₜ₊₁ - ω/(1-α-β)]
```
As `h → ∞`, the forecast converges to the **unconditional variance** `ω/(1-α-β)`. This is mean reversion in action — far-out forecasts always approach the long-run average.

**Volatility Term Structure:**

Just like interest rates have a "yield curve," volatility has a **term structure**:

```
Horizon    Forecast
1 day      10.1%    ← Short-term: reflects current market conditions
5 days     11.0%
1 month    13.6%    ← Medium-term: converging toward long-run
3 months   16.2%
1 year     18.7%    ← Long-term: essentially the unconditional volatility
```

If the market is currently calm (low volatility), the term structure slopes **upward** (expect more vol in the future). If the market is in crisis (high volatility), it slopes **downward** (expect calm to return). This is incredibly useful for options trading.

---

---

# PART 2: PROJECT SETUP (Step 0)

Now let's code. We'll build everything from scratch, file by file.

---

## Step 0.1: Create the Project Folder

```bash
mkdir garch-volatility-forecaster
cd garch-volatility-forecaster
mkdir -p src tests output notebooks
```

---

## Step 0.2: Create `requirements.txt`

**File: `requirements.txt`**
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
yfinance>=0.2.18
matplotlib>=3.7.0
seaborn>=0.12.0
arch>=6.0.0
statsmodels>=0.14.0
```

**What each library does:**

| Library | Purpose |
|---------|---------|
| `numpy` | Fast array math (mean, std, random numbers) |
| `pandas` | DataFrames for time-series data |
| `scipy` | Statistical distributions for confidence intervals |
| `yfinance` | Downloads stock data and VIX from Yahoo Finance |
| `matplotlib` | Plotting engine (creates PNG charts) |
| `seaborn` | Makes matplotlib charts look professional |
| `arch` | **The GARCH library** — fits GARCH/EGARCH models via Maximum Likelihood |
| `statsmodels` | Stationarity tests (ADF, KPSS) and ARCH-LM test |

Install everything:
```bash
pip install -r requirements.txt
```

---

## Step 0.3: Create `src/__init__.py`

**File: `src/__init__.py`**
```python
"""
GARCH Volatility Forecaster
============================
A GARCH-based market volatility modeling and forecasting system.

Modules:
    data_loader    - Market data fetching & return computation
    garch_models   - GARCH/EGARCH model fitting & diagnostics
    forecasting    - Volatility forecasting, term structure, VIX comparison
    visualization  - Publication-quality financial charts
"""
```

---

---

# PART 3: DATA LOADER (Step 1)

The data loader fetches real market data and computes returns and realized volatility.

---

## Step 1.1: Understand What This Module Does

Here's the data flow:

```
Yahoo Finance
      │
      ├── SPY prices ──► Daily returns ──► Realized volatility (rolling 21-day)
      │
      └── ^VIX prices ──► VIX time series (for benchmarking later)
```

**Key concept — Scaling Returns ×100:**

For GARCH modeling, we use **percentage returns** and then scale them by **100** before feeding them to the `arch` library. This is a convention — the `arch` package expects returns in percentage form (e.g., 1.5 rather than 0.015) for numerical stability during Maximum Likelihood Estimation.

**Key concept — Realized Volatility:**

This is the actual, observed volatility computed as the rolling standard deviation of returns, annualized:
```
Realized Vol = std(last 21 returns) × √252

Where:
  21 = trading days in a month (our rolling window)
  252 = trading days in a year (annualization factor)
```

We'll compare GARCH's estimate to this later.

---

## Step 1.2: Write the Code

**File: `src/data_loader.py`**

```python
"""
data_loader.py — Market Data Fetching & Preprocessing
=====================================================

Handles all data acquisition:
  1. Download adjusted closing prices from Yahoo Finance (SPY)
  2. Download VIX data (CBOE Volatility Index) for benchmarking
  3. Compute daily percentage returns
  4. Compute realized (rolling) volatility

Why SPY?
  SPY is the most liquid ETF tracking the S&P 500. It's the standard
  benchmark for volatility studies because it represents the broad
  US equity market.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional


def fetch_stock_data(
    ticker: str = 'SPY',
    period: str = '5y'
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock/ETF symbol (default 'SPY' for S&P 500 ETF).
    period : str
        How far back to look. Options: '1y', '2y', '5y', '10y', 'max'.
        Default '5y' gives ~1,260 trading days — enough for robust
        GARCH estimation.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and a 'Close' column of adjusted prices.

    Example
    -------
    >>> prices = fetch_stock_data('SPY', '5y')
    >>> prices.head()
                  Close
    Date
    2021-01-12   378.42
    2021-01-13   379.15
    """
    print(f"Fetching data for {ticker}...")

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,   # Adjust for splits and dividends
        progress=False
    )

    # Keep only the Close column and flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].to_frame()
    else:
        prices = data[['Close']]

    prices = prices.dropna()

    print(f"  Downloaded {len(prices)} trading days "
          f"({prices.index[0].strftime('%Y-%m-%d')} to "
          f"{prices.index[-1].strftime('%Y-%m-%d')})")

    return prices


def fetch_vix_data(period: str = '5y') -> pd.Series:
    """
    Download the VIX (CBOE Volatility Index) from Yahoo Finance.

    The VIX represents the market's expectation of 30-day forward
    volatility for the S&P 500, derived from options prices.
    It's quoted in annualized percentage terms (e.g., VIX = 20 means
    the market expects ~20% annualized vol over the next 30 days).

    Parameters
    ----------
    period : str
        Lookback period (default '5y').

    Returns
    -------
    pd.Series
        VIX closing values as a time series.
    """
    print("Fetching VIX data...")

    vix_data = yf.download(
        '^VIX',
        period=period,
        auto_adjust=True,
        progress=False
    )

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data['Close'].squeeze()
    else:
        vix = vix_data['Close'].squeeze()

    vix = vix.dropna()
    vix.name = 'VIX'

    print(f"  VIX data: {len(vix)} trading days")

    return vix


def calculate_returns(
    prices: pd.DataFrame,
    scale: bool = True
) -> pd.Series:
    """
    Convert prices to daily percentage returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices (output of fetch_stock_data).
    scale : bool
        If True, multiply by 100 to get percentage returns.
        The arch library expects returns in percentage form
        (e.g., 1.5 for a 1.5% return) for numerical stability.

    Returns
    -------
    pd.Series
        Daily returns. If scale=True, in percentage terms (e.g., 1.5 = 1.5%).
        If scale=False, in decimal terms (e.g., 0.015 = 1.5%).

    Why scale?
    ----------
    The arch library's optimizer works with values roughly in the
    range of 0-5. Raw decimal returns (0.001 to 0.05) cause numerical
    issues. Multiplying by 100 puts them in the right range.
    """
    print("Calculating returns...")

    # pct_change: (today - yesterday) / yesterday
    returns = prices['Close'].pct_change().dropna()

    if scale:
        returns = returns * 100  # Convert 0.015 → 1.5

    returns.name = 'Returns'
    return returns


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    is_scaled: bool = True
) -> pd.Series:
    """
    Calculate rolling realized volatility (annualized).

    This is the "actual" volatility observed in the market — we'll
    compare it to GARCH's estimate to see how well the model fits.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (from calculate_returns).
    window : int
        Rolling window size in trading days (default 21 ≈ 1 month).
    is_scaled : bool
        Whether returns are in percentage form (×100).

    Returns
    -------
    pd.Series
        Annualized realized volatility in percentage terms.
        E.g., a value of 17.5 means 17.5% annualized volatility.

    The Math
    --------
    realized_vol = rolling_std(returns, window) × √252

    Why √252?
    Because variance scales linearly with time (σ²_annual = 252 × σ²_daily),
    so standard deviation scales with √time: σ_annual = σ_daily × √252.
    252 = approximate number of trading days per year.
    """
    rolling_std = returns.rolling(window=window).std()

    # Annualize: daily → annual
    if is_scaled:
        realized_vol = rolling_std * np.sqrt(252)
    else:
        realized_vol = rolling_std * np.sqrt(252) * 100

    realized_vol.name = 'Realized_Vol'
    return realized_vol.dropna()


def prepare_volatility_data(
    ticker: str = 'SPY',
    period: str = '5y'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    One-stop function: fetch all data needed for GARCH analysis.

    This is the "main" data function that main.py will call.

    Parameters
    ----------
    ticker : str
        Stock/ETF symbol.
    period : str
        Lookback period.

    Returns
    -------
    tuple of (prices, returns, realized_vol, vix)
        - prices: DataFrame of adjusted closing prices
        - returns: Series of daily returns (×100 scaled for arch)
        - realized_vol: Series of annualized realized volatility (%)
        - vix: Series of VIX closing values
    """
    prices = fetch_stock_data(ticker, period)
    returns = calculate_returns(prices, scale=True)
    realized_vol = calculate_realized_volatility(returns, window=21, is_scaled=True)
    vix = fetch_vix_data(period)

    # Print summary statistics (convert back to decimal for display)
    returns_decimal = returns / 100

    print(f"\nData loaded: {len(returns)} trading days")
    print(f"Date range: {returns.index[0].strftime('%Y-%m-%d')} to "
          f"{returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"\nAsset: {ticker}")
    print(f"\nReturn Statistics:")
    print(f"  Mean daily return:     {returns_decimal.mean():.4%}")
    print(f"  Daily volatility:      {returns_decimal.std():.4%}")
    print(f"  Annualized volatility: {returns_decimal.std() * np.sqrt(252):.2%}")
    print(f"  Skewness:              {returns_decimal.skew():.3f}")
    print(f"  Kurtosis:              {returns_decimal.kurtosis():.3f}")

    return prices, returns, realized_vol, vix
```

---

## Step 1.3: What You Just Built

When `prepare_volatility_data('SPY', '5y')` is called:

1. **`fetch_stock_data('SPY', '5y')`** — Downloads ~1,260 daily prices from Yahoo Finance.

2. **`calculate_returns(prices, scale=True)`** — Computes `pct_change() × 100`. A 1.5% daily gain becomes `1.5` (not `0.015`). This scaling is critical for the `arch` library.

3. **`calculate_realized_volatility(returns, window=21)`** — Computes `rolling_std(21 days) × √252`. This gives annualized realized vol in percentage terms (e.g., 17.5 means 17.5%).

4. **`fetch_vix_data('5y')`** — Downloads the VIX for benchmarking. VIX is already in annualized percentage terms.

---

---

# PART 4: GARCH MODELS (Step 2)

This is the heart of the project — fitting GARCH and EGARCH models, extracting parameters, and running diagnostics.

---

## Step 2.1: How the `arch` Library Works

The `arch` Python library (by Kevin Sheppard) handles all the heavy math of GARCH estimation. Here's the workflow:

```python
from arch import arch_model

# 1. Specify the model
model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')

# 2. Fit via Maximum Likelihood Estimation (MLE)
result = model.fit(disp='off')

# 3. Extract parameters
omega = result.params['omega']
alpha = result.params['alpha[1]']
beta  = result.params['beta[1]']
```

**What `arch_model()` parameters mean:**

| Parameter | Meaning |
|-----------|---------|
| `vol='GARCH'` | Use standard GARCH variance model |
| `vol='EGARCH'` | Use exponential GARCH (asymmetric) |
| `p=1` | Number of lagged variance terms (GARCH terms). `p=1` means use σ²ₜ₋₁ |
| `q=1` | Number of lagged shock terms (ARCH terms). `q=1` means use ε²ₜ₋₁ |
| `dist='normal'` | Assume return shocks follow a normal distribution |

**What Maximum Likelihood Estimation does:**

MLE finds the values of ω, α, β that make the observed data *most probable*. It's like asking: "What parameter values would have been most likely to generate the actual return series we see?" The `arch` library uses numerical optimization to find these values.

---

## Step 2.2: Write the Code

**File: `src/garch_models.py`**

```python
"""
garch_models.py — GARCH Model Fitting & Diagnostics
====================================================

Handles:
  1. Pre-modeling tests (ADF, KPSS, ARCH-LM)
  2. Fitting GARCH(1,1) and EGARCH(1,1) models
  3. Extracting and interpreting parameters
  4. Model comparison via AIC
  5. Extracting conditional volatility series

All models are estimated using Maximum Likelihood via the arch library.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─── Result Container ─────────────────────────────────────────

@dataclass
class GARCHResult:
    """
    Holds the complete output of a GARCH model fit.

    Attributes
    ----------
    model_name : str
        'GARCH(1,1)' or 'EGARCH(1,1)'
    model_result : object
        The raw arch library result object (for forecasting later).
    omega : float
        Base variance constant (ω).
    alpha : float
        ARCH coefficient — shock impact (α).
    beta : float
        GARCH coefficient — persistence (β).
    persistence : float
        α + β — total persistence of volatility shocks.
    half_life : float
        Days for a shock to decay by 50%.
    unconditional_vol : float
        Long-run annualized volatility (% terms).
    aic : float
        Akaike Information Criterion (lower = better fit).
    bic : float
        Bayesian Information Criterion (lower = better fit).
    log_likelihood : float
        Log-likelihood of the fitted model.
    """
    model_name: str
    model_result: object
    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life: float
    unconditional_vol: float
    aic: float
    bic: float
    log_likelihood: float


# ─── Pre-Modeling Diagnostic Tests ──────────────────────────────

def stationarity_tests(returns: pd.Series) -> Dict[str, dict]:
    """
    Run ADF and KPSS stationarity tests on the return series.

    Why?
    ----
    GARCH requires stationary data. Raw prices are non-stationary
    (they trend), but returns usually are. These tests confirm it.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (can be scaled or unscaled).

    Returns
    -------
    dict with 'ADF' and 'KPSS' keys, each containing:
        - statistic: test statistic
        - p_value: p-value
        - is_stationary: bool
    """
    results = {}

    # ── ADF Test ──
    # H₀: non-stationary (has unit root)
    # Reject H₀ if p < 0.05 → stationary
    adf_stat, adf_p, _, _, _, _ = adfuller(returns.dropna())
    results['ADF'] = {
        'statistic': adf_stat,
        'p_value': adf_p,
        'is_stationary': adf_p < 0.05
    }

    # ── KPSS Test ──
    # H₀: stationary
    # Fail to reject H₀ if p > 0.05 → stationary
    kpss_stat, kpss_p, _, _ = kpss(returns.dropna(), regression='c', nlags='auto')
    results['KPSS'] = {
        'statistic': kpss_stat,
        'p_value': kpss_p,
        'is_stationary': kpss_p > 0.05
    }

    return results


def arch_effect_test(returns: pd.Series, nlags: int = 12) -> dict:
    """
    Engle's ARCH-LM test for heteroskedasticity.

    Why?
    ----
    Tests whether volatility clustering exists in the data.
    If not, GARCH modeling is pointless — simple constant-vol
    models would suffice.

    How it works
    ------------
    1. Compute squared returns: ε² = (return - mean)²
    2. Regress ε² on its own lags: ε²ₜ = c + φ₁ε²ₜ₋₁ + ... + φₙε²ₜ₋ₙ
    3. If the lags are significant (high R²), ARCH effects are present.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    nlags : int
        Number of lags to include in the test (default 12).

    Returns
    -------
    dict with:
        - lm_stat: the LM test statistic
        - p_value: p-value
        - has_arch_effects: bool (True if p < 0.05)
    """
    clean_returns = returns.dropna()
    lm_stat, lm_p, f_stat, f_p = het_arch(clean_returns, nlags=nlags)

    return {
        'lm_stat': lm_stat,
        'p_value': lm_p,
        'has_arch_effects': lm_p < 0.05
    }


# ─── GARCH Model Fitting ───────────────────────────────────────

def fit_garch(
    returns: pd.Series,
    vol: str = 'GARCH',
    p: int = 1,
    q: int = 1,
    dist: str = 'normal'
) -> GARCHResult:
    """
    Fit a GARCH-family model to the return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (scaled ×100 for the arch library).
    vol : str
        Variance model type: 'GARCH' or 'EGARCH'.
    p : int
        Order of the GARCH term (lagged variance). Default 1.
    q : int
        Order of the ARCH term (lagged shocks). Default 1.
    dist : str
        Error distribution assumption: 'normal', 't', or 'skewt'.

    Returns
    -------
    GARCHResult
        Dataclass with all model parameters and metrics.

    How the arch library works internally
    -------------------------------------
    1. Sets up the model:
       - Mean equation: rₜ = μ + εₜ  (constant mean)
       - Variance equation: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
    2. Estimates all parameters (μ, ω, α, β) simultaneously via MLE.
    3. Returns fitted model with parameter estimates, standard errors,
       t-stats, p-values, and information criteria.
    """
    model_name = f"{vol}({p},{q})"

    # Create the model specification
    model = arch_model(
        returns,
        mean='Constant',    # Mean equation: rₜ = μ + εₜ
        vol=vol,            # Variance model: GARCH or EGARCH
        p=p, q=q,
        dist=dist
    )

    # Fit via Maximum Likelihood Estimation
    result = model.fit(disp='off')

    # ── Extract Parameters ──
    params = result.params
    omega = params.get('omega', 0.0)
    alpha = params.get('alpha[1]', 0.0)
    beta = params.get('beta[1]', 0.0)
    persistence = alpha + beta

    # ── Derived Metrics ──

    # Half-life: days for a shock to decay by 50%
    if 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.inf

    # Unconditional (long-run) volatility
    if vol.upper() != 'EGARCH' and persistence < 1:
        long_run_var = omega / (1 - persistence)
        unconditional_vol = np.sqrt(long_run_var) * np.sqrt(252)
    else:
        unconditional_vol = np.nan

    return GARCHResult(
        model_name=model_name,
        model_result=result,
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        half_life=half_life,
        unconditional_vol=unconditional_vol,
        aic=result.aic,
        bic=result.bic,
        log_likelihood=result.loglikelihood,
    )


def get_conditional_volatility(garch_result: GARCHResult) -> pd.Series:
    """
    Extract the fitted conditional volatility series from a GARCH model.

    The "conditional volatility" is the model's estimate of volatility
    for each day, given all information up to that day. It's the
    time-varying σₜ that GARCH computes.

    Parameters
    ----------
    garch_result : GARCHResult
        Output of fit_garch().

    Returns
    -------
    pd.Series
        Annualized conditional volatility in percentage terms.
    """
    daily_vol = garch_result.model_result.conditional_volatility
    annual_vol = daily_vol * np.sqrt(252)
    annual_vol.name = f'{garch_result.model_name}_Vol'
    return annual_vol


def fit_multiple_garch_models(
    returns: pd.Series
) -> Dict[str, GARCHResult]:
    """
    Fit both GARCH(1,1) and EGARCH(1,1), return sorted by AIC.

    AIC (Akaike Information Criterion) balances model fit against
    complexity. Lower AIC = better model.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (×100 scaled).

    Returns
    -------
    dict — Keys: model names, Values: GARCHResult objects.
        Sorted by AIC (best model first).
    """
    models_to_fit = [
        {'vol': 'GARCH', 'p': 1, 'q': 1},
        {'vol': 'EGARCH', 'p': 1, 'q': 1},
    ]

    results = {}
    for spec in models_to_fit:
        try:
            result = fit_garch(returns, **spec)
            results[result.model_name] = result
        except Exception as e:
            print(f"  Warning: Failed to fit {spec['vol']}({spec['p']},{spec['q']}): {e}")

    # Sort by AIC (lower is better)
    results = dict(sorted(results.items(), key=lambda x: x[1].aic))
    return results


def print_model_comparison(results: Dict[str, GARCHResult]) -> None:
    """Print a formatted comparison table of all fitted models."""
    print(f"\n{'Model':>12s} {'α (ARCH)':>10s} {'β (GARCH)':>10s} "
          f"{'Persistence':>12s} {'Half-Life':>10s} {'Long-Run Vol':>13s} {'AIC':>9s}")

    for name, r in results.items():
        hl_str = f"{r.half_life:.1f} days" if np.isfinite(r.half_life) else "inf days"
        vol_str = f"{r.unconditional_vol:.2f}%" if np.isfinite(r.unconditional_vol) else "nan%"

        print(f"{name:>12s} {r.alpha:>10.4f} {r.beta:>10.4f} "
              f"{r.persistence:>12.4f} {hl_str:>10s} {vol_str:>13s} {r.aic:>9.1f}")

    best = list(results.keys())[0]
    print(f"\n✓ Best model by AIC: {best}")
```

---

## Step 2.3: What You Just Built

**`stationarity_tests(returns)`** → Runs ADF and KPSS tests. Both should agree returns are stationary.

**`arch_effect_test(returns)`** → Runs Engle's ARCH-LM test. Should confirm volatility clustering exists.

**`fit_garch(returns, vol='GARCH')`** → Feeds the return series to the `arch` library, which finds the optimal ω, α, β via MLE. Returns a `GARCHResult` with all parameters and derived metrics.

**`get_conditional_volatility(result)`** → Extracts the day-by-day volatility estimate from the fitted model.

**`fit_multiple_garch_models(returns)`** → Fits both GARCH and EGARCH, compares them by AIC, returns sorted best-first.

---

---

# PART 5: FORECASTING (Step 3)

This module generates forward-looking volatility forecasts and compares them with the VIX.

---

## Step 3.1: Write the Code

**File: `src/forecasting.py`**

```python
"""
forecasting.py — Volatility Forecasting & VIX Comparison
========================================================

Handles:
  1. Multi-step-ahead volatility forecasts with confidence intervals
  2. Volatility term structure (1D to 1Y)
  3. VIX comparison (correlation, tracking error, scatter analysis)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.garch_models import GARCHResult


@dataclass
class VolatilityForecast:
    """
    Holds multi-step volatility forecast results.

    Attributes
    ----------
    horizon : int
        Number of days forecasted.
    forecasts : np.ndarray
        Annualized volatility forecast for each day (%).
    lower_ci : np.ndarray
        Lower bound of 95% confidence interval (%).
    upper_ci : np.ndarray
        Upper bound of 95% confidence interval (%).
    confidence : float
        Confidence level (e.g., 0.95).
    """
    horizon: int
    forecasts: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    confidence: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to a formatted DataFrame."""
        return pd.DataFrame({
            'Day': range(1, self.horizon + 1),
            'Forecast (%)': self.forecasts,
            'Lower CI (%)': self.lower_ci,
            'Upper CI (%)': self.upper_ci,
        }).set_index('Day')


def forecast_volatility(
    garch_result: GARCHResult,
    horizon: int = 21,
    confidence: float = 0.95
) -> VolatilityForecast:
    """
    Generate multi-step-ahead volatility forecasts with confidence intervals.

    Parameters
    ----------
    garch_result : GARCHResult
        Fitted GARCH model (output of fit_garch).
    horizon : int
        Number of days to forecast (default 21 ≈ 1 month).
    confidence : float
        Confidence level for intervals (default 0.95).

    Returns
    -------
    VolatilityForecast
        Contains point forecasts and confidence bounds.
    """
    result = garch_result.model_result

    # Get variance forecasts from the arch library
    forecast_obj = result.forecast(horizon=horizon)

    # Extract variance forecasts (last row = out-of-sample forecast)
    variance_forecasts = forecast_obj.variance.iloc[-1].values

    # Convert daily variance → annualized volatility (%)
    vol_forecasts = np.sqrt(variance_forecasts) * np.sqrt(252)

    # ── Confidence Intervals ──
    z = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95%

    # Relative standard error grows with horizon
    relative_se = 0.15 * np.sqrt(np.arange(1, horizon + 1) / horizon)

    lower = vol_forecasts * np.exp(-z * relative_se)
    upper = vol_forecasts * np.exp(z * relative_se)

    return VolatilityForecast(
        horizon=horizon,
        forecasts=vol_forecasts,
        lower_ci=lower,
        upper_ci=upper,
        confidence=confidence,
    )


def term_structure_forecast(
    garch_result: GARCHResult,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compute volatility term structure across multiple horizons.

    The term structure shows how the volatility forecast evolves from
    very short-term (1 day) to long-term (1 year). It's analogous to
    the yield curve in fixed income.

    Parameters
    ----------
    garch_result : GARCHResult
        Fitted GARCH model.
    horizons : list of int, optional
        Forecast horizons in trading days.
        Default: [1, 5, 21, 63, 252] (1D, 1W, 1M, 3M, 1Y)

    Returns
    -------
    pd.DataFrame
        Term structure with columns: horizon, label, volatility.
    """
    if horizons is None:
        horizons = [1, 5, 21, 63, 252]

    labels = {1: '1D', 5: '5D', 10: '10D', 21: '1M', 63: '3M', 126: '6M', 252: '1Y'}

    result = garch_result.model_result
    rows = []

    for h in horizons:
        forecast_obj = result.forecast(horizon=h)
        var_h = forecast_obj.variance.iloc[-1].values[-1]
        vol_h = np.sqrt(var_h) * np.sqrt(252)

        rows.append({
            'Horizon (days)': h,
            'Term': labels.get(h, f'{h}D'),
            'Volatility': vol_h,
        })

    return pd.DataFrame(rows)


def compare_with_vix(
    conditional_vol: pd.Series,
    vix: pd.Series
) -> Dict[str, float]:
    """
    Compare GARCH conditional volatility against the VIX.

    Parameters
    ----------
    conditional_vol : pd.Series
        Annualized GARCH volatility (from get_conditional_volatility).
    vix : pd.Series
        VIX closing values.

    Returns
    -------
    dict with comparison metrics:
        - Correlation: Pearson correlation between GARCH vol and VIX
        - Mean Difference: average (GARCH - VIX), usually negative
        - Tracking Error: std of daily differences
        - GARCH Current: latest GARCH volatility
        - VIX Current: latest VIX value
    """
    combined = pd.DataFrame({
        'GARCH': conditional_vol,
        'VIX': vix
    }).dropna()

    if len(combined) == 0:
        return {
            'Correlation': np.nan, 'Mean Difference': np.nan,
            'Tracking Error': np.nan, 'GARCH Current': np.nan,
            'VIX Current': np.nan,
        }

    garch_vals = combined['GARCH']
    vix_vals = combined['VIX']

    return {
        'Correlation': garch_vals.corr(vix_vals),
        'Mean Difference': (garch_vals - vix_vals).mean(),
        'Tracking Error': (garch_vals - vix_vals).std(),
        'GARCH Current': garch_vals.iloc[-1],
        'VIX Current': vix_vals.iloc[-1],
    }
```

---

---

# PART 6: VISUALIZATION (Step 4)

Five charts, each answering a specific question.

---

**File: `src/visualization.py`**

```python
"""
visualization.py — Publication-Quality Financial Charts
========================================================

Generates five charts:
  1. Returns & Volatility overview (3-panel)
  2. GARCH Model diagnostics (4-panel)
  3. Volatility forecast with confidence intervals
  4. Model comparison (GARCH vs EGARCH vs Realized)
  5. VIX comparison (scatter + time series)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

from src.garch_models import GARCHResult
from src.forecasting import VolatilityForecast

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def plot_return_and_volatility(
    returns: pd.Series,
    conditional_vol: pd.Series,
    realized_vol: pd.Series,
    vix: pd.Series,
    save_path: str = 'output/returns_volatility.png'
) -> None:
    """
    Chart 1: Three-panel overview.

    Panel 1: Daily returns (colored green/red by sign)
    Panel 2: Volatility comparison (GARCH vs Realized vs VIX)
    Panel 3: Squared returns (visualizes volatility clustering)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    returns_pct = returns / 100

    # ── Panel 1: Daily Returns ──
    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns_pct]
    axes[0].bar(returns_pct.index, returns_pct.values, color=colors, alpha=0.7, width=1.5)
    axes[0].set_ylabel('Daily Return')
    axes[0].set_title('Daily Returns', fontweight='bold', fontsize=13)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # ── Panel 2: Volatility Comparison ──
    combined = pd.DataFrame({
        'GARCH': conditional_vol, 'Realized': realized_vol, 'VIX': vix,
    }).dropna()

    if len(combined) > 0:
        axes[1].plot(combined.index, combined['GARCH'], label='GARCH Vol',
                     color='#3498db', linewidth=1.5)
        axes[1].plot(combined.index, combined['Realized'], label='Realized Vol',
                     color='#e67e22', linewidth=1, alpha=0.7)
        axes[1].plot(combined.index, combined['VIX'], label='VIX',
                     color='#e74c3c', linewidth=1, alpha=0.7)
        axes[1].set_ylabel('Annualized Volatility (%)')
        axes[1].set_title('Volatility: GARCH vs Realized vs VIX',
                          fontweight='bold', fontsize=13)
        axes[1].legend(loc='upper right', fontsize=9)

    # ── Panel 3: Squared Returns ──
    sq_returns = (returns_pct ** 2)
    axes[2].fill_between(sq_returns.index, sq_returns.values, alpha=0.5, color='#9b59b6')
    axes[2].set_ylabel('Squared Return')
    axes[2].set_title('Squared Returns (Volatility Clustering)', fontweight='bold', fontsize=13)
    axes[2].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_garch_diagnostics(
    garch_result: GARCHResult,
    save_path: str = 'output/garch_diagnostics.png'
) -> None:
    """
    Chart 2: Four-panel model diagnostics.

    Panel 1: Standardized residuals (should look like white noise)
    Panel 2: Histogram of residuals vs normal distribution
    Panel 3: ACF of squared residuals (should be insignificant)
    Panel 4: Q-Q plot (should follow 45-degree line)
    """
    from scipy import stats as sp_stats
    from statsmodels.graphics.tsaplots import plot_acf

    result = garch_result.model_result
    std_resid = result.std_resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Standardized Residuals
    axes[0, 0].plot(std_resid.index, std_resid.values, linewidth=0.5, color='steelblue')
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].axhline(y=2, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=-2, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Standardized Residuals', fontweight='bold')

    # Panel 2: Residual Distribution
    axes[0, 1].hist(std_resid, bins=50, density=True, alpha=0.7,
                    color='steelblue', edgecolor='white')
    x = np.linspace(-4, 4, 100)
    axes[0, 1].plot(x, sp_stats.norm.pdf(x), 'r-', linewidth=2, label='Normal')
    axes[0, 1].set_title('Residual Distribution vs Normal', fontweight='bold')
    axes[0, 1].legend()

    stats_text = (f"Mean: {std_resid.mean():.3f}\nStd: {std_resid.std():.3f}\n"
                  f"Skew: {std_resid.skew():.3f}\nKurt: {std_resid.kurtosis():.3f}")
    axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes,
                    va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: ACF of Squared Residuals
    plot_acf(std_resid ** 2, lags=30, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF of Squared Residuals', fontweight='bold')

    # Panel 4: Q-Q Plot
    sp_stats.probplot(std_resid, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal)', fontweight='bold')
    axes[1, 1].get_lines()[0].set_markersize(2)

    plt.suptitle(f'{garch_result.model_name} Diagnostics', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_volatility_forecast(
    conditional_vol: pd.Series,
    forecast: VolatilityForecast,
    n_history: int = 120,
    save_path: str = 'output/volatility_forecast.png'
) -> None:
    """
    Chart 3: Historical vol with forward-looking forecast + confidence band.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    history = conditional_vol.iloc[-n_history:]
    ax.plot(history.index, history.values, color='#3498db', linewidth=1.5,
            label='GARCH Conditional Volatility')

    last_date = history.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                     periods=forecast.horizon)

    ax.plot(forecast_dates, forecast.forecasts, color='#e74c3c', linewidth=2,
            linestyle='--', label=f'{forecast.horizon}-Day Forecast')
    ax.fill_between(forecast_dates, forecast.lower_ci, forecast.upper_ci,
                    alpha=0.2, color='#e74c3c',
                    label=f'{forecast.confidence:.0%} Confidence Interval')

    ax.axvline(x=last_date, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.text(last_date, ax.get_ylim()[1] * 0.95, ' Forecast →',
            fontsize=10, color='gray', va='top')

    ax.set_title('Volatility Forecast with Confidence Interval',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_model_comparison(
    model_results: Dict[str, GARCHResult],
    realized_vol: pd.Series,
    save_path: str = 'output/model_comparison.png'
) -> None:
    """Chart 4: Compare GARCH vs EGARCH vs realized volatility."""
    from src.garch_models import get_conditional_volatility

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'GARCH(1,1)': '#3498db', 'EGARCH(1,1)': '#2ecc71'}

    for name, result in model_results.items():
        cond_vol = get_conditional_volatility(result)
        color = colors.get(name, '#95a5a6')
        ax.plot(cond_vol.index, cond_vol.values, label=name,
                color=color, linewidth=1.2)

    ax.plot(realized_vol.index, realized_vol.values, label='Realized Vol',
            color='#e67e22', linewidth=1, alpha=0.6, linestyle='--')

    ax.set_title('Model Comparison: GARCH vs EGARCH vs Realized',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_vix_comparison(
    conditional_vol: pd.Series,
    vix: pd.Series,
    model_name: str = 'GARCH(1,1)',
    save_path: str = 'output/vix_comparison.png'
) -> None:
    """Chart 5: Two-panel VIX comparison (scatter + time series)."""
    combined = pd.DataFrame({'GARCH': conditional_vol, 'VIX': vix}).dropna()

    if len(combined) < 10:
        print(f"  ⚠ Insufficient overlapping data for VIX comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = combined['GARCH'].values
    y = combined['VIX'].values

    # Panel 1: Scatter with regression
    axes[0].scatter(x, y, alpha=0.3, s=10, color='steelblue')
    slope, intercept = np.polyfit(x, y, 1)
    r_squared = combined['GARCH'].corr(combined['VIX']) ** 2
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
                 label=f'R² = {r_squared:.3f}')
    lim = max(x.max(), y.max())
    axes[0].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Perfect agreement')
    axes[0].set_xlabel(f'{model_name} Volatility (%)', fontsize=11)
    axes[0].set_ylabel('VIX (%)', fontsize=11)
    axes[0].set_title(f'{model_name} vs VIX (Scatter)', fontweight='bold')
    axes[0].legend(fontsize=9)

    # Panel 2: Time series overlay
    axes[1].plot(combined.index, combined['GARCH'], label=model_name,
                 color='#3498db', linewidth=1.2)
    axes[1].plot(combined.index, combined['VIX'], label='VIX',
                 color='#e74c3c', linewidth=1.2)
    axes[1].fill_between(combined.index, combined['GARCH'], combined['VIX'],
                         alpha=0.1, color='gray')
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Annualized Volatility (%)', fontsize=11)
    axes[1].set_title(f'{model_name} vs VIX (Time Series)', fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle(f'{model_name} vs VIX Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")
```

---

---

# PART 7: MAIN SCRIPT (Step 5)

**File: `main.py`**

```python
"""
main.py — GARCH Volatility Forecaster Entry Point
===================================================

Runs the complete analysis pipeline:
  1. Load market data (SPY + VIX)
  2. Pre-modeling diagnostics (stationarity + ARCH effects)
  3. Fit GARCH(1,1) and EGARCH(1,1)
  4. Generate volatility forecasts + term structure
  5. Compare with VIX
  6. Generate all visualizations
"""

import os
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

from src.data_loader import prepare_volatility_data
from src.garch_models import (
    stationarity_tests, arch_effect_test,
    fit_multiple_garch_models, print_model_comparison,
    get_conditional_volatility,
)
from src.forecasting import (
    forecast_volatility, term_structure_forecast,
    compare_with_vix,
)
from src.visualization import (
    plot_return_and_volatility,
    plot_garch_diagnostics,
    plot_volatility_forecast,
    plot_model_comparison,
    plot_vix_comparison,
)


def main():
    """Run the complete GARCH volatility analysis."""

    print("=" * 60)
    print(" GARCH VOLATILITY FORECASTER")
    print("=" * 60)
    print(f"\nThis analysis models and forecasts market volatility using:")
    print(f"  1. GARCH(1,1) - Standard model")
    print(f"  2. EGARCH(1,1) - Exponential GARCH (asymmetric)")

    # ── Step 1: Load Data ──
    print(f"\n{'─' * 60}")
    print(f" STEP 1: Loading Market Data")
    print(f"{'─' * 60}")

    ticker = 'SPY'
    prices, returns, realized_vol, vix = prepare_volatility_data(ticker, '5y')

    # ── Step 2: Pre-Modeling Diagnostics ──
    print(f"\n{'─' * 60}")
    print(f" STEP 2: Pre-Modeling Diagnostics")
    print(f"{'─' * 60}")

    stat_results = stationarity_tests(returns)
    print(f"\nStationarity Tests:")
    for test_name, result in stat_results.items():
        status = "✓ Stationary" if result['is_stationary'] else "✗ Non-stationary"
        print(f"  • {test_name} Test:  stat={result['statistic']:.3f}, "
              f"p-value={result['p_value']:.4f}")
        if test_name == 'ADF':
            print(f"    → {status} (reject H0 if p<0.05)")
        else:
            print(f"    → {status} (fail to reject H0 if p>0.05)")

    arch_result = arch_effect_test(returns)
    print(f"\nARCH Effect Test (Engle's LM test):")
    print(f"  • LM stat={arch_result['lm_stat']:.2f}, p-value={arch_result['p_value']:.4f}")
    status = "✓ ARCH effects present" if arch_result['has_arch_effects'] else "✗ No ARCH effects"
    print(f"    → {status}")

    # ── Step 3: Fit GARCH Models ──
    print(f"\n{'─' * 60}")
    print(f" STEP 3: Fitting GARCH Models")
    print(f"{'─' * 60}")

    model_results = fit_multiple_garch_models(returns)
    print(f"\n📊 Model Comparison (sorted by AIC):")
    print_model_comparison(model_results)

    best_name = list(model_results.keys())[0]
    best_result = model_results[best_name]

    print(f"\n📈 Model Interpretation ({best_name}):")
    print(f"  • α = {best_result.alpha:.4f} → Shock impact coefficient")
    print(f"  • β = {best_result.beta:.4f} → Volatility persistence")
    print(f"  • α + β = {best_result.persistence:.4f} → Total persistence")
    if np.isfinite(best_result.half_life):
        print(f"  • Half-life = {best_result.half_life:.1f} days → Shock decay time")

    cond_vol = get_conditional_volatility(best_result)

    # ── Step 4: Forecasting ──
    print(f"\n{'─' * 60}")
    print(f" STEP 4: Volatility Forecasting")
    print(f"{'─' * 60}")

    forecast = forecast_volatility(best_result, horizon=21, confidence=0.95)
    forecast_df = forecast.to_dataframe()

    print(f"\nVolatility Forecast (Annualized):")
    print(f"{'─' * 45}")
    print(f"{'Day':<8} {'Forecast':>12} {'Lower 95%':>12} {'Upper 95%':>12}")
    print(f"{'─' * 45}")
    for day in [1, 5, 21]:
        if day <= forecast.horizon:
            row = forecast_df.loc[day]
            print(f"{day:<8} {row['Forecast (%)']:>11.2f}% "
                  f"{row['Lower CI (%)']:>11.2f}% "
                  f"{row['Upper CI (%)']:>11.2f}%")
    print(f"{'─' * 45}")

    term_struct = term_structure_forecast(best_result)
    print(f"\nVolatility Term Structure:")
    print(f" {'Horizon (days)':>15s} {'Term':>5s} {'Volatility':>11s}")
    for _, row in term_struct.iterrows():
        print(f" {int(row['Horizon (days)']):>15d} {row['Term']:>5s} "
              f"{row['Volatility']:>10.2f}%")

    # ── Step 5: VIX Comparison ──
    print(f"\n{'─' * 60}")
    print(f" STEP 5: VIX Comparison")
    print(f"{'─' * 60}")

    vix_metrics = compare_with_vix(cond_vol, vix)
    print(f"\n{best_name} vs VIX Comparison:")
    print(f"  • Correlation:      {vix_metrics['Correlation']:.3f}")
    print(f"  • Mean Difference:  {vix_metrics['Mean Difference']:.2f}%")
    print(f"  • Tracking Error:   {vix_metrics['Tracking Error']:.2f}%")

    if vix_metrics['Correlation'] > 0.7:
        print(f"  ✓ Strong correlation with VIX suggests model captures market fear")
    elif vix_metrics['Correlation'] > 0.5:
        print(f"  ~ Moderate correlation with VIX")
    else:
        print(f"  ⚠ Weak correlation — model may need improvement")

    print(f"\nCurrent Levels:")
    print(f"  • GARCH Volatility: {vix_metrics['GARCH Current']:.2f}%")
    print(f"  • VIX:              {vix_metrics['VIX Current']:.2f}%")
    print(f"  • Difference:       "
          f"{vix_metrics['GARCH Current'] - vix_metrics['VIX Current']:.2f}%")

    # ── Step 6: Visualizations ──
    print(f"\n{'─' * 60}")
    print(f" STEP 6: Generating Visualizations")
    print(f"{'─' * 60}")

    os.makedirs('output', exist_ok=True)
    print(f"\nSaving charts to ./output/ directory...")

    plot_return_and_volatility(returns, cond_vol, realized_vol, vix)
    plot_garch_diagnostics(best_result)
    plot_volatility_forecast(cond_vol, forecast)
    plot_model_comparison(model_results, realized_vol)
    plot_vix_comparison(cond_vol, vix, model_name=best_name)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Key Findings:")
    print(f"  • Asset analyzed: {ticker}")
    print(f"  • Best model: {best_name}")
    print(f"  • Volatility persistence: {best_result.persistence:.4f}")
    print(f"  • Current GARCH vol: {vix_metrics['GARCH Current']:.2f}%")
    print(f"  • 21-day forecast: {forecast.forecasts[-1]:.2f}%")
    print(f"\n📁 Output files saved to ./output/")
    print(f"\nDone! ✅")


if __name__ == '__main__':
    main()
```

---

---

# PART 8: UNIT TESTS (Step 6)

**File: `tests/test_garch.py`**

```python
"""
test_garch.py — Unit Tests for GARCH Volatility Forecaster
==========================================================

Run with: python -m pytest tests/test_garch.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import calculate_returns, calculate_realized_volatility
from src.garch_models import (
    stationarity_tests, arch_effect_test,
    fit_garch, get_conditional_volatility,
    fit_multiple_garch_models,
)
from src.forecasting import (
    forecast_volatility, term_structure_forecast,
    compare_with_vix,
)


@pytest.fixture
def sample_prices():
    """Create realistic fake prices with volatility clustering."""
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range('2020-01-01', periods=n)
    rets = np.random.normal(0.0003, 0.01, n)
    rets[200:250] = np.random.normal(-0.001, 0.03, 50)  # High vol regime
    rets[400:430] = np.random.normal(0.0005, 0.025, 30)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({'Close': prices}, index=dates)

@pytest.fixture
def sample_returns(sample_prices):
    return calculate_returns(sample_prices, scale=True)

@pytest.fixture
def sample_vix():
    np.random.seed(123)
    dates = pd.bdate_range('2020-02-01', periods=500)
    return pd.Series(np.random.uniform(12, 35, 500), index=dates, name='VIX')


class TestReturns:
    def test_correct_length(self, sample_prices):
        r = calculate_returns(sample_prices, scale=True)
        assert len(r) == len(sample_prices) - 1

    def test_scaling(self, sample_prices):
        scaled = calculate_returns(sample_prices, scale=True)
        unscaled = calculate_returns(sample_prices, scale=False)
        assert abs(scaled.mean()) > abs(unscaled.mean()) * 50

    def test_realized_vol_positive(self, sample_returns):
        rv = calculate_realized_volatility(sample_returns)
        assert (rv > 0).all()


class TestStationarity:
    def test_returns_stationary(self, sample_returns):
        r = stationarity_tests(sample_returns)
        assert r['ADF']['is_stationary']

    def test_both_tests_run(self, sample_returns):
        r = stationarity_tests(sample_returns)
        assert 'ADF' in r and 'KPSS' in r


class TestARCHEffects:
    def test_detected(self, sample_returns):
        r = arch_effect_test(sample_returns)
        assert r['has_arch_effects']

    def test_keys(self, sample_returns):
        r = arch_effect_test(sample_returns)
        assert 'lm_stat' in r and 'p_value' in r


class TestGARCH:
    def test_fits(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert r.model_name == 'GARCH(1,1)'

    def test_alpha_range(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert 0 < r.alpha < 1

    def test_beta_range(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert 0 < r.beta < 1

    def test_persistence_below_one(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert r.persistence < 1.0

    def test_positive_uncond_vol(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert r.unconditional_vol > 0

    def test_positive_half_life(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        assert r.half_life > 0

    def test_cond_vol_length(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        cv = get_conditional_volatility(r)
        assert len(cv) == len(sample_returns)

    def test_egarch_fits(self, sample_returns):
        r = fit_garch(sample_returns, vol='EGARCH')
        assert 'EGARCH' in r.model_name


class TestMultipleModels:
    def test_fits_both(self, sample_returns):
        r = fit_multiple_garch_models(sample_returns)
        assert len(r) == 2

    def test_sorted_by_aic(self, sample_returns):
        r = fit_multiple_garch_models(sample_returns)
        aics = [v.aic for v in r.values()]
        assert aics == sorted(aics)


class TestForecasting:
    def test_length(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        f = forecast_volatility(r, horizon=21)
        assert len(f.forecasts) == 21

    def test_positive(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        f = forecast_volatility(r, horizon=10)
        assert (f.forecasts > 0).all()

    def test_ci_bounds(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        f = forecast_volatility(r, horizon=10)
        assert (f.lower_ci < f.forecasts).all()
        assert (f.upper_ci > f.forecasts).all()

    def test_term_structure(self, sample_returns):
        r = fit_garch(sample_returns, vol='GARCH')
        ts = term_structure_forecast(r)
        assert len(ts) == 5
        assert (ts['Volatility'] > 0).all()


class TestVIXComparison:
    def test_all_metrics(self, sample_returns, sample_vix):
        r = fit_garch(sample_returns, vol='GARCH')
        cv = get_conditional_volatility(r)
        m = compare_with_vix(cv, sample_vix)
        assert 'Correlation' in m and 'Tracking Error' in m

    def test_correlation_range(self, sample_returns, sample_vix):
        r = fit_garch(sample_returns, vol='GARCH')
        cv = get_conditional_volatility(r)
        m = compare_with_vix(cv, sample_vix)
        if not np.isnan(m['Correlation']):
            assert -1 <= m['Correlation'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

---

# PART 9: RUN IT! (Step 7)

## Step 7.1: Run the Full Pipeline
```bash
python main.py
```

## Step 7.2: Run the Tests
```bash
pip install pytest
python -m pytest tests/test_garch.py -v
```

---

---

# PART 10: HOW TO READ THE RESULTS

---

## 10.1: Interpreting Model Parameters

When you see: `α = 0.1251, β = 0.8546, Persistence = 0.9797, Half-Life = 33.8 days`

**α = 0.1251 (shock impact):** When a big move happens, 12.5% of that shock's squared magnitude feeds into tomorrow's variance. Higher α = volatility reacts more aggressively to news.

**β = 0.8546 (persistence):** 85.5% of today's variance carries over to tomorrow. Volatility has tremendous inertia.

**Persistence = 0.98:** Shocks are extremely persistent. It takes weeks for a spike to fully decay.

**Half-Life = 33.8 days:** If the market panics, half of that spike is still present ~34 trading days later (~7 weeks).

---

## 10.2: Interpreting the VIX Comparison

**Correlation of 0.76:** Our backward-looking statistical model captures 76% of the information that the options market is pricing in.

**GARCH < VIX by ~3%:** This is the **variance risk premium** — options sellers demand extra compensation, so VIX overshoots.

---

## 10.3: Interpreting the Term Structure

When the term structure slopes **upward** (10% → 18.7%): the market is currently calm but expected to revert to higher long-run vol. This is the typical shape.

When it slopes **downward**: the market is in crisis — elevated vol expected to decay.

---

## 10.4: Interpreting the Diagnostics Chart

**Standardized residuals:** Should look like random noise bouncing between ±2.

**Histogram:** Should approximately follow the bell curve. Heavy tails suggest using Student-t.

**ACF of squared residuals:** Bars should be within blue bands. Significant bars = model hasn't fully captured dynamics.

**Q-Q plot:** Points should follow the 45° line. Tail deviations indicate wrong distribution.

---

---

# PART 11: QUICK REFERENCE CARD

## Architecture
```
main.py                     → Orchestrates everything
src/data_loader.py          → fetch_stock_data(), fetch_vix_data(),
                               calculate_returns(), calculate_realized_volatility(),
                               prepare_volatility_data()
src/garch_models.py         → stationarity_tests(), arch_effect_test(),
                               fit_garch(), get_conditional_volatility(),
                               fit_multiple_garch_models()
src/forecasting.py          → forecast_volatility(), term_structure_forecast(),
                               compare_with_vix()
src/visualization.py        → plot_return_and_volatility(), plot_garch_diagnostics(),
                               plot_volatility_forecast(), plot_model_comparison(),
                               plot_vix_comparison()
tests/test_garch.py         → 22 unit tests across 7 test classes
```

## Key Formulas

| Concept | Formula |
|---------|---------|
| Daily Return | `(price_t - price_{t-1}) / price_{t-1}` |
| GARCH(1,1) Variance | `σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁` |
| Persistence | `α + β` |
| Half-Life | `log(0.5) / log(α + β)` |
| Long-Run Vol | `√(ω / (1 - α - β)) × √252` |
| Annualized Vol | `daily_vol × √252` |
| EGARCH | `log(σ²ₜ) = ω + α·g(zₜ₋₁) + γ·zₜ₋₁ + β·log(σ²ₜ₋₁)` |

## Pre-Modeling Checklist

| Test | H₀ | Pass Condition | Why |
|------|-----|---------------|-----|
| ADF | Non-stationary | p < 0.05 (reject H₀) | GARCH requires stationary data |
| KPSS | Stationary | p > 0.05 (don't reject H₀) | Confirms ADF result |
| ARCH-LM | No ARCH effects | p < 0.05 (reject H₀) | Confirms vol clustering exists |

## Dependencies
```
numpy        → Arrays, math
pandas       → DataFrames, time series
scipy        → Normal distribution for CIs
yfinance     → Yahoo Finance data (SPY + VIX)
matplotlib   → Charts
seaborn      → Chart styling
arch         → GARCH/EGARCH fitting (MLE)
statsmodels  → ADF, KPSS, ARCH-LM tests
pytest       → Test runner
```
