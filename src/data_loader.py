"""
Data Loader Module
Fetches and prepares market data for volatility analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


def fetch_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock/index ticker symbol (e.g., 'SPY', '^GSPC')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str
        Period to fetch if dates not specified (default: '5y')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(ticker, period=period, progress=False)
    
    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data


def calculate_returns(
    prices: pd.Series,
    method: str = 'log'
) -> pd.Series:
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series (typically Close prices)
    method : str
        'log' for log returns, 'simple' for arithmetic returns
    
    Returns:
    --------
    pd.Series
        Returns series
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.dropna()


def fetch_vix_data(period: str = "5y") -> pd.DataFrame:
    """
    Fetch VIX (Volatility Index) data for benchmarking.
    
    Parameters:
    -----------
    period : str
        Period to fetch (default: '5y')
    
    Returns:
    --------
    pd.DataFrame
        VIX price data
    """
    vix = yf.download("^VIX", period=period, progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    return vix


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling realized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size (default: 21 for monthly)
    annualize : bool
        Whether to annualize (multiply by sqrt(252))
    
    Returns:
    --------
    pd.Series
        Realized volatility series
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def prepare_volatility_data(
    ticker: str = "SPY",
    period: str = "5y"
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepare complete dataset for volatility modeling.
    
    Parameters:
    -----------
    ticker : str
        Stock/index ticker
    period : str
        Data period
    
    Returns:
    --------
    Tuple containing:
        - prices: DataFrame with OHLCV data
        - returns: Series of log returns
        - realized_vol: Series of realized volatility
        - vix: Series of VIX levels
    """
    print(f"Fetching data for {ticker}...")
    prices = fetch_stock_data(ticker, period=period)
    
    print("Calculating returns...")
    close_col = 'Close' if 'Close' in prices.columns else 'Adj Close'
    returns = calculate_returns(prices[close_col])
    
    print("Calculating realized volatility...")
    realized_vol = calculate_realized_volatility(returns)
    
    print("Fetching VIX data for benchmarking...")
    vix_data = fetch_vix_data(period=period)
    vix_close = 'Close' if 'Close' in vix_data.columns else 'Adj Close'
    vix = vix_data[vix_close] / 100  # Convert to decimal
    
    # Align indices
    common_idx = returns.index.intersection(vix.index)
    returns = returns.loc[common_idx]
    realized_vol = realized_vol.loc[common_idx]
    vix = vix.loc[common_idx]
    
    print(f"Data loaded: {len(returns)} trading days")
    print(f"Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    
    return prices, returns, realized_vol, vix


if __name__ == "__main__":
    # Test the module
    prices, returns, realized_vol, vix = prepare_volatility_data("SPY", "2y")
    
    print("\nData Statistics:")
    print(f"  Mean daily return: {returns.mean():.4%}")
    print(f"  Daily volatility: {returns.std():.4%}")
    print(f"  Annualized volatility: {returns.std() * np.sqrt(252):.2%}")
    print(f"  Current realized vol (21d): {realized_vol.iloc[-1]:.2%}")
    print(f"  Current VIX: {vix.iloc[-1]:.2%}")
