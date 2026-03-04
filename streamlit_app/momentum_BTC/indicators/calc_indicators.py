"""
Indicator Calculation Module for Crypto Momentum Strategy.
Computes all technical indicators needed for Long/Short signal generation.

Indicators computed:
- BTC: 5-day return, rolling median, rolling std, SMA
- ALT/BTC: SMA for trend validation
- ALT: 3-day return for ranking
"""

import os
import pandas as pd
import numpy as np


def _get_data_dirs():
    """Resolve data directories relative to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # indicators/ -> momentum_BTC/ -> streamlit_app/ -> project_root/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    alt_usdt_dir = os.path.join(project_root, "data", "crypto", "ALT_USDT")
    alt_btc_dir = os.path.join(project_root, "data", "crypto", "ALT_BTC")
    return alt_usdt_dir, alt_btc_dir


def load_closes(data_dir, symbols=None):
    """
    Load close prices from CSV files in a directory.
    Returns a DataFrame with date index and one column per symbol.
    """
    frames = {}
    if symbols is None:
        # Load all CSVs in the directory
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                symbol = f.replace(".csv", "")
                df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"])
                df = df.set_index("date")
                frames[symbol] = df["close"]
    else:
        for symbol in symbols:
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, parse_dates=["date"])
                df = df.set_index("date")
                frames[symbol] = df["close"]

    if not frames:
        return pd.DataFrame()

    closes = pd.DataFrame(frames)
    closes = closes.sort_index().ffill()
    return closes


def load_ohlcv(data_dir, symbol):
    """
    Load full OHLCV data for a single symbol.
    Returns a DataFrame with date index and columns: open, high, low, close, volume.
    """
    csv_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def compute_btc_indicators(btc_close, sma_period=50, roll_lookback=600, btc_volume=None):
    """
    Compute BTC-specific indicators.
    
    Returns dict with:
        - btc_ret_5d: 5-day return
        - btc_median: Rolling median of 5D return
        - btc_std: Rolling std of 5D return
        - btc_sma: Simple Moving Average
        - btc_above_sma_3d: True if BTC > SMA for 3 consecutive days
        - btc_below_sma_3d: True if BTC < SMA for 3 consecutive days
        - btc_skew: 120-day rolling skewness of 5D returns
        - btc_vol_confirm: True if today's BTC volume > SMA(20) of volume
    """
    btc_ret_5d = btc_close.pct_change(5)
    btc_median = btc_ret_5d.rolling(roll_lookback, min_periods=30).median()
    # Calculate Skewness to determine the regime of the return distribution over exactly 120 days
    btc_skew = btc_ret_5d.rolling(120, min_periods=30).skew()
    
    btc_std = btc_ret_5d.rolling(roll_lookback, min_periods=30).std()
    btc_sma = btc_close.rolling(sma_period, min_periods=sma_period).mean()

    # BTC > SMA for 3 consecutive days (using shift to avoid look-ahead)
    above_sma = btc_close.shift(1) > btc_sma.shift(1)
    btc_above_sma_3d = above_sma & above_sma.shift(1) & above_sma.shift(2)

    # BTC < SMA for 3 consecutive days
    below_sma = btc_close.shift(1) < btc_sma.shift(1)
    btc_below_sma_3d = below_sma & below_sma.shift(1) & below_sma.shift(2)

    # Volume Filter: today's volume must exceed the 20-day SMA of volume
    # This filters out fake pumps/dumps with low participation
    if btc_volume is not None and not btc_volume.empty:
        vol_sma_20 = btc_volume.rolling(20, min_periods=10).mean()
        btc_vol_confirm = btc_volume.shift(1) > vol_sma_20.shift(1)  # shifted to avoid look-ahead
    else:
        btc_vol_confirm = pd.Series(True, index=btc_close.index)  # default: always True

    return {
        "btc_ret_5d": btc_ret_5d,
        "btc_median": btc_median,
        "btc_std": btc_std,
        "btc_sma": btc_sma,
        "btc_above_sma_3d": btc_above_sma_3d,
        "btc_below_sma_3d": btc_below_sma_3d,
        "btc_skew": btc_skew,
        "btc_vol_confirm": btc_vol_confirm,
    }


def compute_alt_btc_sma(alt_btc_closes, sma_period=50):
    """
    Compute ALT/BTC SMA and trend conditions.
    
    Returns dict with:
        - alt_btc_sma: SMA of ALT/BTC ratio per altcoin
        - alt_btc_above_sma_5d: True if ALT/BTC > SMA for 5 consecutive days
        - alt_btc_below_sma_5d: True if ALT/BTC < SMA for 5 consecutive days
    """
    sma = alt_btc_closes.rolling(sma_period, min_periods=sma_period).mean()

    # Above SMA for 5 consecutive days (shifted to avoid look-ahead)
    above = alt_btc_closes.shift(1) > sma.shift(1)
    above_5d = above.copy()
    for lag in range(1, 5):
        above_5d = above_5d & above.shift(lag)

    # Below SMA for 5 consecutive days
    below = alt_btc_closes.shift(1) < sma.shift(1)
    below_5d = below.copy()
    for lag in range(1, 5):
        below_5d = below_5d & below.shift(lag)

    return {
        "alt_btc_sma": sma,
        "alt_btc_above_sma_5d": above_5d,
        "alt_btc_below_sma_5d": below_5d,
    }


def compute_alt_returns(alt_usdt_closes):
    """
    Compute altcoin returns for ranking.
    
    Returns:
        - ret_3d: 3-day return per altcoin (for selection ranking)
        - ret_daily: daily return per altcoin (for exit basket comparison)
        - basket_avg_ret: average daily return of all altcoins
    """
    # Exclude BTCUSDT from altcoin universe
    alt_cols = [c for c in alt_usdt_closes.columns if c != "BTCUSDT"]
    alt_closes = alt_usdt_closes[alt_cols]

    ret_3d = alt_closes.pct_change(3).shift(1)  # shifted to avoid look-ahead
    ret_daily = alt_closes.pct_change(1)
    basket_avg_ret = ret_daily.mean(axis=1)

    return {
        "ret_3d": ret_3d,
        "ret_daily": ret_daily,
        "basket_avg_ret": basket_avg_ret,
    }


def compute_alt_atr(alt_usdt_dir, symbols=None, atr_period=14):
    """
    Compute ATR (Average True Range) for each altcoin.
    ATR measures daily volatility and is used for dynamic trailing stops.
    
    Trailing Stop rule: exit if price moves 2× ATR against the position
    from its best price since entry.
    
    Returns:
        - alt_atr: DataFrame with ATR per altcoin (columns) per date (index)
    """
    frames = {}
    if symbols is None:
        files = [f for f in os.listdir(alt_usdt_dir) if f.endswith(".csv")]
        symbols = [f.replace(".csv", "") for f in files]
    
    for symbol in symbols:
        if symbol == "BTCUSDT":
            continue
        ohlcv = load_ohlcv(alt_usdt_dir, symbol)
        if ohlcv.empty or not all(c in ohlcv.columns for c in ['high', 'low', 'close']):
            continue
        
        high = ohlcv['high']
        low = ohlcv['low']
        close_prev = ohlcv['close'].shift(1)
        
        # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        frames[symbol] = tr.rolling(atr_period, min_periods=atr_period).mean()
    
    if not frames:
        return pd.DataFrame()
    
    alt_atr = pd.DataFrame(frames).sort_index().ffill()
    return alt_atr


def compute_all_indicators(sma_period=50, roll_lookback=600):
    """
    Master function: load all data and compute every indicator.
    
    Returns a dict with all computed indicators needed by generate_signals.
    """
    alt_usdt_dir, alt_btc_dir = _get_data_dirs()

    # Load data
    alt_usdt_closes = load_closes(alt_usdt_dir)
    alt_btc_closes = load_closes(alt_btc_dir)

    if alt_usdt_closes.empty or "BTCUSDT" not in alt_usdt_closes.columns:
        raise ValueError("BTCUSDT data not found in ALT_USDT directory")

    btc_close = alt_usdt_closes["BTCUSDT"]

    # Load BTC volume for the volume confirmation filter
    btc_ohlcv = load_ohlcv(alt_usdt_dir, "BTCUSDT")
    btc_volume = btc_ohlcv["volume"] if not btc_ohlcv.empty and "volume" in btc_ohlcv.columns else None

    # Compute indicators
    btc_ind = compute_btc_indicators(btc_close, sma_period, roll_lookback, btc_volume=btc_volume)
    alt_btc_ind = compute_alt_btc_sma(alt_btc_closes, sma_period)
    alt_ret = compute_alt_returns(alt_usdt_closes)
    alt_atr = compute_alt_atr(alt_usdt_dir, atr_period=14)

    return {
        "btc_close": btc_close,
        "alt_usdt_closes": alt_usdt_closes,
        "alt_btc_closes": alt_btc_closes,
        "alt_atr": alt_atr,
        **btc_ind,
        **alt_btc_ind,
        **alt_ret,
    }
