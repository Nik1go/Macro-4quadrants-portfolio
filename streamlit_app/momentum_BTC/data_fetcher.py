"""
Data Fetcher for Crypto Momentum Pipeline.
Dynamically fetches Top 20 cryptos by 24h volume from Binance API.
Saves to data/crypto/ALT_USDT/ and data/crypto/ALT_BTC/ as CSV files.
"""

import os
import time
import requests
import pandas as pd

API_BASE = "https://api.binance.com"
INTERVAL = "1d"
MAX_LIMIT = 1000

# Stablecoins, wrapped tokens, and fiat-backed tokens to exclude
EXCLUDED_SYMBOLS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
    "USDPUSDT", "EURUSDT", "WBTCUSDT", "WBETHUSDT", "STETHUSDT",
    "BETHUSDT", "USD1USDT", "USDTUSDT", "PYUSDUSDT", "GBPUSDT",
    "AEURUSDT", "USTCUSDT",
}

# Keywords that indicate stablecoins/fiat tokens to auto-exclude
EXCLUDED_KEYWORDS = ["USD", "EUR", "GBP", "USDS"]

TOP_N = 80  # Download a wide universe so the rolling-volume filter can pick the real top-20 per date


def _get_data_dirs():
    """Resolve data directories relative to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # momentum_BTC/ -> streamlit_app/ -> project_root/
    project_root = os.path.dirname(os.path.dirname(current_dir))

    alt_usdt_dir = os.path.join(project_root, "data", "crypto", "ALT_USDT")
    alt_btc_dir = os.path.join(project_root, "data", "crypto", "ALT_BTC")

    os.makedirs(alt_usdt_dir, exist_ok=True)
    os.makedirs(alt_btc_dir, exist_ok=True)

    return alt_usdt_dir, alt_btc_dir


def _is_excluded(symbol):
    """Check if a symbol is a stablecoin/fiat token by name or keyword."""
    if symbol in EXCLUDED_SYMBOLS:
        return True
    # Check base asset (remove USDT suffix) against keywords
    base = symbol.replace("USDT", "")
    for kw in EXCLUDED_KEYWORDS:
        if kw in base:
            return True
    return False


def fetch_top_symbols(n=TOP_N):
    """
    Dynamically fetch the Top N cryptocurrencies by 24h quote volume
    from Binance. Only keeps cryptos that have a valid ALT/BTC pair.
    Over-fetches to ensure we get N valid pairs.
    
    Returns:
        usdt_symbols: list of USDT pair symbols (e.g. ["BTCUSDT", "ETHUSDT", ...])
        btc_symbols:  list of confirmed ALT/BTC pair symbols (e.g. ["ETHBTC", ...])
    """
    url = f"{API_BASE}/api/v3/ticker/24hr"
    
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        tickers = r.json()
    except Exception as e:
        print(f"❌ Failed to fetch ticker data: {e}")
        print("   Falling back to default symbol list")
        return _fallback_symbols()
    
    # Build set of all existing symbols for BTC pair validation
    all_symbols = {t["symbol"] for t in tickers}
    
    # Filter USDT pairs, exclude stablecoins/fiat
    usdt_tickers = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and not _is_excluded(t["symbol"])
        and float(t.get("quoteVolume", 0)) > 0
    ]
    
    # Sort by 24h quote volume (descending)
    usdt_tickers.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    
    # Iterate through candidates, only keep those with valid ALT/BTC pair
    top_usdt = ["BTCUSDT"]  # BTC always first
    btc_pairs = []
    skipped = []
    
    for t in usdt_tickers:
        sym = t["symbol"]
        if sym == "BTCUSDT":
            continue
        if len(top_usdt) >= n + 1:  # +1 for BTC
            break
        
        btc_sym = sym.replace("USDT", "BTC")
        if btc_sym in all_symbols:
            top_usdt.append(sym)
            btc_pairs.append(btc_sym)
        else:
            skipped.append(sym)
    
    # Log results
    print(f"\n📋 Top {len(top_usdt)} selected (by 24h volume, with ALT/BTC pair):")
    for i, sym in enumerate(top_usdt, 1):
        vol = next((float(t["quoteVolume"]) for t in usdt_tickers if t["symbol"] == sym), 0)
        print(f"   {i:2d}. {sym:<12s} — Vol: ${vol:>15,.0f}")
    
    if skipped:
        print(f"\n⚠️ Skipped {len(skipped)} (no BTC pair): {', '.join(skipped[:10])}")
    
    print(f"\n📋 ALT/BTC pairs available: {len(btc_pairs)}")
    
    return top_usdt, btc_pairs


def _fallback_symbols():
    """Fallback symbol list if API call fails."""
    usdt = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
        "DOTUSDT", "LTCUSDT", "MATICUSDT", "NEARUSDT", "UNIUSDT",
        "ATOMUSDT", "AAVEUSDT", "XLMUSDT", "SUIUSDT", "FETUSDT",
    ]
    btc = [s.replace("USDT", "BTC") for s in usdt if s != "BTCUSDT"]
    return usdt, btc


def fetch_binance_klines(symbol, interval=INTERVAL, start_dt="2020-01-01"):
    """
    Downloads all daily klines for `symbol` from Binance via pagination.
    Returns a DataFrame with columns: [date, open, high, low, close, volume].
    """
    url = f"{API_BASE}/api/v3/klines"
    start_ms = int(pd.Timestamp(start_dt, tz="UTC").timestamp() * 1000)
    all_rows = []
    session = requests.Session()

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": MAX_LIMIT
        }
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                print(f"  Rate limited on {symbol}, sleeping 5s...")
                time.sleep(5)
                continue
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            all_rows.extend(rows)
            last_close_ms = rows[-1][6]
            next_start_ms = last_close_ms + 1
            if next_start_ms == start_ms:
                break
            start_ms = next_start_ms
            time.sleep(0.05)
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _update_csv(symbol, data_dir, start_date="2020-01-01"):
    """
    Incrementally update a CSV file for the given symbol.
    If the file exists, only fetches new data after the last recorded date.
    """
    csv_path = os.path.join(data_dir, f"{symbol}.csv")

    existing_df = pd.DataFrame()
    fetch_start = start_date

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if not existing_df.empty and "date" in existing_df.columns:
            last_date = pd.to_datetime(existing_df["date"].max())
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            # Compare tz-naive timestamps (strip tz from utcnow)
            if pd.Timestamp(fetch_start) > pd.Timestamp.utcnow().tz_localize(None):
                print(f"  ✅ {symbol} already up to date")
                return csv_path

    new_df = fetch_binance_klines(symbol, start_dt=fetch_start)

    if new_df.empty and not existing_df.empty:
        print(f"  ✅ {symbol} no new data")
        return csv_path
    elif new_df.empty:
        print(f"  ⚠️ {symbol} no data available")
        return None

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df])
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    print(f"  ✅ {symbol} → {len(combined)} rows saved")
    return csv_path


def fetch_all_crypto_data(start_date="2020-01-01"):
    """
    Fetch and save all crypto data:
    1. Dynamically discover Top 20 by 24h volume from Binance
    2. Download USDT pairs → data/crypto/ALT_USDT/
    3. Download ALT/BTC pairs → data/crypto/ALT_BTC/
    
    Returns dict with paths to CSV files.
    """
    alt_usdt_dir, alt_btc_dir = _get_data_dirs()
    results = {"ALT_USDT": {}, "ALT_BTC": {}}

    # Dynamically fetch top symbols
    usdt_symbols, btc_symbols = fetch_top_symbols(n=TOP_N)

    print("\n" + "=" * 60)
    print("📡 Fetching ALT/USDT pairs (including BTCUSDT)...")
    print("=" * 60)
    for symbol in usdt_symbols:
        path = _update_csv(symbol, alt_usdt_dir, start_date)
        if path:
            results["ALT_USDT"][symbol] = path

    print()
    print("=" * 60)
    print("📡 Fetching ALT/BTC pairs...")
    print("=" * 60)
    for symbol in btc_symbols:
        path = _update_csv(symbol, alt_btc_dir, start_date)
        if path:
            results["ALT_BTC"][symbol] = path

    print(f"\n✅ Total: {len(results['ALT_USDT'])} USDT pairs, {len(results['ALT_BTC'])} BTC pairs")
    return results


if __name__ == "__main__":
    fetch_all_crypto_data()
