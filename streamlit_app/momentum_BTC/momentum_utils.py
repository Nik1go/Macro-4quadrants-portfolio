import os
import requests
import time
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_BASE = "https://api.binance.com"
INTERVAL = "1d"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_data")
os.makedirs(DATA_DIR, exist_ok=True)

MAX_SYMBOLS_PER_REQUEST = 1000

def fetch_binance_klines(symbol, interval, start_dt):
    """Downloads all 1D klines for `symbol` starting from `start_dt` via pagination."""
    url = f"{API_BASE}/api/v3/klines"
    start_ms = int(pd.Timestamp(start_dt, tz="UTC").timestamp() * 1000)
    all_rows = []
    
    session = requests.Session()
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": MAX_SYMBOLS_PER_REQUEST
        }
        try:
            r = session.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2)
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
            print(f"Error fetching data for {symbol}: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base_vol","taker_buy_quote_vol","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.set_index("close_time").sort_index()
    return df[["open","high","low","close","volume"]]

def _get_pipeline_data_dirs():
    """Resolve the shared pipeline CSV directories (same as data_fetcher.py uses)."""
    # momentum_utils.py is at: streamlit_app/momentum_BTC/momentum_utils.py
    # project root is two levels up: project_root/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    alt_usdt_dir = os.path.join(project_root, "data", "crypto", "ALT_USDT")
    alt_btc_dir  = os.path.join(project_root, "data", "crypto", "ALT_BTC")
    os.makedirs(alt_usdt_dir, exist_ok=True)
    os.makedirs(alt_btc_dir,  exist_ok=True)
    return alt_usdt_dir, alt_btc_dir


def _is_btc_pair(symbol: str) -> bool:
    return symbol.endswith("BTC") and symbol != "BTCUSDT"


def _csv_path_for(symbol: str) -> str:
    """Return the canonical CSV path for a symbol (same location as data_fetcher.py)."""
    alt_usdt_dir, alt_btc_dir = _get_pipeline_data_dirs()
    if _is_btc_pair(symbol):
        return os.path.join(alt_btc_dir,  f"{symbol}.csv")
    return os.path.join(alt_usdt_dir, f"{symbol}.csv")


def _fetch_and_save_csv(symbol: str, start_date: str) -> pd.DataFrame:
    """Download klines from Binance and append/create the shared CSV file."""
    csv_path = _csv_path_for(symbol)

    # ── Incremental update: only fetch missing tail ──
    fetch_start = start_date
    existing_df = pd.DataFrame()
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if not existing_df.empty and "date" in existing_df.columns:
            last_date = pd.to_datetime(existing_df["date"].max())
            next_day  = last_date + pd.Timedelta(days=1)
            if next_day > pd.Timestamp.utcnow().tz_localize(None):
                # Already up to date — parse and return
                existing_df["date"] = pd.to_datetime(existing_df["date"])
                return existing_df.set_index("date")
            fetch_start = next_day.strftime("%Y-%m-%d")

    print(f"  ⬇️  Downloading {symbol} from {fetch_start}...")
    raw = fetch_binance_klines(symbol, INTERVAL, fetch_start)
    if raw.empty and not existing_df.empty:
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        return existing_df.set_index("date")
    if raw.empty:
        return pd.DataFrame()

    # Convert raw (DatetimeIndex) → flat CSV format (date column, tz-naive)
    new_df = raw.copy().reset_index()
    new_df.columns = ["date", "open", "high", "low", "close", "volume"]
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df])
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)

    combined["date"] = pd.to_datetime(combined["date"])
    return combined.set_index("date")


def load_or_download(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Load OHLCV data for `symbol` starting from `start_date`.

    Strategy (no Parquet cache, no double download):
      1. Read from the shared pipeline CSV  (data/crypto/ALT_USDT/ or ALT_BTC/)
         → If it exists AND covers `start_date`, return it immediately.
      2. Otherwise fetch from Binance and save to the shared CSV
         (incremental: only missing tail is downloaded).

    Both the Airflow pipeline (data_fetcher.py) and the backtest (momentum_utils.py)
    point to the same files, so data is never downloaded twice.
    """
    csv_path = _csv_path_for(symbol)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        # Check coverage: if we have data on or before start_date, use it
        if not df.empty:
            earliest = df.index[0]
            if earliest.tz is not None:
                earliest = earliest.tz_localize(None)
            target_ts = pd.to_datetime(start_date)
            if earliest <= target_ts:
                return df  # ✅ cache hit — zero network call

    # Cache miss or insufficient history → fetch and save
    return _fetch_and_save_csv(symbol, start_date)



# ══════════════════════════════════════════════════════════
# Stablecoins / Fiat tokens to ALWAYS exclude from the universe
# ══════════════════════════════════════════════════════════
_STABLECOIN_KEYWORDS = ["USD", "EUR", "GBP", "USDS", "DAI", "TUSD", "BUSD", "FDUSD"]
_STABLECOIN_EXACT = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
    "USDPUSDT", "EURUSDT", "WBTCUSDT", "WBETHUSDT", "STETHUSDT",
    "BETHUSDT", "USD1USDT", "USDTUSDT", "PYUSDUSDT", "GBPUSDT",
    "AEURUSDT", "USTCUSDT",
}


def _is_stablecoin(symbol: str) -> bool:
    """Return True if the symbol should be excluded (stablecoin / fiat / wrapped)."""
    if symbol in _STABLECOIN_EXACT:
        return True
    base = symbol.replace("USDT", "")
    return any(kw in base for kw in _STABLECOIN_KEYWORDS)


def get_rolling_universe(all_volumes: pd.DataFrame, date, n: int = 20) -> list:
    """
    Return the top-N altcoin USDT symbols by 30-day rolling average volume
    as of `date`, excluding BTCUSDT and stablecoins.

    This replaces the static top-20 list and eliminates survivorship bias:
    at each date we only consider what was objectively the biggest volume
    at that point in time — exactly what we could have known in real life.
    """
    # Use data up to (but not including) `date` to avoid look-ahead
    try:
        idx = all_volumes.index.get_loc(date)
    except KeyError:
        # Fallback: use closest previous date
        past = all_volumes.index[all_volumes.index <= date]
        if past.empty:
            return []
        idx = all_volumes.index.get_loc(past[-1])

    window_start = max(0, idx - 30)
    window_df = all_volumes.iloc[window_start:idx]  # 30 rows, excluding current

    avg_vol = window_df.mean()

    # Filter: exclude BTC and stablecoins
    avg_vol = avg_vol[
        (avg_vol.index != "BTCUSDT") &
        (~avg_vol.index.map(_is_stablecoin))
    ]

    top_syms = avg_vol.nlargest(n).index.tolist()
    return top_syms


def _build_signal_loop(
    close, open_price, alt_usdt_volumes,
    btc_ret_5d, btc_median, btc_std, btc_skew,
    btc_above_sma_3d, btc_below_sma_3d,
    btc_vol_confirm,
    alt_btc_closes,
    alt_btc_above_sma_5d, alt_btc_below_sma_5d,
    alt_atr_df,
    daily_ret, basket_avg_ret,
    btc="BTCUSDT",
    top_n=20,
):
    """
    Core signal generation loop shared by run_momentum_backtest and run_heatmap_simulation.

    Key design decisions:
    - ONE position at a time (Long XOR Short) — no simultaneous long+short.
    - Universe refreshed daily using 30-day rolling volume rank (anti-survivorship bias).
    - Stablecoins excluded from the universe at every step.

    Returns: entries, exits, short_entries, short_exits (DataFrames, indexed like `close`)
    """
    dates = close.index
    all_syms = [s for s in close.columns if s != btc]

    entries       = pd.DataFrame(False, index=dates, columns=all_syms)
    exits         = pd.DataFrame(False, index=dates, columns=all_syms)
    short_entries = pd.DataFrame(False, index=dates, columns=all_syms)
    short_exits   = pd.DataFrame(False, index=dates, columns=all_syms)

    # Single position slot: dict with keys 'symbol', 'side', 'peak_or_trough'
    current_pos = None
    underperf_streak = 0

    for i in range(1, len(dates)):
        dt      = dates[i]
        prev_dt = dates[i - 1]  # All signal evaluation uses T-1 close

        # ── ROLLING UNIVERSE — top-N by 30d avg volume at prev_dt ──
        active_universe = get_rolling_universe(alt_usdt_volumes, prev_dt, n=top_n)
        # Keep only symbols for which we actually have data in `close`
        active_universe = [s for s in active_universe if s in close.columns]

        # ── EXIT CHECK ──
        if current_pos is not None:
            sym  = current_pos["symbol"]
            side = current_pos["side"]
            streak_key = f"{sym}_{side}"

            current_price = close.at[prev_dt, sym] if sym in close.columns else np.nan
            r_sym = daily_ret.at[prev_dt, sym] if sym in daily_ret.columns else np.nan
            r_avg = basket_avg_ret.at[prev_dt]

            # Update peak / trough for ATR trailing stop
            if pd.notna(current_price):
                if side == "long":
                    current_pos["peak"] = max(current_pos.get("peak", current_price), current_price)
                else:
                    current_pos["trough"] = min(current_pos.get("trough", current_price), current_price)

            # Update underperformance streak
            if pd.notna(r_sym) and pd.notna(r_avg):
                if (side == "long" and r_sym < r_avg) or (side == "short" and r_sym > r_avg):
                    underperf_streak += 1
                else:
                    underperf_streak = 0

            # ATR trailing stop
            atr_stop = False
            if sym in alt_atr_df.columns and prev_dt in alt_atr_df.index:
                atr_val = alt_atr_df.at[prev_dt, sym]
                if pd.notna(atr_val) and atr_val > 0 and pd.notna(current_price):
                    if side == "long" and current_price < (current_pos.get("peak", current_price) - 2.0 * atr_val):
                        atr_stop = True
                    elif side == "short" and current_price > (current_pos.get("trough", current_price) + 2.0 * atr_val):
                        atr_stop = True

            # BTC trend reversal exit
            btc_trend_exit = (
                (side == "long"  and bool(btc_below_sma_3d.at[prev_dt])) or
                (side == "short" and bool(btc_above_sma_3d.at[prev_dt]))
            )

            should_exit = underperf_streak >= 3 or btc_trend_exit or atr_stop

            if should_exit:
                if side == "long" and sym in exits.columns:
                    exits.loc[dt, sym] = True
                elif side == "short" and sym in short_exits.columns:
                    short_exits.loc[dt, sym] = True

                current_pos      = None
                underperf_streak = 0

        # ── ENTRY CHECK (only if no position is open) ──
        if current_pos is not None:
            continue

        ret_5 = btc_ret_5d.at[prev_dt]
        med   = btc_median.at[prev_dt]
        std   = btc_std.at[prev_dt]
        skew  = btc_skew.at[prev_dt]
        vol_ok = bool(btc_vol_confirm.at[prev_dt]) if prev_dt in btc_vol_confirm.index else True

        if pd.isna(ret_5) or pd.isna(med) or pd.isna(std) or pd.isna(skew):
            continue

        long_cond  = (ret_5 > (med + std)) and bool(btc_above_sma_3d.at[prev_dt]) and (skew > 0.15)  and vol_ok
        short_cond = (ret_5 < (med - std)) and bool(btc_below_sma_3d.at[prev_dt]) and (skew < -0.15) and vol_ok

        # ── LONG ENTRY ──
        if long_cond:
            best_sym, best_ret = None, -np.inf
            for sym in active_universe:
                btc_pair = sym.replace("USDT", "BTC")
                if btc_pair not in alt_btc_above_sma_5d.columns:
                    continue
                if not bool(alt_btc_above_sma_5d.at[prev_dt, btc_pair]):
                    continue
                r3 = close.at[prev_dt, sym] / close.iloc[max(0, i - 4)].get(sym, np.nan) - 1 if sym in close.columns else np.nan
                # Use pre-computed 3d return from close directly
                close_3d_ago = close.iloc[max(0, i - 4)][sym] if sym in close.columns else np.nan
                close_now    = close.at[prev_dt, sym]
                if pd.notna(close_3d_ago) and pd.notna(close_now) and close_3d_ago > 0:
                    r3 = (close_now / close_3d_ago) - 1.0
                else:
                    continue
                if r3 > best_ret:
                    best_ret, best_sym = r3, sym

            if best_sym and best_sym in entries.columns:
                entries.loc[dt, best_sym] = True
                entry_price = open_price.at[dt, best_sym] if best_sym in open_price.columns else close.at[dt, best_sym]
                current_pos = {"symbol": best_sym, "side": "long", "peak": entry_price}
                underperf_streak = 0

        # ── SHORT ENTRY ──
        elif short_cond:
            worst_sym, worst_ret = None, np.inf
            for sym in active_universe:
                btc_pair = sym.replace("USDT", "BTC")
                if btc_pair not in alt_btc_below_sma_5d.columns:
                    continue
                if not bool(alt_btc_below_sma_5d.at[prev_dt, btc_pair]):
                    continue
                close_3d_ago = close.iloc[max(0, i - 4)][sym] if sym in close.columns else np.nan
                close_now    = close.at[prev_dt, sym]
                if pd.notna(close_3d_ago) and pd.notna(close_now) and close_3d_ago > 0:
                    r3 = (close_now / close_3d_ago) - 1.0
                else:
                    continue
                if r3 < worst_ret:
                    worst_ret, worst_sym = r3, sym

            if worst_sym and worst_sym in short_entries.columns:
                short_entries.loc[dt, worst_sym] = True
                entry_price = open_price.at[dt, worst_sym] if worst_sym in open_price.columns else close.at[dt, worst_sym]
                current_pos = {"symbol": worst_sym, "side": "short", "trough": entry_price}
                underperf_streak = 0

    return entries, exits, short_entries, short_exits


def run_momentum_backtest(
    symbols,
    start_date="2020-01-01",
    sma_period=50,
    roll_lookback=180,
    fees_bps=6,
    slippage_bps=10
):
    """
    Executes the Crypto Momentum Strategy (1 position max, 100% capital).
    Uses the shared _build_signal_loop for anti-look-ahead, anti-survivorship signal generation.
    Returns: pf (vbt.Portfolio), None
    """
    all_closes_usdt = {}
    all_opens_usdt  = {}
    all_volumes_usdt = {}
    all_closes_btc  = {}

    btc_df = load_or_download("BTCUSDT", start_date)
    if btc_df.empty:
        return None, None

    for sym in symbols:
        usdt_df = load_or_download(sym, start_date)
        if not usdt_df.empty:
            all_closes_usdt[sym]  = usdt_df["close"]
            all_opens_usdt[sym]   = usdt_df["open"]
            all_volumes_usdt[sym] = usdt_df["volume"]
        if sym != "BTCUSDT":
            btc_pair_df = load_or_download(sym.replace("USDT", "BTC"), start_date)
            if not btc_pair_df.empty:
                all_closes_btc[sym.replace("USDT", "BTC")] = btc_pair_df["close"]

    if "BTCUSDT" not in all_closes_usdt:
        return None, None

    alt_usdt_closes  = pd.DataFrame(all_closes_usdt).ffill().loc[start_date:]
    alt_usdt_opens   = pd.DataFrame(all_opens_usdt).ffill().loc[start_date:]
    alt_usdt_volumes = pd.DataFrame(all_volumes_usdt).ffill().loc[start_date:]
    alt_btc_closes   = pd.DataFrame(all_closes_btc).ffill().loc[start_date:]

    btc   = "BTCUSDT"
    close = alt_usdt_closes.copy()

    btc_ret_5d = close[btc].pct_change(5)
    btc_median = btc_ret_5d.rolling(roll_lookback, min_periods=30).median()
    btc_std    = btc_ret_5d.rolling(roll_lookback, min_periods=30).std()
    btc_skew   = btc_ret_5d.rolling(120, min_periods=30).skew()

    btc_sma       = close[btc].rolling(sma_period, min_periods=sma_period).mean()
    btc_above_sma = close[btc] > btc_sma
    btc_below_sma = close[btc] < btc_sma
    btc_above_sma_3d = btc_above_sma.rolling(3).sum() == 3
    btc_below_sma_3d = btc_below_sma.rolling(3).sum() == 3

    btc_vol = btc_df["volume"].reindex(close.index).ffill() if "volume" in btc_df.columns else None
    if btc_vol is not None:
        btc_vol_confirm = btc_vol > btc_vol.rolling(20, min_periods=10).mean()
    else:
        btc_vol_confirm = pd.Series(True, index=close.index)

    alt_atr = {}
    for sym in [s for s in close.columns if s != btc]:
        sym_df = load_or_download(sym, start_date)
        if not sym_df.empty and all(c in sym_df.columns for c in ["high", "low", "close"]):
            h, l, cp = sym_df["high"], sym_df["low"], sym_df["close"].shift(1)
            tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
            alt_atr[sym] = tr.rolling(14, min_periods=14).mean().reindex(close.index).ffill()
    alt_atr_df = pd.DataFrame(alt_atr)

    alt_btc_sma        = alt_btc_closes.rolling(sma_period, min_periods=sma_period).mean()
    alt_btc_above_sma_5d = (alt_btc_closes > alt_btc_sma).rolling(5).sum() == 5
    alt_btc_below_sma_5d = (alt_btc_closes < alt_btc_sma).rolling(5).sum() == 5

    universe       = [s for s in close.columns if s != btc]
    daily_ret      = close.pct_change(1)
    basket_avg_ret = daily_ret[universe].mean(axis=1)

    entries, exits, short_entries, short_exits = _build_signal_loop(
        close=close,
        open_price=alt_usdt_opens,
        alt_usdt_volumes=alt_usdt_volumes,
        btc_ret_5d=btc_ret_5d, btc_median=btc_median, btc_std=btc_std, btc_skew=btc_skew,
        btc_above_sma_3d=btc_above_sma_3d, btc_below_sma_3d=btc_below_sma_3d,
        btc_vol_confirm=btc_vol_confirm,
        alt_btc_closes=alt_btc_closes,
        alt_btc_above_sma_5d=alt_btc_above_sma_5d, alt_btc_below_sma_5d=alt_btc_below_sma_5d,
        alt_atr_df=alt_atr_df,
        daily_ret=daily_ret, basket_avg_ret=basket_avg_ret,
        btc=btc,
    )

    if not entries.any().any() and not short_entries.any().any():
        return None, None

    syms  = [s for s in close.columns if s != btc]
    op    = alt_usdt_opens[syms]
    fees  = fees_bps / 10000.0
    slip  = slippage_bps / 10000.0

    pf = vbt.Portfolio.from_signals(
        close=op,
        entries=entries[syms],
        exits=exits[syms],
        short_entries=short_entries[syms],
        short_exits=short_exits[syms],
        fees=fees,
        slippage=slip,
        init_cash=10000.0,
        cash_sharing=True,
        size=1.0,
        size_type="percent",
        freq="1D",
    )
    return pf, None

def run_heatmap_simulation(
    symbols,
    start_date="2020-01-01",
    sma_periods=[20, 30, 40, 50, 60, 70, 80, 90, 100],
    roll_lookbacks=[30, 60, 90, 120, 180, 240, 300, 400, 500, 600],
    fees_bps=6,
    slippage_bps=10
):
    """
    Run the STRICT momentum strategy over a grid of (SMA, lookback) parameters.
    NOW: 1 position max, 100% sizing, rolling volume universe (anti-survivorship).
    Returns: Heatmap DF, Best Params, Best Portfolio, BTC Close Series
    """
    all_closes_usdt  = {}
    all_opens_usdt   = {}
    all_volumes_usdt = {}
    all_closes_btc   = {}

    btc_df = load_or_download("BTCUSDT", start_date)
    if btc_df.empty:
        return None, None, None, None

    for sym in symbols:
        usdt_df = load_or_download(sym, start_date)
        if not usdt_df.empty:
            all_closes_usdt[sym]  = usdt_df["close"]
            all_opens_usdt[sym]   = usdt_df["open"]
            all_volumes_usdt[sym] = usdt_df["volume"]
        if sym != "BTCUSDT":
            btc_pair_df = load_or_download(sym.replace("USDT", "BTC"), start_date)
            if not btc_pair_df.empty:
                all_closes_btc[sym.replace("USDT", "BTC")] = btc_pair_df["close"]

    if "BTCUSDT" not in all_closes_usdt:
        return None, None, None, None

    alt_usdt_closes  = pd.DataFrame(all_closes_usdt).ffill().loc[start_date:]
    alt_usdt_opens   = pd.DataFrame(all_opens_usdt).ffill().loc[start_date:]
    alt_usdt_volumes = pd.DataFrame(all_volumes_usdt).ffill().loc[start_date:]
    alt_btc_closes   = pd.DataFrame(all_closes_btc).ffill().loc[start_date:]

    btc   = "BTCUSDT"
    close = alt_usdt_closes.copy()
    btc_ret_5d = close[btc].pct_change(5)

    # Volume + ATR computed once (independent of SMA/lookback params)
    btc_vol = btc_df["volume"].reindex(close.index).ffill() if "volume" in btc_df.columns else None
    if btc_vol is not None:
        btc_vol_confirm = btc_vol > btc_vol.rolling(20, min_periods=10).mean()
    else:
        btc_vol_confirm = pd.Series(True, index=close.index)

    alt_atr = {}
    for sym in [s for s in close.columns if s != btc]:
        sym_df = load_or_download(sym, start_date)
        if not sym_df.empty and all(c in sym_df.columns for c in ["high", "low", "close"]):
            h, l, cp = sym_df["high"], sym_df["low"], sym_df["close"].shift(1)
            tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
            alt_atr[sym] = tr.rolling(14, min_periods=14).mean().reindex(close.index).ffill()
    alt_atr_df = pd.DataFrame(alt_atr)

    universe       = [s for s in close.columns if s != btc]
    daily_ret      = close.pct_change(1)
    basket_avg_ret = daily_ret[universe].mean(axis=1)

    syms = universe
    op   = alt_usdt_opens[syms]
    fees = fees_bps / 10000.0
    slip = slippage_bps / 10000.0

    results     = []
    best_pf     = None
    best_return = -1000.0
    best_params = (None, None)

    for sma in sma_periods:
        btc_sma          = close[btc].rolling(sma, min_periods=sma).mean()
        btc_above_sma_3d = (close[btc] > btc_sma).rolling(3).sum() == 3
        btc_below_sma_3d = (close[btc] < btc_sma).rolling(3).sum() == 3

        alt_btc_sma          = alt_btc_closes.rolling(sma, min_periods=sma).mean()
        alt_btc_above_sma_5d = (alt_btc_closes > alt_btc_sma).rolling(5).sum() == 5
        alt_btc_below_sma_5d = (alt_btc_closes < alt_btc_sma).rolling(5).sum() == 5

        for lookback in roll_lookbacks:
            btc_median = btc_ret_5d.rolling(lookback, min_periods=30).median()
            btc_std    = btc_ret_5d.rolling(lookback, min_periods=30).std()
            btc_skew   = btc_ret_5d.rolling(lookback, min_periods=30).skew()

            entries, exits, short_entries, short_exits = _build_signal_loop(
                close=close,
                open_price=alt_usdt_opens,
                alt_usdt_volumes=alt_usdt_volumes,
                btc_ret_5d=btc_ret_5d, btc_median=btc_median, btc_std=btc_std, btc_skew=btc_skew,
                btc_above_sma_3d=btc_above_sma_3d, btc_below_sma_3d=btc_below_sma_3d,
                btc_vol_confirm=btc_vol_confirm,
                alt_btc_closes=alt_btc_closes,
                alt_btc_above_sma_5d=alt_btc_above_sma_5d, alt_btc_below_sma_5d=alt_btc_below_sma_5d,
                alt_atr_df=alt_atr_df,
                daily_ret=daily_ret, basket_avg_ret=basket_avg_ret,
                btc=btc,
            )

            if not entries[syms].any().any() and not short_entries[syms].any().any():
                tot_ret = 0.0
            else:
                pf = vbt.Portfolio.from_signals(
                    close=op,
                    entries=entries[syms],
                    exits=exits[syms],
                    short_entries=short_entries[syms],
                    short_exits=short_exits[syms],
                    fees=fees,
                    slippage=slip,
                    init_cash=10000.0,
                    cash_sharing=True,
                    size=1.0,
                    size_type="percent",
                    freq="1D",
                )
                tot_ret = pf.stats().get("Total Return [%]", 0.0)

                if tot_ret > best_return:
                    best_return = tot_ret
                    best_pf     = pf
                    best_params = (sma, lookback)

            results.append({"SMA": sma, "Lookback": lookback, "Return": tot_ret})

    heatmap_df = pd.DataFrame(results).pivot(index="SMA", columns="Lookback", values="Return")
    return heatmap_df, best_params, best_pf, close[btc]
