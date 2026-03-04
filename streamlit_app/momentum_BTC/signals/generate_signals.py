"""
Signal Generation Module for Crypto Momentum Strategy.
Evaluates Long/Short entry and exit conditions daily.

Long Entry:  BTC bullish (5D ret > median + 1σ) + BTC > SMA 3d + ALT/BTC > SMA 5d → best 3D alt
Short Entry: BTC bearish (5D ret < median - 1σ) + BTC < SMA 3d + ALT/BTC < SMA 5d → worst 3D alt
Long Exit:   asset underperf basket 3d OR BTC < SMA 3d
Short Exit:  asset outperf basket 3d OR BTC > SMA 3d
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_signals_dir():
    """Resolve signals output directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    signals_dir = os.path.join(project_root, "data", "crypto", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    return signals_dir


def _get_state_path():
    """Resolve state.json path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return os.path.join(project_root, "data", "crypto", "state.json")


def load_state():
    """Load current portfolio state from state.json."""
    state_path = _get_state_path()
    default_state = {
        "positions": [],          # List of open positions
        "cash": 10000.0,          # Available cash
        "initial_cash": 10000.0,
        "underperf_streaks": {},  # {symbol: consecutive underperf days}
    }
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            return default_state
    return default_state


def save_state(state):
    """Save current portfolio state to state.json."""
    state_path = _get_state_path()
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def evaluate_exits(state, indicators, today_idx):
    """
    Evaluate exit conditions for all open positions.
    
    Returns list of exit signals: [{"symbol": ..., "side": ..., "reason": ...}]
    """
    exits = []
    today = indicators["ret_daily"].index[today_idx]

    for pos in state["positions"]:
        symbol = pos["symbol"]
        side = pos["side"]  # "long" or "short"
        streak_key = f"{symbol}_{side}"

        # Get today's return for this asset vs basket
        if symbol in indicators["ret_daily"].columns:
            asset_ret = indicators["ret_daily"].at[today, symbol]
            basket_ret = indicators["basket_avg_ret"].iloc[today_idx]
        else:
            # Symbol removed from universe — force exit
            exits.append({"symbol": symbol, "side": side, "reason": "symbol_delisted"})
            continue

        if pd.isna(asset_ret) or pd.isna(basket_ret):
            continue

        # ── LONG EXIT ──
        if side == "long":
            # Momentum stop: underperforms basket 3 consecutive days
            if asset_ret < basket_ret:
                state["underperf_streaks"][streak_key] = state["underperf_streaks"].get(streak_key, 0) + 1
            else:
                state["underperf_streaks"][streak_key] = 0

            if state["underperf_streaks"].get(streak_key, 0) >= 3:
                exits.append({"symbol": symbol, "side": side, "reason": "momentum_stop_underperf"})
                continue

            # BTC trend stop: BTC < SMA 3 consecutive days
            if indicators["btc_below_sma_3d"].iloc[today_idx]:
                exits.append({"symbol": symbol, "side": side, "reason": "btc_trend_stop"})
                continue

        # ── SHORT EXIT ──
        elif side == "short":
            # Momentum stop: outperforms basket 3 consecutive days
            if asset_ret > basket_ret:
                state["underperf_streaks"][streak_key] = state["underperf_streaks"].get(streak_key, 0) + 1
            else:
                state["underperf_streaks"][streak_key] = 0

            if state["underperf_streaks"].get(streak_key, 0) >= 3:
                exits.append({"symbol": symbol, "side": side, "reason": "momentum_stop_outperf"})
                continue

            # BTC trend stop: BTC > SMA 3 consecutive days
            if indicators["btc_above_sma_3d"].iloc[today_idx]:
                exits.append({"symbol": symbol, "side": side, "reason": "btc_trend_stop"})
                continue

    return exits


def evaluate_long_entry(indicators, today_idx, alt_btc_cols):
    """
    Evaluate long entry conditions.
    
    Returns: {"symbol": best_alt, "side": "long"} or None
    """
    today = indicators["btc_ret_5d"].index[today_idx]

    # BTC conditions
    btc_ret = indicators["btc_ret_5d"].iloc[today_idx]
    btc_med = indicators["btc_median"].iloc[today_idx]
    btc_std_val = indicators["btc_std"].iloc[today_idx]
    btc_skew = indicators["btc_skew"].iloc[today_idx]

    if pd.isna(btc_ret) or pd.isna(btc_med) or pd.isna(btc_std_val) or pd.isna(btc_skew):
        return None

    # Condition 1A: Regime Filter (Distribution Skewness must lean positive/bullish)
    if btc_skew <= 0.15:
        logger.info(f"[BTC] Conditions Long ignorées : Skewness ({btc_skew:.2f}) <= 0.15 (Régime Asymétrique Faible/Baissier).")
        return None

    # Condition 1B: Volume Confirmation (BTC volume > SMA20 of volume)
    if "btc_vol_confirm" in indicators:
        vol_ok = indicators["btc_vol_confirm"].iloc[today_idx]
        if pd.notna(vol_ok) and not vol_ok:
            logger.info(f"[BTC] Conditions Long ignorées : Volume BTC insuffisant (pas de confirmation institutionnelle).")
            return None

    # Condition 1B: Momentum Trigger (BTC 5D return > median + 1σ)
    if btc_ret <= (btc_med + btc_std_val):
        return None

    # Condition 2: BTC > SMA for 3 consecutive days
    if not indicators["btc_above_sma_3d"].iloc[today_idx]:
        return None

    # Condition 3: Filter altcoins where ALT/BTC > SMA for 5 consecutive days
    filtered_alts = []
    for alt_btc_sym in alt_btc_cols:
        if alt_btc_sym in indicators["alt_btc_above_sma_5d"].columns:
            if indicators["alt_btc_above_sma_5d"].at[today, alt_btc_sym]:
                # Convert ALT/BTC symbol to USDT symbol (e.g., ETHBTC → ETHUSDT)
                usdt_sym = alt_btc_sym.replace("BTC", "USDT")
                if usdt_sym in indicators["ret_3d"].columns:
                    filtered_alts.append(usdt_sym)

    if not filtered_alts:
        print("   🟢 LONG: BTC conditions met ✅ but NO altcoin passes ALT/BTC > SMA (5d) filter")
        return None

    # Condition 4: Select best 3D return among filtered
    ret_3d_today = indicators["ret_3d"].loc[today, filtered_alts]
    ret_3d_today = ret_3d_today.dropna()
    if ret_3d_today.empty:
        return None

    # Log the filtered basket with returns
    ret_sorted = ret_3d_today.sort_values(ascending=False)
    print(f"   🟢 LONG BASKET — {len(ret_sorted)} altcoins pass ALT/BTC > SMA (5d):")
    for sym, ret in ret_sorted.items():
        print(f"      {'→' if sym == ret_sorted.index[0] else ' '} {sym:<12s}  3D ret: {ret:+.2%}")

    best_alt = ret_sorted.index[0]
    print(f"   ✅ SELECTED: {best_alt} (best 3D: {ret_sorted.iloc[0]:+.2%})")
    return {"symbol": best_alt, "side": "long", "ret_3d": float(ret_sorted.iloc[0])}


def evaluate_short_entry(indicators, today_idx, alt_btc_cols):
    """
    Evaluate short entry conditions (mirror of long).
    
    Returns: {"symbol": worst_alt, "side": "short"} or None
    """
    today = indicators["btc_ret_5d"].index[today_idx]

    btc_ret = indicators["btc_ret_5d"].iloc[today_idx]
    btc_med = indicators["btc_median"].iloc[today_idx]
    btc_std_val = indicators["btc_std"].iloc[today_idx]
    btc_skew = indicators["btc_skew"].iloc[today_idx]

    if pd.isna(btc_ret) or pd.isna(btc_med) or pd.isna(btc_std_val) or pd.isna(btc_skew):
        return None

    # Condition 1A: Regime Filter (Distribution Skewness must lean negative/bearish)
    if btc_skew >= -0.15:
        logger.info(f"[BTC] Conditions Short ignorées : Skewness ({btc_skew:.2f}) >= -0.15 (Régime Asymétrique Faible/Haussier).")
        return None

    # Condition 1B: Volume Confirmation (BTC volume > SMA20 of volume)
    if "btc_vol_confirm" in indicators:
        vol_ok = indicators["btc_vol_confirm"].iloc[today_idx]
        if pd.notna(vol_ok) and not vol_ok:
            logger.info(f"[BTC] Conditions Short ignorées : Volume BTC insuffisant (pas de confirmation institutionnelle).")
            return None

    # Condition 1B: Momentum Trigger (BTC 5D return < median - 1σ)
    if btc_ret >= (btc_med - btc_std_val):
        return None

    # Condition 2: BTC < SMA for 3 consecutive days
    if not indicators["btc_below_sma_3d"].iloc[today_idx]:
        return None

    # Condition 3: Filter altcoins where ALT/BTC < SMA for 5 consecutive days
    filtered_alts = []
    for alt_btc_sym in alt_btc_cols:
        if alt_btc_sym in indicators["alt_btc_below_sma_5d"].columns:
            if indicators["alt_btc_below_sma_5d"].at[today, alt_btc_sym]:
                usdt_sym = alt_btc_sym.replace("BTC", "USDT")
                if usdt_sym in indicators["ret_3d"].columns:
                    filtered_alts.append(usdt_sym)

    if not filtered_alts:
        print("   🔴 SHORT: BTC conditions met ✅ but NO altcoin passes ALT/BTC < SMA (5d) filter")
        return None

    # Condition 4: Select worst 3D return (most abandoned)
    ret_3d_today = indicators["ret_3d"].loc[today, filtered_alts]
    ret_3d_today = ret_3d_today.dropna()
    if ret_3d_today.empty:
        return None

    # Log the filtered basket with returns
    ret_sorted = ret_3d_today.sort_values(ascending=True)
    print(f"   🔴 SHORT BASKET — {len(ret_sorted)} altcoins pass ALT/BTC < SMA (5d):")
    for sym, ret in ret_sorted.items():
        print(f"      {'→' if sym == ret_sorted.index[0] else ' '} {sym:<12s}  3D ret: {ret:+.2%}")

    worst_alt = ret_sorted.index[0]
    print(f"   ✅ SELECTED: {worst_alt} (worst 3D: {ret_sorted.iloc[0]:+.2%})")
    return {"symbol": worst_alt, "side": "short", "ret_3d": float(ret_sorted.iloc[0])}


def generate_daily_signals(indicators, state=None):
    """
    Main function: evaluate today's signals.
    Called by the DAG or dry-run script.
    
    Args:
        indicators: dict from calc_indicators.compute_all_indicators()
        state: portfolio state (loaded from state.json). If None, loads from disk.
        
    Returns:
        signal_report: dict with exits, entries, updated state
    """
    if state is None:
        state = load_state()

    today_idx = len(indicators["btc_ret_5d"]) - 1
    today = indicators["btc_ret_5d"].index[today_idx]
    today_str = str(today.date()) if hasattr(today, 'date') else str(today)[:10]

    alt_btc_cols = list(indicators["alt_btc_closes"].columns) if "alt_btc_closes" in indicators else []

    report = {
        "date": today_str,
        "exits": [],
        "entries": [],
        "btc_close": float(indicators["btc_close"].iloc[today_idx]) if pd.notna(indicators["btc_close"].iloc[today_idx]) else None,
        "btc_sma": float(indicators["btc_sma"].iloc[today_idx]) if pd.notna(indicators["btc_sma"].iloc[today_idx]) else None,
        "btc_ret_5d": float(indicators["btc_ret_5d"].iloc[today_idx]) if pd.notna(indicators["btc_ret_5d"].iloc[today_idx]) else None,
    }

    # ── STEP 1: Evaluate exits on open positions ──
    exit_signals = evaluate_exits(state, indicators, today_idx)
    for ex in exit_signals:
        report["exits"].append(ex)
        # Remove position from state
        state["positions"] = [p for p in state["positions"] if not (p["symbol"] == ex["symbol"] and p["side"] == ex["side"])]
        # Clean up streak
        streak_key = f"{ex['symbol']}_{ex['side']}"
        state["underperf_streaks"].pop(streak_key, None)

    # ── STEP 2: Evaluate entries if we have capacity ──
    # Count current open long and short positions
    n_long = sum(1 for p in state["positions"] if p["side"] == "long")
    n_short = sum(1 for p in state["positions"] if p["side"] == "short")

    # Try Long entry (max 1 long position at a time for simplicity)
    if n_long == 0:
        long_signal = evaluate_long_entry(indicators, today_idx, alt_btc_cols)
        if long_signal:
            entry_price = float(indicators["alt_usdt_closes"].at[today, long_signal["symbol"]])
            size_cash = state["cash"] * 0.25  # 25% allocation
            qty = size_cash / entry_price if entry_price > 0 else 0

            long_signal["entry_price"] = entry_price
            long_signal["qty"] = qty
            long_signal["entry_date"] = today_str
            long_signal["size_cash"] = size_cash
            report["entries"].append(long_signal)

            state["positions"].append({
                "symbol": long_signal["symbol"],
                "side": "long",
                "entry_price": entry_price,
                "qty": qty,
                "entry_date": today_str,
            })
            state["cash"] -= size_cash

    # Try Short entry (max 1 short position at a time)
    if n_short == 0:
        short_signal = evaluate_short_entry(indicators, today_idx, alt_btc_cols)
        if short_signal:
            entry_price = float(indicators["alt_usdt_closes"].at[today, short_signal["symbol"]])
            size_cash = state["cash"] * 0.25  # 25% allocation
            qty = size_cash / entry_price if entry_price > 0 else 0

            short_signal["entry_price"] = entry_price
            short_signal["qty"] = qty
            short_signal["entry_date"] = today_str
            short_signal["size_cash"] = size_cash
            report["entries"].append(short_signal)

            state["positions"].append({
                "symbol": short_signal["symbol"],
                "side": "short",
                "entry_price": entry_price,
                "qty": qty,
                "entry_date": today_str,
            })
            state["cash"] -= size_cash  # margin reserved

    # Save signal report
    signals_dir = _get_signals_dir()
    signal_path = os.path.join(signals_dir, f"{today_str}.json")
    with open(signal_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save updated state
    save_state(state)

    return report, state
