"""
Test script: Force a simulated trade using the REAL strategy logic.
Bypasses BTC conditions but applies the actual ALT/BTC filter + ranking
to select the crypto "star" or "absente" du moment.

Usage:
    cd streamlit_app/momentum_BTC
    python test_force_trade.py
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime

# Setup imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.calc_indicators import compute_all_indicators
from execution.binance_executor import execute_signals
from monitoring.portfolio_tracker import update_monitoring
from signals.generate_signals import load_state, save_state


def _fetch_live_price(symbol):
    """Fetch current spot price from Binance API."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        print(f"      ⚠️ Could not fetch live price for {symbol}: {e}")
        return None


def _fmt_price(price):
    """Smart price formatting — handles micro-prices like SHIB."""
    if price == 0:
        return "$0"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.001:
        return f"${price:.6f}"
    else:
        return f"${price:.10f}"


def find_long_candidate(indicators, today_idx):
    """
    Apply the REAL altcoin filter for Long:
    ALT/BTC > SMA for 5 consecutive days → pick best 3D return.
    (BTC conditions are bypassed for this test.)
    """
    today = indicators["btc_ret_5d"].index[today_idx]
    alt_btc_cols = list(indicators["alt_btc_closes"].columns)

    filtered = []
    for sym in alt_btc_cols:
        if sym in indicators["alt_btc_above_sma_5d"].columns:
            if indicators["alt_btc_above_sma_5d"].at[today, sym]:
                usdt_sym = sym.replace("BTC", "USDT")
                if usdt_sym in indicators["ret_3d"].columns:
                    ret_val = indicators["ret_3d"].at[today, usdt_sym]
                    if pd.notna(ret_val):
                        filtered.append((usdt_sym, float(ret_val)))

    return filtered


def find_short_candidate(indicators, today_idx):
    """
    Apply the REAL altcoin filter for Short:
    ALT/BTC < SMA for 5 consecutive days → pick worst 3D return.
    """
    today = indicators["btc_ret_5d"].index[today_idx]
    alt_btc_cols = list(indicators["alt_btc_closes"].columns)

    filtered = []
    for sym in alt_btc_cols:
        if sym in indicators["alt_btc_below_sma_5d"].columns:
            if indicators["alt_btc_below_sma_5d"].at[today, sym]:
                usdt_sym = sym.replace("BTC", "USDT")
                if usdt_sym in indicators["ret_3d"].columns:
                    ret_val = indicators["ret_3d"].at[today, usdt_sym]
                    if pd.notna(ret_val):
                        filtered.append((usdt_sym, float(ret_val)))

    return filtered


def force_test_trade():
    """Force a simulated trade using real strategy filters."""

    print("=" * 60)
    print("🧪 TEST: Forcing trade with REAL strategy selection logic")
    print("=" * 60)

    # Load indicators
    print("\n📊 Loading indicators...")
    indicators = compute_all_indicators(sma_period=50, roll_lookback=600)

    today_idx = len(indicators["btc_close"]) - 1
    today = indicators["btc_close"].index[today_idx]
    today_str = str(today.date()) if hasattr(today, 'date') else str(today)[:10]

    btc_close = float(indicators["btc_close"].iloc[today_idx])
    btc_sma = float(indicators["btc_sma"].iloc[today_idx])
    btc_ret = float(indicators["btc_ret_5d"].iloc[today_idx])

    print(f"\n📅 Date: {today_str}")
    print(f"📈 BTC Close: ${btc_close:,.0f} | SMA: ${btc_sma:,.0f} | 5D Ret: {btc_ret:.2%}")
    print(f"⚠️  BTC conditions BYPASSED for this test\n")

    # ── LONG candidate selection ──
    print("─" * 50)
    long_filtered = find_long_candidate(indicators, today_idx)

    if long_filtered:
        long_filtered.sort(key=lambda x: x[1], reverse=True)
        print(f"🟢 LONG SIGNAL PREVIEW — {len(long_filtered)} alts pass ALT/BTC > SMA (5d):")
        for sym, ret in long_filtered:
            marker = "⭐" if sym == long_filtered[0][0] else "  "
            print(f"   {marker} {sym:<12s}  3D ret: {ret:+.2%}")
        star = long_filtered[0]
        print(f"   → Crypto ⭐ star du moment: {star[0]} ({star[1]:+.2%})")
    else:
        print("🟢 LONG: Aucune alt ne passe le filtre ALT/BTC > SMA (5d)")
        star = None

    # ── SHORT candidate selection ──
    print()
    short_filtered = find_short_candidate(indicators, today_idx)

    if short_filtered:
        short_filtered.sort(key=lambda x: x[1])
        print(f"🔴 SHORT SIGNAL PREVIEW — {len(short_filtered)} alts pass ALT/BTC < SMA (5d):")
        for sym, ret in short_filtered:
            marker = "💀" if sym == short_filtered[0][0] else "  "
            print(f"   {marker} {sym:<12s}  3D ret: {ret:+.2%}")
        absente = short_filtered[0]
        print(f"   → Crypto 💀 absente du moment: {absente[0]} ({absente[1]:+.2%})")
    else:
        print("🔴 SHORT: Aucune alt ne passe le filtre ALT/BTC < SMA (5d)")
        absente = None

    print("─" * 50)

    # ── Execute the best available trade ──
    if not star and not absente:
        print("\n❌ Aucune crypto ne passe les filtres. Rien à tester.")
        return

    state = load_state()
    print(f"\n💰 Cash avant trade: ${state['cash']:,.2f}")

    exits = []

    # ── STEP 0: Close any existing positions first ──
    if state["positions"]:
        print(f"\n🚪 Clôture de {len(state['positions'])} position(s) existante(s):")
        for pos in state["positions"]:
            sym = pos["symbol"]
            side = pos["side"]
            entry_price = pos.get("entry_price", 0)
            qty = pos.get("qty", 0)

            # Fetch LIVE price from Binance for real P&L
            current_price = _fetch_live_price(sym)
            if current_price is None:
                # Fallback to CSV daily close
                if sym in indicators["alt_usdt_closes"].columns:
                    current_price = float(indicators["alt_usdt_closes"].at[today, sym])
                else:
                    current_price = entry_price

            if side == "long":
                pnl = (current_price - entry_price) * qty
            else:
                pnl = (entry_price - current_price) * qty

            # Return cash (entry amount + P&L)
            returned = entry_price * qty + pnl
            state["cash"] += returned

            exits.append({
                "symbol": sym,
                "side": side,
                "reason": "test_force_close",
                "qty": qty,
            })

            print(f"   ✂️ CLOSED {side.upper()} {sym}")
            print(f"      Entry: {_fmt_price(entry_price)} → Live: {_fmt_price(current_price)} | P&L: ${pnl:+,.4f}")

        state["positions"] = []
        state["underperf_streaks"] = {}
        save_state(state)
        print(f"   💰 Cash après clôture: ${state['cash']:,.2f}")
    else:
        print("\n📭 Aucune position ouverte à clôturer")

    entries = []

    # Force Long if we have a star
    if star:
        test_sym, test_ret = star
        test_price = _fetch_live_price(test_sym)
        if test_price is None:
            test_price = float(indicators["alt_usdt_closes"].at[today, test_sym])
        size_cash = state["cash"] * 0.25
        qty = size_cash / test_price if test_price > 0 else 0

        entries.append({
            "symbol": test_sym,
            "side": "long",
            "ret_3d": test_ret,
            "entry_price": test_price,
            "qty": qty,
            "entry_date": today_str,
            "size_cash": size_cash,
        })

        state["positions"].append({
            "symbol": test_sym, "side": "long",
            "entry_price": test_price, "qty": qty, "entry_date": today_str,
        })
        state["cash"] -= size_cash
        print(f"\n🟢 FORCED LONG: {test_sym} @ {_fmt_price(test_price)} | Qty: {qty:.6f} | ${size_cash:,.2f}")

    # Force Short if we have an absente
    if absente:
        test_sym, test_ret = absente
        test_price = _fetch_live_price(test_sym)
        if test_price is None:
            test_price = float(indicators["alt_usdt_closes"].at[today, test_sym])
        size_cash = state["cash"] * 0.25
        qty = size_cash / test_price if test_price > 0 else 0

        entries.append({
            "symbol": test_sym,
            "side": "short",
            "ret_3d": test_ret,
            "entry_price": test_price,
            "qty": qty,
            "entry_date": today_str,
            "size_cash": size_cash,
        })

        state["positions"].append({
            "symbol": test_sym, "side": "short",
            "entry_price": test_price, "qty": qty, "entry_date": today_str,
        })
        state["cash"] -= size_cash
        print(f"🔴 FORCED SHORT: {test_sym} @ {_fmt_price(test_price)} | Qty: {qty:.6f} | ${size_cash:,.2f}")

    # Build signal report
    fake_report = {
        "date": today_str,
        "exits": exits,
        "entries": entries,
        "btc_close": btc_close,
        "btc_sma": btc_sma,
        "btc_ret_5d": btc_ret,
    }

    # Execute (dry-run)
    print("\n" + "=" * 60)
    print("⚡ Executing orders (DRY RUN)...")
    print("=" * 60)
    execution_log = execute_signals(fake_report, dry_run=True)

    # Save state
    save_state(state)

    # Update monitoring
    print("\n" + "=" * 60)
    print("📊 Updating monitoring...")
    print("=" * 60)
    nav = update_monitoring(fake_report, execution_log, indicators)

    # Show results
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

    state = load_state()
    print(f"💰 Cash restant: ${state['cash']:,.2f}")
    print(f"📦 Positions ouvertes: {len(state['positions'])}")
    for pos in state['positions']:
        print(f"   → {pos['side'].upper()} {pos['symbol']} @ ${pos['entry_price']:,.4f}")
    print(f"📈 NAV: ${nav:,.2f}")

    # Check files
    project_root = os.path.dirname(os.path.dirname(current_dir))
    crypto_dir = os.path.join(project_root, "data", "crypto")
    state_path = os.path.join(crypto_dir, "state.json")

    print(f"\n📁 Fichiers mis à jour:")
    for subdir in ["signals", "execution_logs"]:
        d = os.path.join(crypto_dir, subdir)
        if os.path.exists(d):
            files = sorted(os.listdir(d))[-3:]
            print(f"   {subdir}/: {files}")
    print(f"   state.json: ✅")
    print(f"   nav_history.csv: ✅")

    print(f"\n🧹 Pour reset après test: rm {state_path}")


if __name__ == "__main__":
    force_test_trade()
