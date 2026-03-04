"""
Close all open positions and log the exits.
Usage: cd streamlit_app/momentum_BTC && python close_positions.py
"""
import os, sys, json, requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.calc_indicators import compute_all_indicators
from execution.binance_executor import execute_signals
from monitoring.portfolio_tracker import update_monitoring
from signals.generate_signals import load_state, save_state


def _live_price(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except:
        return None


def _fmt(p):
    if p >= 1: return f"${p:,.4f}"
    elif p >= 0.001: return f"${p:.6f}"
    else: return f"${p:.10f}"


def close_all():
    state = load_state()
    positions = state.get("positions", [])

    if not positions:
        print("📭 Aucune position ouverte.")
        return

    print("=" * 60)
    print(f"🚪 Clôture de {len(positions)} position(s)...")
    print("=" * 60)

    indicators = compute_all_indicators(sma_period=50, roll_lookback=600)
    today_idx = len(indicators["btc_close"]) - 1
    today = indicators["btc_close"].index[today_idx]
    today_str = str(today.date()) if hasattr(today, 'date') else str(today)[:10]

    exits = []
    for pos in positions:
        sym = pos["symbol"]
        side = pos["side"]
        entry_price = pos.get("entry_price", 0)
        qty = pos.get("qty", 0)

        live = _live_price(sym)
        current_price = live if live else entry_price

        pnl = (current_price - entry_price) * qty if side == "long" else (entry_price - current_price) * qty
        state["cash"] += entry_price * qty + pnl

        exits.append({"symbol": sym, "side": side, "reason": "manual_force_close", "qty": qty})

        print(f"\n   ✂️ CLOSED {side.upper()} {sym}")
        print(f"      Entry: {_fmt(entry_price)} → Live: {_fmt(current_price)}")
        print(f"      Qty: {qty:.6f} | P&L: ${pnl:+,.4f}")

    state["positions"] = []
    state["underperf_streaks"] = {}
    save_state(state)

    report = {"date": today_str, "exits": exits, "entries": [], "btc_close": float(indicators["btc_close"].iloc[today_idx]),
              "btc_sma": float(indicators["btc_sma"].iloc[today_idx]), "btc_ret_5d": float(indicators["btc_ret_5d"].iloc[today_idx])}

    execution_log = execute_signals(report, dry_run=True)
    nav = update_monitoring(report, execution_log, indicators)

    print(f"\n{'=' * 60}")
    print(f"✅ Toutes positions clôturées")
    print(f"💰 Cash: ${state['cash']:,.2f} | NAV: ${nav:,.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    close_all()
