"""
Portfolio Tracker / Monitoring Module for Crypto Momentum Strategy.
Tracks portfolio state, computes NAV, and provides data for Streamlit dashboard.
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime


def _get_crypto_data_dir():
    """Resolve crypto data directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return os.path.join(project_root, "data", "crypto")


def load_state():
    """Load current portfolio state."""
    data_dir = _get_crypto_data_dir()
    state_path = os.path.join(data_dir, "state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"positions": [], "cash": 10000.0, "initial_cash": 10000.0}


def compute_nav(state, current_prices):
    """
    Compute current Net Asset Value.
    
    Args:
        state: portfolio state dict
        current_prices: dict of {symbol: current_price}
        
    Returns:
        float: total NAV (cash + position values)
    """
    nav = state.get("cash", 0)
    for pos in state.get("positions", []):
        symbol = pos["symbol"]
        qty = pos.get("qty", 0)
        entry_price = pos.get("entry_price", 0)
        current_price = current_prices.get(symbol, entry_price)

        if pos["side"] == "long":
            # Long P&L
            pnl = (current_price - entry_price) * qty
        else:
            # Short P&L (profit when price drops)
            pnl = (entry_price - current_price) * qty
        
        position_value = entry_price * qty + pnl
        nav += position_value

    return nav


def update_monitoring(signal_report, execution_log, indicators=None):
    """
    Update monitoring data after signal generation and execution.
    Appends NAV history entry and saves summary.
    
    Args:
        signal_report: dict from generate_signals
        execution_log: dict from binance_executor
        indicators: dict from calc_indicators (optional, for current prices)
    """
    data_dir = _get_crypto_data_dir()
    state = load_state()

    # Get current prices for NAV calculation
    current_prices = {}
    if indicators and "alt_usdt_closes" in indicators:
        closes = indicators["alt_usdt_closes"]
        if not closes.empty:
            last_row = closes.iloc[-1]
            for col in closes.columns:
                if pd.notna(last_row[col]):
                    current_prices[col] = float(last_row[col])

    nav = compute_nav(state, current_prices)

    # Append to NAV history
    nav_history_path = os.path.join(data_dir, "nav_history.csv")
    nav_entry = {
        "date": signal_report["date"],
        "nav": round(nav, 2),
        "cash": round(state.get("cash", 0), 2),
        "n_positions": len(state.get("positions", [])),
        "n_exits": len(signal_report.get("exits", [])),
        "n_entries": len(signal_report.get("entries", [])),
        "btc_close": signal_report.get("btc_close"),
    }

    if os.path.exists(nav_history_path):
        nav_df = pd.read_csv(nav_history_path)
        nav_df = pd.concat([nav_df, pd.DataFrame([nav_entry])], ignore_index=True)
        nav_df = nav_df.drop_duplicates(subset=["date"], keep="last")
    else:
        nav_df = pd.DataFrame([nav_entry])

    nav_df.to_csv(nav_history_path, index=False)
    print(f"  📊 NAV updated: ${nav:,.2f} ({signal_report['date']})")

    return nav


def get_trade_history(limit=50):
    """
    Load recent trade history from execution logs.
    Returns a DataFrame of recent trades.
    """
    data_dir = _get_crypto_data_dir()
    logs_dir = os.path.join(data_dir, "execution_logs")

    if not os.path.exists(logs_dir):
        return pd.DataFrame()

    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.json")))[-limit:]
    trades = []

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                log = json.load(f)
            for order in log.get("orders", []):
                trades.append({
                    "date": log["date"],
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "signal_type": order.get("signal_type"),
                    "position_side": order.get("position_side", ""),
                    "reason": order.get("reason", ""),
                    "status": order.get("status"),
                })
        except Exception:
            continue

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def get_nav_history():
    """Load full NAV history as DataFrame."""
    data_dir = _get_crypto_data_dir()
    nav_path = os.path.join(data_dir, "nav_history.csv")
    if os.path.exists(nav_path):
        return pd.read_csv(nav_path, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "nav", "cash", "n_positions"])
