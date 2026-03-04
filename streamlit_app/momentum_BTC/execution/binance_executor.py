"""
Binance Executor Module for Crypto Momentum Strategy.
Handles order placement via Binance Futures Testnet API.

Testnet URLs:
- Futures: https://testnet.binancefuture.com
- API: https://testnet.binancefuture.com/fapi/v1

For now, this module logs orders locally (dry-run mode).
To activate live paper trading, set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET
in environment variables or .env file.
"""

import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime
from urllib.parse import urlencode


# Binance Futures Testnet endpoints
TESTNET_BASE = "https://testnet.binancefuture.com"
TESTNET_ORDER_URL = f"{TESTNET_BASE}/fapi/v1/order"
TESTNET_ACCOUNT_URL = f"{TESTNET_BASE}/fapi/v2/account"


def _get_api_credentials():
    """Load API credentials from environment or .env file."""
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
    return api_key, api_secret


def _sign_request(params, secret):
    """Sign request parameters with HMAC SHA256."""
    query_string = urlencode(params)
    signature = hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return signature


def _get_execution_logs_dir():
    """Resolve execution logs directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    logs_dir = os.path.join(project_root, "data", "crypto", "execution_logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def place_order(symbol, side, quantity, order_type="MARKET", dry_run=True):
    """
    Place an order on Binance Futures Testnet.
    
    Args:
        symbol: Trading pair (e.g., "ETHUSDT")
        side: "BUY" or "SELL"
        quantity: Amount to trade
        order_type: "MARKET" or "LIMIT"
        dry_run: If True, only log the order without sending to Binance
        
    Returns:
        dict with order result
    """
    order_info = {
        "symbol": symbol,
        "side": side,
        "quantity": round(quantity, 6),
        "type": order_type,
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
    }

    if dry_run:
        order_info["status"] = "DRY_RUN"
        order_info["message"] = "Order logged locally (dry-run mode)"
        print(f"  🔸 DRY RUN: {side} {quantity:.6f} {symbol}")
        return order_info

    # Live Testnet execution
    api_key, api_secret = _get_api_credentials()
    if not api_key or not api_secret:
        order_info["status"] = "ERROR"
        order_info["message"] = "Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET"
        print(f"  ❌ Missing API credentials for {symbol}")
        return order_info

    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": round(quantity, 6),
        "timestamp": int(time.time() * 1000),
    }
    params["signature"] = _sign_request(params, api_secret)

    headers = {"X-MBX-APIKEY": api_key}

    try:
        r = requests.post(TESTNET_ORDER_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        result = r.json()
        order_info["status"] = "FILLED"
        order_info["response"] = result
        print(f"  ✅ ORDER FILLED: {side} {quantity:.6f} {symbol}")
    except Exception as e:
        order_info["status"] = "ERROR"
        order_info["message"] = str(e)
        print(f"  ❌ ORDER FAILED: {side} {quantity:.6f} {symbol} — {e}")

    return order_info


def execute_signals(signal_report, dry_run=True):
    """
    Execute all signals from a signal report.
    
    Args:
        signal_report: dict from generate_signals.generate_daily_signals()
        dry_run: If True, only log orders without sending to Binance
        
    Returns:
        execution_log: dict with all order results
    """
    execution_log = {
        "date": signal_report["date"],
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "orders": [],
    }

    # Process exits
    for ex in signal_report.get("exits", []):
        symbol = ex["symbol"]
        side_action = "SELL" if ex["side"] == "long" else "BUY"  # Close position
        qty = ex.get("qty", 0)

        order = place_order(symbol, side_action, qty, dry_run=dry_run)
        order["signal_type"] = "exit"
        order["reason"] = ex.get("reason", "unknown")
        execution_log["orders"].append(order)

    # Process entries
    for entry in signal_report.get("entries", []):
        symbol = entry["symbol"]
        side_action = "BUY" if entry["side"] == "long" else "SELL"
        qty = entry.get("qty", 0)

        order = place_order(symbol, side_action, qty, dry_run=dry_run)
        order["signal_type"] = "entry"
        order["position_side"] = entry["side"]
        execution_log["orders"].append(order)

    # Save execution log
    logs_dir = _get_execution_logs_dir()
    log_path = os.path.join(logs_dir, f"{signal_report['date']}.json")
    with open(log_path, "w") as f:
        json.dump(execution_log, f, indent=2, default=str)

    n_orders = len(execution_log["orders"])
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"  📝 Execution log saved: {n_orders} orders ({mode})")

    return execution_log
