"""Performance metrics for backtesting outputs."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def calculate_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Calculate key backtest metrics from result DataFrame."""
    if results.empty:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0.0,
        }

    df = results.copy()
    if "equity" not in df.columns:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0.0,
        }

    equity = pd.to_numeric(df["equity"], errors="coerce").fillna(method="ffill").fillna(0.0)
    initial_equity = float(equity.iloc[0]) if len(equity) > 0 else 0.0
    final_equity = float(equity.iloc[-1]) if len(equity) > 0 else 0.0

    total_return = _safe_div(final_equity - initial_equity, initial_equity)

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_mean = float(returns.mean())
    ret_std = float(returns.std(ddof=0))
    sharpe_ratio = _safe_div(ret_mean, ret_std) * np.sqrt(365.25)

    rolling_peak = equity.cummax()
    drawdown = _safe_div((rolling_peak - equity).max(), rolling_peak.max()) if len(equity) > 0 else 0.0

    trades = df[df.get("trade_executed", False) == True].copy()  # noqa: E712
    trade_pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    num_trades = float(len(trades))

    win_rate = float((trade_pnl > 0).mean()) if len(trade_pnl) > 0 else 0.0
    gross_profit = float(trade_pnl[trade_pnl > 0].sum())
    gross_loss = abs(float(trade_pnl[trade_pnl < 0].sum()))
    profit_factor = _safe_div(gross_profit, gross_loss)

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "num_trades": num_trades,
    }
