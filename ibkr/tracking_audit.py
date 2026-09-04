"""
IBKR tracking audit utilities.

The backtest and the paper account are allowed to disagree for small market
microstructure reasons, but this module makes the disagreement explicit:
target weights, current weights, NAV, unfilled orders, and tracking gap are all
computed from persisted files so the dashboard still works when IB Gateway is
down.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


FINAL_FILL_STATUSES = {"Filled"}
NON_FINAL_STATUSES = {
    "ApiPending",
    "PendingSubmit",
    "PreSubmitted",
    "Submitted",
    "PendingCancel",
}


@dataclass
class TrackingAudit:
    summary: Dict[str, Any]
    nav: pd.DataFrame
    orders: pd.DataFrame
    tracking: pd.DataFrame
    drift: pd.DataFrame
    failed_runs: pd.DataFrame
    unconfirmed_orders: pd.DataFrame


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            data["_path"] = path
            return data
    except Exception:
        return None


def _execution_logs(log_dir: str) -> List[Dict[str, Any]]:
    logs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        payload = _load_json(path)
        if payload is not None:
            logs.append(payload)
    return logs


def _as_weight_dict(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    clean = {}
    for key, raw in value.items():
        val = _safe_float(raw)
        if val is not None:
            clean[str(key)] = val
    return clean


def _l1_drift(current: Dict[str, float], target: Dict[str, float]) -> Optional[float]:
    if not current or not target:
        return None
    keys = set(current).union(target)
    return sum(abs(current.get(k, 0.0) - target.get(k, 0.0)) for k in keys)


def _order_is_confirmed(order: Dict[str, Any]) -> bool:
    status = str(order.get("status") or "")
    shares = _safe_float(order.get("shares")) or 0.0
    filled = _safe_float(order.get("filled")) or 0.0
    if status in FINAL_FILL_STATUSES and (shares <= 0 or filled >= shares):
        return True
    return False


def load_backtest(backtest_path: str) -> pd.DataFrame:
    df = pd.read_csv(backtest_path, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df


def parse_execution_logs(log_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []

    for payload in _execution_logs(log_dir):
        ts = pd.to_datetime(payload.get("timestamp"), errors="coerce")
        target = _as_weight_dict(payload.get("target_weights"))
        current = _as_weight_dict(payload.get("current_weights"))
        post_weights = _as_weight_dict(payload.get("post_execution_weights"))
        tracking_weights = post_weights or current
        pre_value = _safe_float(payload.get("portfolio_value"))
        post_value = _safe_float(payload.get("post_execution_portfolio_value"))
        tracking_value = post_value if post_value is not None else pre_value
        drift = _l1_drift(tracking_weights, target)

        orders = payload.get("orders") or []
        if not isinstance(orders, list):
            orders = []

        run_rows.append(
            {
                "timestamp": ts,
                "date": ts.normalize() if not pd.isna(ts) else pd.NaT,
                "path": payload.get("_path"),
                "success": bool(payload.get("success")),
                "error": payload.get("error"),
                "dry_run": bool(payload.get("dry_run")),
                "quadrant": payload.get("quadrant"),
                "portfolio_value": pre_value,
                "post_execution_portfolio_value": post_value,
                "tracking_portfolio_value": tracking_value,
                "base_currency": payload.get("base_currency"),
                "target_weights": target,
                "current_weights": current,
                "post_execution_weights": post_weights,
                "tracking_weights": tracking_weights,
                "weight_drift_l1": drift,
                "orders_count": len(orders),
            }
        )

        for order in orders:
            if not isinstance(order, dict):
                continue
            status = str(order.get("status") or "")
            shares = _safe_float(order.get("shares")) or 0.0
            filled = _safe_float(order.get("filled")) or 0.0
            order_rows.append(
                {
                    "timestamp": ts,
                    "date": ts.normalize() if not pd.isna(ts) else pd.NaT,
                    "path": payload.get("_path"),
                    "run_success": bool(payload.get("success")),
                    "asset": order.get("asset"),
                    "symbol": order.get("symbol"),
                    "action": order.get("action"),
                    "status": status,
                    "shares": shares,
                    "filled": filled,
                    "avgFillPrice": _safe_float(order.get("avgFillPrice")),
                    "estimated_value": _safe_float(order.get("estimated_value")),
                    "weight_delta": _safe_float(order.get("weight_delta")),
                    "error": order.get("error"),
                    "confirmed_fill": _order_is_confirmed(order),
                    "non_final_status": status in NON_FINAL_STATUSES,
                }
            )

    runs = pd.DataFrame(run_rows)
    orders = pd.DataFrame(order_rows)
    if not runs.empty:
        runs = runs.sort_values("timestamp")
    if not orders.empty:
        orders = orders.sort_values("timestamp")
    return runs, orders


def build_tracking_curve(
    backtest: pd.DataFrame,
    runs: pd.DataFrame,
    wealth_col: str = "wealth",
    return_col: str = "portfolio_return",
    model_source: str = "model_us_proxy",
) -> pd.DataFrame:
    value_col = "tracking_portfolio_value" if "tracking_portfolio_value" in runs.columns else "portfolio_value"
    if backtest.empty or runs.empty or value_col not in runs.columns or wealth_col not in backtest.columns:
        return pd.DataFrame()

    nav = runs.dropna(subset=[value_col]).copy()
    if nav.empty:
        return pd.DataFrame()

    nav = nav.sort_values("timestamp").drop_duplicates(subset=["date"], keep="last")
    cols = ["date", wealth_col]
    for optional in [return_col, "smooth_quadrant"]:
        if optional in backtest.columns and optional not in cols:
            cols.append(optional)
    bt = backtest[cols].copy()
    bt["date"] = pd.to_datetime(bt["date"]).dt.normalize()
    bt = bt.rename(columns={wealth_col: "model_wealth", return_col: "model_period_return"})

    merged = nav[["date", "timestamp", value_col, "quadrant", "weight_drift_l1"]].rename(
        columns={value_col: "live_nav"}
    ).merge(
        bt,
        on="date",
        how="left",
    )
    merged = merged.dropna(subset=["model_wealth"])
    if merged.empty:
        return merged

    first_live = merged["live_nav"].iloc[0]
    first_bt = merged["model_wealth"].iloc[0]
    merged["live_index"] = merged["live_nav"] / first_live * 1000.0
    merged["model_index"] = merged["model_wealth"] / first_bt * 1000.0
    merged["live_return"] = merged["live_nav"] / first_live - 1.0
    merged["model_return"] = merged["model_wealth"] / first_bt - 1.0
    merged["tracking_gap"] = merged["live_return"] - merged["model_return"]
    merged["model_source"] = model_source
    return merged


def build_drift_table(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    rows = []
    source = runs.dropna(subset=["timestamp"]).copy()
    source = source[source["current_weights"].map(bool) & source["target_weights"].map(bool)]
    for _, row in source.iterrows():
        current = row["current_weights"]
        target = row["target_weights"]
        for asset in sorted(set(current).union(target)):
            cur = current.get(asset, 0.0)
            tar = target.get(asset, 0.0)
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "date": row["date"],
                    "asset": asset,
                    "current_weight": cur,
                    "target_weight": tar,
                    "diff": cur - tar,
                    "abs_diff": abs(cur - tar),
                }
            )
    return pd.DataFrame(rows)


def _latest_dict(series: Iterable[Dict[str, float]]) -> Dict[str, float]:
    values = list(series)
    for value in reversed(values):
        if value:
            return value
    return {}


def build_tracking_audit(base_dir: str) -> TrackingAudit:
    backtest_dir = os.path.join(base_dir, "backtest_results")
    backtest_path = os.path.join(backtest_dir, "backtest_timeseries.csv")
    live_backtest_path = os.path.join(backtest_dir, "backtest_ibkr_live_timeseries.csv")
    log_dir = os.path.join(base_dir, "execution_logs")

    backtest = load_backtest(backtest_path) if os.path.exists(backtest_path) else pd.DataFrame()
    live_backtest = load_backtest(live_backtest_path) if os.path.exists(live_backtest_path) else pd.DataFrame()
    runs, orders = parse_execution_logs(log_dir)
    if not live_backtest.empty and "ibkr_live_wealth" in live_backtest.columns:
        tracking = build_tracking_curve(
            live_backtest,
            runs,
            wealth_col="ibkr_live_wealth",
            return_col="ibkr_live_return",
            model_source="ibkr_live_compatible",
        )
    else:
        tracking = build_tracking_curve(backtest, runs, model_source="model_us_proxy")
    drift = build_drift_table(runs)

    failed_runs = runs[runs["success"] == False].copy() if not runs.empty else pd.DataFrame()
    if not orders.empty:
        unconfirmed = orders[orders["confirmed_fill"] == False].copy()
    else:
        unconfirmed = pd.DataFrame()

    summary: Dict[str, Any] = {
        "runs_count": int(len(runs)),
        "orders_count": int(len(orders)),
        "failed_runs_count": int(len(failed_runs)),
        "unconfirmed_orders_count": int(len(unconfirmed)),
        "first_nav_date": None,
        "last_nav_date": None,
        "live_return": None,
        "model_return": None,
        "tracking_gap": None,
        "latest_weight_drift_l1": None,
        "latest_run_success": None,
        "latest_run_error": None,
        "latest_target_weights": {},
        "latest_current_weights": {},
        "model_source": None,
    }

    if not runs.empty:
        latest = runs.iloc[-1]
        summary["latest_run_success"] = bool(latest.get("success"))
        summary["latest_run_error"] = latest.get("error")
        summary["latest_target_weights"] = _latest_dict(runs["target_weights"])
        summary["latest_current_weights"] = _latest_dict(runs["tracking_weights"] if "tracking_weights" in runs.columns else runs["current_weights"])
        summary["latest_weight_drift_l1"] = _safe_float(latest.get("weight_drift_l1"))

    if not tracking.empty:
        first = tracking.iloc[0]
        last = tracking.iloc[-1]
        summary.update(
            {
                "first_nav_date": first["date"].date().isoformat(),
                "last_nav_date": last["date"].date().isoformat(),
                "live_return": float(last["live_return"]),
                "model_return": float(last["model_return"]),
                "tracking_gap": float(last["tracking_gap"]),
                "latest_weight_drift_l1": _safe_float(last.get("weight_drift_l1")),
                "model_source": last.get("model_source"),
            }
        )

    nav_value_col = "tracking_portfolio_value" if "tracking_portfolio_value" in runs.columns else "portfolio_value"
    nav = runs.dropna(subset=[nav_value_col]).copy() if not runs.empty and nav_value_col in runs.columns else pd.DataFrame()
    if not nav.empty:
        nav = nav.sort_values("timestamp").drop_duplicates(subset=["date"], keep="last")
        nav["live_nav"] = nav[nav_value_col]

    return TrackingAudit(
        summary=summary,
        nav=nav,
        orders=orders,
        tracking=tracking,
        drift=drift,
        failed_runs=failed_runs,
        unconfirmed_orders=unconfirmed,
    )


def save_tracking_audit(audit: TrackingAudit, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(audit.summary, handle, indent=2, default=str)

    for name in ["nav", "orders", "tracking", "drift", "failed_runs", "unconfirmed_orders"]:
        df = getattr(audit, name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

