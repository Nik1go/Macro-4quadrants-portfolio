#!/usr/bin/env python3
"""CLI report for IBKR/live tracking versus the model backtest."""

import argparse
import os
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ROOT = _project_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ibkr.tracking_audit import build_tracking_audit, save_tracking_audit  # noqa: E402


def _fmt_pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=os.path.join(ROOT, "data", "US"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    audit = build_tracking_audit(args.base_dir)
    summary = audit.summary

    print("IBKR TRACKING AUDIT")
    print("=" * 60)
    print(f"Runs:                 {summary['runs_count']}")
    print(f"Orders:               {summary['orders_count']}")
    print(f"Failed runs:          {summary['failed_runs_count']}")
    print(f"Unconfirmed orders:   {summary['unconfirmed_orders_count']}")
    print(f"NAV window:           {summary['first_nav_date']} -> {summary['last_nav_date']}")
    print(f"Live return:          {_fmt_pct(summary['live_return'])}")
    print(f"Model return:         {_fmt_pct(summary['model_return'])}")
    print(f"Tracking gap:         {_fmt_pct(summary['tracking_gap'])}")
    print(f"Latest L1 drift:      {_fmt_pct(summary['latest_weight_drift_l1'])}")
    print(f"Latest run success:   {summary['latest_run_success']}")
    if summary.get("latest_run_error"):
        print(f"Latest run error:     {summary['latest_run_error']}")

    if not audit.unconfirmed_orders.empty:
        print("\nRECENT UNCONFIRMED ORDERS")
        cols = ["timestamp", "asset", "action", "status", "shares", "filled", "error"]
        print(audit.unconfirmed_orders[cols].tail(20).to_string(index=False))

    if not audit.failed_runs.empty:
        print("\nRECENT FAILED RUNS")
        cols = ["timestamp", "quadrant", "error", "portfolio_value", "orders_count"]
        print(audit.failed_runs[cols].tail(20).to_string(index=False))

    if not args.no_save:
        output_dir = args.output_dir or os.path.join(args.base_dir, "tracking_audit")
        save_tracking_audit(audit, output_dir)
        print(f"\nSaved audit files to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
