#!/usr/bin/env python3
"""Check whether the configured live universe qualifies and can be traded at IBKR.

This script is non-destructive: it never places a real order. With --what-if it
uses IBKR whatIfOrder for a BUY 1 test, which is the closest automated check for
account-level permissions, KID/PRIIPs blocks, and margin warnings.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List

import nest_asyncio
from ib_insync import IB, Contract, MarketOrder

from ibkr.config import ACCOUNT_ID, CURRENT_PORT, HOST
from ibkr.live_universe import LIVE_CANDIDATES, LIVE_UNIVERSE, selected_contract_details, selected_etf_mapping

nest_asyncio.apply()


def _contract_for(asset: str, mapping: Dict[str, str], details: Dict[str, Dict[str, object]]) -> Contract:
    symbol = mapping[asset]
    meta = details.get(symbol, {})
    kwargs = {
        "symbol": meta.get("symbol", symbol),
        "secType": meta.get("secType", "STK"),
        "exchange": meta.get("exchange", "SMART"),
        "currency": meta.get("currency", "EUR"),
    }
    if meta.get("primaryExchange"):
        kwargs["primaryExchange"] = meta["primaryExchange"]
    if meta.get("secIdType"):
        kwargs["secIdType"] = meta["secIdType"]
        kwargs["secId"] = meta["secId"]
    return Contract(**kwargs)


def run_check(include_cfds: bool, what_if: bool, output_csv: str | None) -> List[Dict[str, object]]:
    ib = IB()
    client_id = random.randint(5000, 9999)
    ib.connect(HOST, CURRENT_PORT, clientId=client_id, timeout=30)
    ib.reqMarketDataType(3)

    mapping = selected_etf_mapping()
    details = selected_contract_details()
    rows: List[Dict[str, object]] = []

    try:
        for asset, meta in LIVE_UNIVERSE.items():
            if meta.get("sec_type") == "CFD" and not include_cfds:
                continue

            contract = _contract_for(asset, mapping, details)
            qualified = ib.qualifyContracts(contract)
            if (not qualified or not contract.conId) and getattr(contract, "primaryExchange", None):
                contract.exchange = contract.primaryExchange
                qualified = ib.qualifyContracts(contract)

            row: Dict[str, object] = {
                "asset": asset,
                "model_proxy": meta.get("model_proxy"),
                "ibkr_symbol": mapping[asset],
                "yahoo_ticker": meta.get("yahoo_ticker"),
                "isin": meta.get("isin"),
                "currency": meta.get("currency"),
                "status": "qualified" if qualified and contract.conId else "not_qualified",
                "conId": getattr(contract, "conId", None),
                "qualified_symbol": getattr(contract, "symbol", None),
                "qualified_exchange": getattr(contract, "exchange", None),
                "qualified_currency": getattr(contract, "currency", None),
                "what_if_ok": None,
                "what_if_warning": None,
                "what_if_error": None,
            }

            if qualified and contract.conId and what_if:
                try:
                    order = MarketOrder("BUY", 1, tif="DAY", account=ACCOUNT_ID or "")
                    state = ib.whatIfOrder(contract, order)
                    warning = getattr(state, "warningText", "") or ""
                    row["what_if_ok"] = not bool(warning)
                    row["what_if_warning"] = warning
                    row["init_margin_change"] = getattr(state, "initMarginChange", None)
                    row["maint_margin_change"] = getattr(state, "maintMarginChange", None)
                    row["commission"] = getattr(state, "commission", None)
                except Exception as exc:
                    row["what_if_ok"] = False
                    row["what_if_error"] = f"{type(exc).__name__}: {exc}"

            rows.append(row)
            warning = row["what_if_warning"] or row["what_if_error"] or ""
            print(
                f"{row['asset']:14s} {row['ibkr_symbol']:8s} "
                f"{row['status']:14s} conId={row['conId']} "
                f"whatIf={row['what_if_ok']} warning={warning}"
            )

        if LIVE_CANDIDATES:
            print("\nCandidate alternatives to test manually if a primary fails:")
            for asset, candidates in LIVE_CANDIDATES.items():
                print(f"  {asset}:")
                for cand in candidates:
                    print(f"    - {cand['ibkr_symbol']} / {cand['yahoo_ticker']} / {cand['isin']} - {cand['note']}")

    finally:
        ib.disconnect()

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {output_csv}")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-cfds", action="store_true")
    parser.add_argument("--what-if", action="store_true", help="Run IBKR what-if BUY 1 permission/margin checks; no real order is placed.")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    run_check(include_cfds=args.include_cfds, what_if=args.what_if, output_csv=args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
