#!/usr/bin/env python3
"""Build an IBKR live-compatible backtest curve from existing model weights.

This does not change the model, quadrants, trend overlay, or weights. It only
re-prices the final *_weight columns with the instruments that the IBKR paper
account can realistically trade: UCITS/ETC tickers plus FX conversion to the
account base currency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ibkr.live_universe import LIVE_UNIVERSE, selected_live_rows, selected_ter, selected_yfinance_mapping  # noqa: E402

TRANSACTION_COST = 0.0010
TRADING_DAYS = 252


def _download_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    data = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=7)).date().isoformat(),
        end=(end + pd.Timedelta(days=2)).date().isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if data.empty or "Close" not in data:
        return pd.Series(dtype=float, name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = ticker
    return close


def _load_forex(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.drop_duplicates(subset=["date"]).set_index("date").sort_index()


def _convert_to_base(
    asset: str,
    price: pd.Series,
    forex: pd.DataFrame,
    base_currency: str,
) -> Tuple[pd.Series, str]:
    if asset.startswith("USD_"):
        return price, "FX raw series"

    currency = str(LIVE_UNIVERSE.get(asset, {}).get("currency") or base_currency).upper()
    base_currency = base_currency.upper()
    if currency == base_currency:
        return price, currency

    if currency == "USD" and base_currency == "EUR" and "USD_EUR" in forex.columns:
        fx = forex["USD_EUR"].astype(float).reindex(price.index).ffill()
        converted = price * fx
        converted.name = price.name
        return converted, f"{currency}->{base_currency} via USD_EUR"

    if currency == "EUR" and base_currency == "USD" and "USD_EUR" in forex.columns:
        fx = forex["USD_EUR"].astype(float).reindex(price.index).ffill()
        converted = price / fx
        converted.name = price.name
        return converted, f"{currency}->{base_currency} via USD_EUR"

    return price, f"{currency} (unconverted; missing FX for {base_currency})"


def _calc_stats(returns: pd.Series, wealth: pd.Series) -> Dict[str, float]:
    clean = returns.dropna()
    if clean.empty or wealth.dropna().empty:
        return {}
    total_return = float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)
    vol = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(clean) > 1 else 0.0
    sharpe = float(clean.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0
    peak = wealth.cummax()
    max_dd = float(((wealth - peak) / peak).min())
    days = max((wealth.index[-1] - wealth.index[0]).days, 1)
    years = days / 365.25
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else total_return
    return {
        "ibkr_live_total_return": total_return,
        "ibkr_live_cagr": cagr,
        "ibkr_live_vol_annual": vol,
        "ibkr_live_sharpe_annual": sharpe,
        "ibkr_live_max_drawdown": max_dd,
    }


def build_live_compatible_backtest(
    backtest_csv: str,
    forex_parquet: str | None,
    initial_capital: float,
    base_currency: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    bt = pd.read_csv(backtest_csv, parse_dates=["date"])
    bt = bt.drop_duplicates(subset=["date"]).sort_values("date")
    bt["date"] = pd.to_datetime(bt["date"]).dt.tz_localize(None).dt.normalize()
    bt = bt.set_index("date")
    if bt.empty:
        raise ValueError(f"Empty backtest file: {backtest_csv}")

    weight_cols = [c for c in bt.columns if c.endswith("_weight") and not c.endswith("_base_weight")]
    assets = [c[:-7] for c in weight_cols]
    mapping = selected_yfinance_mapping()
    forex = _load_forex(forex_parquet)

    start, end = bt.index.min(), bt.index.max()
    prices = pd.DataFrame(index=bt.index)
    price_notes = []
    missing_assets = []

    for asset in assets:
        ticker = mapping.get(asset)
        if not ticker:
            missing_assets.append(asset)
            continue

        if ticker.endswith("=X"):
            if asset in forex.columns:
                raw = forex[asset].astype(float)
            else:
                raw = _download_close(ticker, start, end)
        else:
            raw = _download_close(ticker, start, end)

        if raw.empty:
            missing_assets.append(asset)
            price_notes.append({"asset": asset, "ticker": ticker, "status": "missing_price"})
            continue

        raw = raw.reindex(bt.index).ffill()
        converted, currency_note = _convert_to_base(asset, raw, forex, base_currency)
        prices[asset] = converted
        clean = converted.dropna()
        price_notes.append({
            "asset": asset,
            "ticker": ticker,
            "currency_note": currency_note,
            "first_price": float(clean.iloc[0]) if not clean.empty else None,
            "last_price": float(clean.iloc[-1]) if not clean.empty else None,
        })

    returns = prices.pct_change().fillna(0.0)
    out = bt.copy()

    ters = selected_ter()
    out["ibkr_live_return_gross"] = 0.0
    out["ibkr_live_missing_weight"] = 0.0
    for asset in assets:
        w_col = f"{asset}_weight"
        if asset in returns.columns:
            out[f"{asset}_ibkr_live_ret"] = returns[asset]
            out[f"{asset}_ibkr_live_price"] = prices[asset]
            out["ibkr_live_return_gross"] += out[w_col].fillna(0.0) * returns[asset].reindex(out.index).fillna(0.0)
        else:
            out["ibkr_live_missing_weight"] += out[w_col].fillna(0.0)

    turnover = out[weight_cols].diff().abs().sum(axis=1).fillna(0.0)
    out["ibkr_live_transaction_cost"] = turnover * TRANSACTION_COST

    out["ibkr_live_ter_cost"] = 0.0
    for asset in assets:
        out["ibkr_live_ter_cost"] += out[f"{asset}_weight"].fillna(0.0) * (ters.get(asset, 0.0) / TRADING_DAYS)

    out["ibkr_live_return"] = (
        out["ibkr_live_return_gross"]
        - out["ibkr_live_transaction_cost"]
        - out["ibkr_live_ter_cost"]
    )
    out["ibkr_live_wealth"] = initial_capital * (1.0 + out["ibkr_live_return"]).cumprod()

    stats = _calc_stats(out["ibkr_live_return"], out["ibkr_live_wealth"])
    missing_assets_unique = sorted(set(missing_assets))
    missing_active_assets = [
        asset for asset in missing_assets_unique
        if f"{asset}_weight" in out.columns and out[f"{asset}_weight"].fillna(0.0).abs().max() > 1e-9
    ]
    inactive_missing_assets = sorted(set(missing_assets_unique) - set(missing_active_assets))
    stats.update({
        "base_currency": base_currency.upper(),
        "start_date": out.index.min().date().isoformat(),
        "end_date": out.index.max().date().isoformat(),
        "missing_assets": ",".join(missing_active_assets),
        "inactive_missing_assets": ",".join(inactive_missing_assets),
        "max_missing_weight": float(out["ibkr_live_missing_weight"].max()) if len(out) else 0.0,
        "cum_transaction_cost": float(out["ibkr_live_transaction_cost"].sum()),
        "cum_ter_cost": float(out["ibkr_live_ter_cost"].sum()),
        "final_wealth": float(out["ibkr_live_wealth"].iloc[-1]),
    })

    mapping_rows = []
    price_note_by_asset = {row["asset"]: row for row in price_notes}
    for row in selected_live_rows(assets):
        note = price_note_by_asset.get(row["asset"], {})
        mapping_rows.append({**row, **{f"price_{k}": v for k, v in note.items() if k != "asset"}})
    mapping_df = pd.DataFrame(mapping_rows)
    return out.reset_index(), pd.DataFrame([stats]), mapping_df, {"price_notes": price_notes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-csv", default=os.path.join(ROOT, "data", "US", "backtest_results", "backtest_timeseries.csv"))
    parser.add_argument("--forex-parquet", default=os.path.join(ROOT, "data", "US", "output_dag", "Forex_daily.parquet"))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--base-currency", default="EUR")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data", "US", "backtest_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timeseries, stats, mapping, diagnostics = build_live_compatible_backtest(
        args.backtest_csv,
        args.forex_parquet,
        args.initial_capital,
        args.base_currency,
    )

    timeseries.to_csv(os.path.join(args.output_dir, "backtest_ibkr_live_timeseries.csv"), index=False)
    stats.to_csv(os.path.join(args.output_dir, "backtest_ibkr_live_stats.csv"), index=False)
    mapping.to_csv(os.path.join(args.output_dir, "backtest_ibkr_live_mapping.csv"), index=False)
    price_cols = ["date"] + [c for c in timeseries.columns if c.endswith("_ibkr_live_price")]
    timeseries[price_cols].to_csv(os.path.join(args.output_dir, "ibkr_live_prices.csv"), index=False)
    with open(os.path.join(args.output_dir, "backtest_ibkr_live_diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=str)

    row = stats.iloc[0].to_dict()
    print("IBKR live-compatible backtest saved")
    print(f"  Range: {row['start_date']} -> {row['end_date']}")
    print(f"  Final wealth: {row['final_wealth']:.2f}")
    print(f"  Total return: {row.get('ibkr_live_total_return', 0.0) * 100:.2f}%")
    print(f"  Missing assets: {row.get('missing_assets') or 'none'}")
    if row.get("inactive_missing_assets"):
        print(f"  Inactive missing assets: {row.get('inactive_missing_assets')}")
    print(f"  Max missing weight: {row.get('max_missing_weight', 0.0) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

