"""Refresh monthly BZ=F Brent in EUR using the notebook's Yahoo tickers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT = DATA_DIR / "brent_bz_eur_monthly.csv"


def main() -> None:
    quotes = yf.download(
        ["BZ=F", "EURUSD=X"],
        start="2020-01-01",
        end="2026-08-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if quotes.empty:
        raise RuntimeError("Yahoo Finance n'a renvoyé aucune donnée.")
    close = quotes["Close"]
    daily = pd.DataFrame(
        {
            "bz_usd": close["BZ=F"],
            "eurusd": close["EURUSD=X"],
        }
    ).dropna()
    daily = daily[(daily > 0).all(axis=1)]
    daily["brent_eur"] = daily["bz_usd"] / daily["eurusd"]
    monthly = daily["brent_eur"].resample("MS").agg(["mean", "count"])
    monthly = monthly.loc[monthly.index <= "2026-07-01"]
    monthly = monthly.rename(columns={"mean": "brent_eur", "count": "daily_observations"})
    monthly.index.name = "date"
    if len(monthly) < 70 or (monthly["daily_observations"] < 10).any():
        raise RuntimeError("Historique Brent incomplet ; export annulé.")
    monthly.to_csv(OUTPUT, float_format="%.8f")
    print(f"{len(monthly)} mois : {monthly.index.min().date()} à {monthly.index.max().date()}")
    print(monthly.head(2).to_string())
    print(monthly.tail(2).to_string())


if __name__ == "__main__":
    main()
