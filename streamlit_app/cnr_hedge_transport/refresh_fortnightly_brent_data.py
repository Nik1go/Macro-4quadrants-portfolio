"""Save Brent in EUR averaged over the same calendar half-months as CNR."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
CNR_FILE = DATA_DIR / "cnr_gazole_professionnel_fortnightly.csv"
OUTPUT = DATA_DIR / "brent_bz_eur_fortnightly.csv"

def main() -> None:
    cnr = pd.read_csv(CNR_FILE)
    slots = cnr["date"].tolist()
    if not slots or cnr["date"].duplicated().any():
        raise ValueError("Dates CNR bimensuelles absentes ou dupliquees.")
    start = pd.Timestamp(slots[0][:7] + "-01")
    end = pd.Timestamp(slots[-1][:7] + "-01") + pd.offsets.MonthBegin(1)
    quotes = yf.download(
        ["BZ=F", "EURUSD=X"], start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )
    if quotes.empty:
        raise RuntimeError("Yahoo Finance ne renvoie aucune cotation.")
    close = quotes["Close"]
    daily = pd.DataFrame({"brent_usd": close["BZ=F"], "eurusd": close["EURUSD=X"]}).dropna()
    daily = daily[(daily > 0).all(axis=1)]
    daily["brent_eur"] = daily["brent_usd"] / daily["eurusd"]
    daily["date"] = daily.index.strftime("%Y-%m") + "-H" + np.where(daily.index.day <= 15, "1", "2")
    halves = daily.groupby("date", sort=True)["brent_eur"].agg(brent_eur="mean", daily_observations="count")
    halves = halves.reindex(slots)
    if halves.isna().any().any() or (halves["daily_observations"] < 6).any():
        raise RuntimeError("Une quinzaine CNR manque de cotations Brent communes.")
    halves.to_csv(OUTPUT, float_format="%.8f")
    print(f"{len(halves)} quinzaines : {slots[0]} a {slots[-1]}")

if __name__ == "__main__":
    main()
