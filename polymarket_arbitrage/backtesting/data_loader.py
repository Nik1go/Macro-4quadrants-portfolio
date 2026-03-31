"""Historical data loader for strategy backtesting."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

try:
    from ..utils.config import Config
except ImportError:  # pragma: no cover
    from utils.config import Config


DateLike = Union[str, date, datetime]


def _to_iso_utc(value: DateLike) -> str:
    """Convert date-like input to ISO UTC timestamp string."""
    if isinstance(value, str):
        dt = pd.to_datetime(value, utc=True, errors="coerce")
    elif isinstance(value, date) and not isinstance(value, datetime):
        dt = pd.Timestamp(value).tz_localize("UTC")
    else:
        dt = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(dt):
        raise ValueError(f"Invalid date input: {value}")

    return dt.to_pydatetime().isoformat()


def load_historical_data(
    start_date: DateLike,
    end_date: DateLike,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical spread snapshots used for backtesting.

    The resulting DataFrame is sorted by timestamp and sanitized for missing
    values and outliers.
    """
    database = db_path or Config.DB_PATH
    start_iso = _to_iso_utc(start_date)
    end_iso = _to_iso_utc(end_date)

    conn = sqlite3.connect(database)
    query = """
        SELECT
            timestamp,
            asset_pair,
            spot_price,
            implied_vol,
            polymarket_price,
            theoretical_prob,
            rfr,
            net_spread,
            strategy,
            slug
        FROM spreads
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=[start_iso, end_iso])
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    numeric_columns = ["spot_price", "implied_vol", "polymarket_price", "theoretical_prob", "rfr", "net_spread"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "spot_price", "polymarket_price", "theoretical_prob"]).copy()
    if df.empty:
        return df

    df["spot_price"] = df["spot_price"].clip(lower=1e-6)
    df["implied_vol"] = df["implied_vol"].fillna(0.60).clip(lower=0.01, upper=3.0)
    df["polymarket_price"] = df["polymarket_price"].fillna(0.5).clip(lower=1e-4, upper=1 - 1e-4)
    df["theoretical_prob"] = df["theoretical_prob"].fillna(0.5).clip(lower=1e-4, upper=1 - 1e-4)
    df["net_spread"] = df["net_spread"].fillna(0.0)
    df["rfr"] = df["rfr"].fillna(0.0)

    df["slug"] = df["slug"].fillna("unknown_slug")
    df["asset_pair"] = df["asset_pair"].fillna("UNKNOWN")

    # Conservative de-duplication to avoid look-ahead contamination from repeated snapshots.
    df = df.drop_duplicates(subset=["timestamp", "slug"], keep="first").reset_index(drop=True)
    return df
