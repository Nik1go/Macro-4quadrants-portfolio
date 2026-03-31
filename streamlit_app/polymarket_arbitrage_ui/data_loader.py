"""SQLite data loading helpers for the Polymarket dashboard within streamlit_app."""

from __future__ import annotations
import sqlite3
import pandas as pd
import streamlit as st
from polymarket_arbitrage.utils.config import Config

@st.cache_data(ttl=1)
def _run_query(query: str, params: tuple | None = None, db_path: str | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    target_db = db_path or Config.DB_PATH
    try:
        conn = sqlite3.connect(target_db)
        df = pd.read_sql_query(query, conn, params=params or tuple())
        conn.close()
        return df
    except Exception as exc:
        st.error(f"Base de données indisponible ou requête invalide: {exc}")
        return pd.DataFrame()

def load_open_positions(db_path: str | None = None) -> pd.DataFrame:
    """Load currently open positions inferred from trade lifecycle columns."""
    query = """
        SELECT
            timestamp,
            asset_pair,
            side,
            size AS size_usd,
            poly_price AS entry_price,
            strategy,
            expected_profit,
            realized_pnl,
            fees_paid,
            exit_timestamp
        FROM trades
        WHERE upper(status) = 'FILLED'
          AND (exit_timestamp IS NULL OR trim(exit_timestamp) = '')
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = _run_query(query, (Config.STREAMLIT_MAX_ROWS,), db_path=db_path)

    if df.empty:
        return df

    df["size_usd"] = pd.to_numeric(df["size_usd"], errors="coerce").fillna(0.0)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.5).clip(1e-4, 1 - 1e-4)
    df["current_price"] = df["entry_price"]
    df["unrealized_pnl"] = 0.0
    return df

def load_trades_history(db_path: str | None = None) -> pd.DataFrame:
    """Load trade history with realized PnL and exit fields."""
    query = """
        SELECT
            timestamp,
            asset_pair,
            side,
            size,
            poly_price AS entry_price,
            exit_price,
            exit_timestamp,
            strategy,
            realized_pnl,
            fees_paid,
            status,
            trade_type
        FROM trades
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = _run_query(query, (Config.STREAMLIT_MAX_ROWS,), db_path=db_path)

    if df.empty:
        return df

    numeric_cols = ["size", "entry_price", "exit_price", "realized_pnl", "fees_paid"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["size"] = df["size"].fillna(0.0)
    df["entry_price"] = df["entry_price"].fillna(0.5).clip(1e-4, 1 - 1e-4)
    df["exit_price"] = df["exit_price"].fillna(df["entry_price"]).clip(1e-4, 1 - 1e-4)
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    df["fees_paid"] = df["fees_paid"].fillna(0.0)

    return df

def load_spread_history(limit: int = 2000, db_path: str | None = None) -> pd.DataFrame:
    """Load spread snapshots for charting."""
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
            is_opportunity,
            strategy,
            slug
        FROM spreads
        ORDER BY timestamp ASC
        LIMIT ?
    """
    df = _run_query(query, (int(limit),), db_path=db_path)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["spot_price", "implied_vol", "polymarket_price", "theoretical_prob", "net_spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "spot_price", "polymarket_price", "theoretical_prob"]).copy()
    return df
