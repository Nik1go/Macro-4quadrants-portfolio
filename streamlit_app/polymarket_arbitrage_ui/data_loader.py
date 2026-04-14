"""SQLite data loading helpers for the Polymarket dashboard.
Works across Windows (Local) and Linux (VPS/Docker).
"""

from __future__ import annotations
import sqlite3
import pandas as pd
import streamlit as st
import os
import json
import re
from pathlib import Path

import sys as _sys

# 1. Base path discovery
# This file is in: root/streamlit_app/polymarket_arbitrage_ui/data_loader.py
# parents[2] should be the repository root.
BASE_DIR = Path(__file__).resolve().parents[2]
STREAMLIT_MAX_ROWS = int(os.getenv("STREAMLIT_MAX_ROWS", "1000"))

_DB_SUB_PATHS = [
    "polymarket_arbitrage/data/dn/arbitrage.db",
    "polymarket_arbitrage/data/dir/arbitrage.db",
    "polymarket_arbitrage/data/arbitrage.db",
]

def _get_db_path() -> str:
    """Find the best candidate for the SQLite database path (Cross-platform)."""

    # Priority 1: Environment variable (Explicit override)
    env_path = os.getenv("SQLITE_DB_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            print(f"[polymarket_ui] DB via SQLITE_DB_PATH: {p.resolve()}", file=_sys.stderr)
            return str(p.resolve())
        p_rel = BASE_DIR / env_path
        if p_rel.exists():
            print(f"[polymarket_ui] DB via SQLITE_DB_PATH (relative): {p_rel.resolve()}", file=_sys.stderr)
            return str(p_rel.resolve())
        print(f"[polymarket_ui] SQLITE_DB_PATH set but not found: {env_path}", file=_sys.stderr)

    # Priority 2: Walk UPWARD from this file — works anywhere without hardcoded paths
    _here = Path(__file__).resolve()
    for ancestor in _here.parents:
        for sub in _DB_SUB_PATHS:
            candidate = ancestor / sub
            if candidate.exists():
                print(f"[polymarket_ui] DB found via ancestor walk: {candidate}", file=_sys.stderr)
                return str(candidate.resolve())

    # Priority 3: Common Docker mount points (internal container paths)
    docker_candidates = [
        Path("/app/data/arbitrage.db"),
        Path("/app/polymarket_arbitrage/data/dn/arbitrage.db"),
        Path("/data/arbitrage.db"),
    ]
    for p in docker_candidates:
        if p.exists():
            print(f"[polymarket_ui] DB found via Docker candidate: {p}", file=_sys.stderr)
            return str(p.resolve())

    # Priority 4: Glob search in /home/* and /root (any username)
    for search_root in [Path("/home"), Path("/root"), Path("/srv"), Path("/opt")]:
        if not search_root.exists():
            continue
        try:
            for match in search_root.glob("**/polymarket_arbitrage/data/dn/arbitrage.db"):
                print(f"[polymarket_ui] DB found via glob in {search_root}: {match}", file=_sys.stderr)
                return str(match.resolve())
        except PermissionError:
            pass

    # Final Fallback: Return the standard path for the error message (so user sees a clear path)
    std_path = BASE_DIR / "polymarket_arbitrage" / "data" / "dn" / "arbitrage.db"
    print(f"[polymarket_ui] DB NOT found anywhere. BASE_DIR={BASE_DIR} | std_path={std_path}", file=_sys.stderr)
    return str(std_path)

DB_PATH = _get_db_path()

@st.cache_data(ttl=30)
def _run_query(query: str, params: tuple | None = None, db_path: str | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    target_db = db_path or _get_db_path()
    
    if not Path(target_db).exists():
        st.warning(f"Base de données introuvable : `{target_db}`")
        st.info(f"Vérifiez l'emplacement du fichier ou la configuration Docker/Service.")
        return pd.DataFrame()

    try:
        # Force Read-Only using proper URI format
        db_uri = f"{Path(target_db).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        df = pd.read_sql_query(query, conn, params=params or tuple())
        conn.close()
        return df
    except Exception as exc:
        st.error(f"Erreur d'accès à la base de données : {exc}")
        return pd.DataFrame()

def load_open_positions(db_path: str | None = None) -> pd.DataFrame:
    """Load currently open positions inferred from trade lifecycle columns."""
    query = """
        SELECT
            timestamp,
            asset_pair,
            side,
            size AS size_usd,
            poly_price AS entry_price_poly,
            exchange_price AS entry_price_binance,
            strategy,
            expected_profit,
            fees_paid,
            exit_timestamp,
            metadata_json
        FROM trades
        WHERE upper(status) = 'FILLED'
          AND (exit_timestamp IS NULL OR trim(exit_timestamp) = '')
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = _run_query(query, (STREAMLIT_MAX_ROWS,), db_path=db_path)

    if df.empty:
        return df

    df["size_usd"]            = pd.to_numeric(df["size_usd"], errors="coerce").fillna(0.0)
    df["entry_price_poly"]    = pd.to_numeric(df["entry_price_poly"], errors="coerce").fillna(0.5).clip(1e-4, 1 - 1e-4)
    df["entry_price_binance"] = pd.to_numeric(df["entry_price_binance"], errors="coerce").fillna(0.0)
    df["unrealized_pnl"]      = 0.0
    df["realized_pnl"]        = None
    df["statut"]              = "Ouvert"

    def _slug_to_label(meta_raw: object) -> str:
        try:
            meta  = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            slug  = str(meta.get("slug", ""))
            if not slug:
                return ""
            asset_map = {"bitcoin": "BTC", "ethereum": "ETH", "xrp": "XRP"}
            m = re.match(
                r"(bitcoin|ethereum|xrp)-above-([\d]+(?:pt[\d]+)?k?)-on-(\w+)-(\d+)",
                slug, re.IGNORECASE
            )
            if not m:
                return slug  
            asset  = asset_map.get(m.group(1).lower(), m.group(1).upper())
            raw_px = m.group(2).lower()
            month  = m.group(3).capitalize()[:3]   
            day    = m.group(4)
            if "pt" in raw_px:
                num = raw_px.replace("pt", ".")
                price_str = f"${float(num):g}"
            elif raw_px.endswith("k"):
                price_str = f"${raw_px[:-1]}K"
            else:
                price_str = f"${raw_px}"
            return f"{asset} > {price_str} on {month} {day}"
        except Exception:
            return ""

    df["marché"] = df["metadata_json"].apply(_slug_to_label)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True) \
                        .dt.tz_convert("Europe/Paris") \
                        .dt.strftime("%d/%m %H:%M")
    
    def _extract_slug(meta_raw: object) -> str:
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            return str(meta.get("slug", ""))
        except Exception: return ""
    df["slug"] = df["metadata_json"].apply(_extract_slug)
    
    return df

@st.cache_data(ttl=15)
def get_latest_bot_scans(slugs: list[str], symbols: list[str]) -> dict[str, dict[str, any]]:
    """Fetch scans from DB with targeted real-time fallback."""
    import requests
    from datetime import datetime, timezone
    
    if not slugs:
        return {}
    
    slug_list = "', '".join(slugs)
    query = f"SELECT slug, spot_price, polymarket_price, timestamp FROM spreads WHERE slug IN ('{slug_list}') AND timestamp IN (SELECT MAX(timestamp) FROM spreads WHERE slug IN ('{slug_list}') GROUP BY slug)"
    df = _run_query(query)
    
    results = {}
    now = datetime.now(timezone.utc)
    for slug in slugs:
        results[slug] = {"spot": 0.0, "poly": 0.0, "ts": "jamais", "source": "manquant"}
        row = df[df["slug"] == slug].iloc[0] if not df.empty and slug in df["slug"].values else None
        
        is_stale = True
        if row is not None:
            results[slug] = {
                "spot": float(row["spot_price"]), 
                "poly": float(row["polymarket_price"]), 
                "ts": str(row["timestamp"]), 
                "source": "bot_scan"
            }
            try:
                ts_str = str(row["timestamp"])
                if " " in ts_str and "T" not in ts_str: 
                    ts = datetime.fromisoformat(ts_str.replace(" ", "T")).replace(tzinfo=timezone.utc)
                else: 
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
                age_sec = (now - ts).total_seconds()
                if age_sec < 60: is_stale = False
                else: results[slug]["source"] = "bot_scan (stale)"
            except Exception: pass
            
        if is_stale:
            headers = {"User-Agent": "Mozilla/5.0"}
            asset_prefix = slug.split("-")[0].lower()
            sym_map = {"bitcoin": "BTC/USDT", "ethereum": "ETH/USDT", "xrp": "XRP/USDT"}
            sym = sym_map.get(asset_prefix, "BTC/USDT")
            try:
                r_b = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={sym.replace('/','')}", headers=headers, timeout=5)
                if r_b.status_code == 200:
                    results[slug]["spot"] = float(r_b.json()["price"])
                    results[slug]["source"] = "realtime (spot)"

                r_p = requests.get(f"https://gamma-api.polymarket.com/markets?slug={slug}", headers=headers, timeout=5)
                if r_p.status_code == 200:
                    data = r_p.json()
                    if data and len(data) > 0:
                        outcome_prices = data[0].get("outcomePrices")
                        if outcome_prices:
                            results[slug]["poly"] = [float(p) for p in outcome_prices]
                            results[slug]["source"] = "realtime" if results[slug]["source"] == "realtime (spot)" else "realtime (poly)"
                results[slug]["ts"] = now.isoformat()
            except Exception: pass
    return results

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
    df = _run_query(query, (STREAMLIT_MAX_ROWS,), db_path=db_path)
    if df.empty: return df
    numeric_cols = ["size", "entry_price", "exit_price", "realized_pnl", "fees_paid"]
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["size"] = df["size"].fillna(0.0)
    df["entry_price"] = df["entry_price"].fillna(0.5).clip(1e-4, 1 - 1e-4)
    df["exit_price"] = df["exit_price"].fillna(df["entry_price"]).clip(1e-4, 1 - 1e-4)
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    df["fees_paid"] = df["fees_paid"].fillna(0.0)
    return df

def load_spread_history(limit: int = 2000, db_path: str | None = None, order: str = "ASC", only_opportunities: bool = False) -> pd.DataFrame:
    """Load spread snapshots for charting or activity monitoring."""
    target_db = db_path or _get_db_path()
    where_clause = "WHERE is_opportunity = 1 AND net_spread >= 0.04" if only_opportunities else ""
    order_clause = "DESC" if order.upper() == "DESC" else "ASC"
    
    has_signal_type = False
    try:
        db_uri = f"{Path(target_db).resolve().as_uri()}?mode=ro"
        tmp_conn = sqlite3.connect(db_uri, uri=True)
        cols = [row[1] for row in tmp_conn.execute("PRAGMA table_info(spreads)").fetchall()]
        has_signal_type = "signal_type" in cols
        tmp_conn.close()
    except Exception: pass

    sig_col = ", signal_type" if has_signal_type else ""
    query = f"""
        SELECT timestamp, asset_pair, spot_price, implied_vol, polymarket_price, theoretical_prob, rfr, net_spread, is_opportunity, strategy, slug {sig_col}
        FROM spreads {where_clause} ORDER BY timestamp {order_clause} LIMIT ?
    """
    df = _run_query(query, (int(limit),), db_path=db_path)
    if df.empty: return df
    if not has_signal_type: df["signal_type"] = "—"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["spot_price", "implied_vol", "polymarket_price", "theoretical_prob", "net_spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "spot_price", "polymarket_price", "theoretical_prob"]).copy()
