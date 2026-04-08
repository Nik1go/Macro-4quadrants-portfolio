"""SQLite data loading helpers for the Polymarket dashboard within streamlit_app."""

from __future__ import annotations
import sqlite3
import pandas as pd
import streamlit as st
import os
from pathlib import Path

# Configuration locale pour éviter d'importer le module 'utils' du bot
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB = str(BASE_DIR / "polymarket_arbitrage" / "data" / "dn" / "arbitrage.db")
DB_PATH = os.getenv("SQLITE_DB_PATH", DEFAULT_DB)
STREAMLIT_MAX_ROWS = int(os.getenv("STREAMLIT_MAX_ROWS", "1000"))

@st.cache_data(ttl=30)
def _run_query(query: str, params: tuple | None = None, db_path: str | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    target_db = db_path or DB_PATH
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

    # Parse slug depuis metadata_json → label lisible "BTC above $66K on Apr 6"
    import json, re
    def _slug_to_label(meta_raw: object) -> str:
        try:
            meta  = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            slug  = str(meta.get("slug", ""))
            if not slug:
                return ""
            # Ex : bitcoin-above-66k-on-april-6  →  BTC above $66K on Apr 6
            #      ethereum-above-2k-on-april-7  →  ETH above $2K on Apr 7
            #      xrp-above-1pt6-on-april-7     →  XRP above $1.6 on Apr 7
            asset_map = {"bitcoin": "BTC", "ethereum": "ETH", "xrp": "XRP"}
            m = re.match(
                r"(bitcoin|ethereum|xrp)-above-([\d]+(?:pt[\d]+)?k?)-on-(\w+)-(\d+)",
                slug, re.IGNORECASE
            )
            if not m:
                return slug  # fallback brut
            asset  = asset_map.get(m.group(1).lower(), m.group(1).upper())
            raw_px = m.group(2).lower()
            month  = m.group(3).capitalize()[:3]   # 'april' → 'Apr'
            day    = m.group(4)
            # Parse prix : '66k' → '$66K', '1pt6' → '$1.6', '2k' → '$2K'
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

    # Format date lisible : "05/04 09:02"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True) \
                        .dt.tz_convert("Europe/Paris") \
                        .dt.strftime("%d/%m %H:%M")
    
    # Extraire le slug brut pour le fetch live
    def _extract_slug(meta_raw: object) -> str:
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            return str(meta.get("slug", ""))
        except Exception: return ""
    df["slug"] = df["metadata_json"].apply(_extract_slug)
    
    return df

@st.cache_data(ttl=15)
def get_latest_bot_scans(slugs: list[str], symbols: list[str]) -> dict[str, dict[str, any]]:
    """Fetch scans from DB with targeted real-time fallback for stale/missing data."""
    import requests
    from datetime import datetime, timezone
    
    if not slugs:
        return {}
    
    # 1. SQL Query for latest scans
    slug_list = "', '".join(slugs)
    query = f"SELECT slug, spot_price, polymarket_price, timestamp FROM spreads WHERE slug IN ('{slug_list}') AND timestamp IN (SELECT MAX(timestamp) FROM spreads WHERE slug IN ('{slug_list}') GROUP BY slug)"
    df = _run_query(query)
    
    results = {}
    now = datetime.now(timezone.utc)
    
    for slug in slugs:
        # Default empty result
        results[slug] = {"spot": 0.0, "poly": 0.0, "ts": "jamais", "source": "manquant"}
        
        # 1. SQL Query for latest scans
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
                # [FIX] On s'assure de comparer des UTC
                ts_str = str(row["timestamp"])
                if " " in ts_str and "T" not in ts_str: # Format SQL classique
                    ts = datetime.fromisoformat(ts_str.replace(" ", "T")).replace(tzinfo=timezone.utc)
                else: 
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
                
                age_sec = (now - ts).total_seconds()
                # Seuil réduit à 60s pour forcer le refresh si le bot n'est pas actif SUR ce slug
                if age_sec < 60:
                    is_stale = False
                else:
                    results[slug]["source"] = "bot_scan (stale)"
            except Exception: pass
            
        # 2. Targeted Real-time Fallback (If stale or missing)
        if is_stale:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
            
            # Match asset name from slug (ex: 'ethereum-above...' -> 'ETH/USDT')
            asset_prefix = slug.split("-")[0].lower()
            sym_map = {"bitcoin": "BTC/USDT", "ethereum": "ETH/USDT", "xrp": "XRP/USDT"}
            sym = sym_map.get(asset_prefix, "BTC/USDT")
            
            new_spot = results[slug]["spot"]
            new_poly = results[slug]["poly"]
            source   = results[slug]["source"]
            
            try:
                # Always try Binance for spot price
                r_b = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={sym.replace('/','')}", headers=headers, timeout=5)
                if r_b.status_code == 200:
                    new_spot = float(r_b.json()["price"])
                    source   = "realtime (spot)"

                # Try Polymarket for orderbook
                r_p = requests.get(f"https://gamma-api.polymarket.com/markets?slug={slug}", headers=headers, timeout=5)
                if r_p.status_code == 200:
                    data = r_p.json()
                    if data and len(data) > 0:
                        outcome_prices = data[0].get("outcomePrices")
                        if outcome_prices:
                            new_poly = [float(p) for p in outcome_prices]
                            source   = "realtime" if source == "realtime (spot)" else "realtime (poly)"
                
                results[slug] = {
                    "spot": new_spot,
                    "poly": new_poly,
                    "ts": now.isoformat(),
                    "source": source
                }
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

def load_spread_history(limit: int = 2000, db_path: str | None = None, order: str = "ASC", only_opportunities: bool = False) -> pd.DataFrame:
    """Load spread snapshots for charting or activity monitoring."""
    target_db = db_path or DB_PATH
    # [MAJ] Seuil de 4% pour l'affichage (Demande utilisateur)
    where_clause = "WHERE is_opportunity = 1 AND net_spread >= 0.04" if only_opportunities else ""
    order_clause = "DESC" if order.upper() == "DESC" else "ASC"
    
    # [NOUVEAU] Vérification de la présence de la colonne signal_type pour éviter les erreurs de migration
    has_signal_type = False
    try:
        tmp_conn = sqlite3.connect(target_db)
        cols = [row[1] for row in tmp_conn.execute("PRAGMA table_info(spreads)").fetchall()]
        has_signal_type = "signal_type" in cols
        tmp_conn.close()
    except Exception:
        pass

    sig_col = ", signal_type" if has_signal_type else ""
    
    query = f"""
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
            {sig_col}
        FROM spreads
        {where_clause}
        ORDER BY timestamp {order_clause}
        LIMIT ?
    """
    df = _run_query(query, (int(limit),), db_path=db_path)

    if df.empty:
        return df

    # Si la colonne manquait, on l'ajoute vide pour ne pas casser le composant UI
    if not has_signal_type:
        df["signal_type"] = "—"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["spot_price", "implied_vol", "polymarket_price", "theoretical_prob", "net_spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "spot_price", "polymarket_price", "theoretical_prob"]).copy()
    return df
