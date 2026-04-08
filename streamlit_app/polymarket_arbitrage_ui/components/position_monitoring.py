"""Position monitoring view - Macro style."""

from __future__ import annotations
import pandas as pd
import streamlit as st
from ..data_loader import load_open_positions, load_trades_history, get_latest_bot_scans

def render_position_monitoring(db_path: str | None = None) -> None:
    """Render metrics and tables for current arbitrage activity."""
    
    st.header("Monitoring Temps Réel")

    # Load data
    positions_df = load_open_positions(db_path=db_path)
    trades_df    = load_trades_history(db_path=db_path)

    # ── Real-Time Bot Scan Enrichment ────────────────────────────────────────
    live_pnl_total = 0.0
    if not positions_df.empty:
        slugs = positions_df["slug"].unique().tolist()
        symbols = positions_df["asset_pair"].unique().tolist()
        # On récupère les derniers scans réels faits par le bot (ou fetch direct si vieux)
        latest_scans = get_latest_bot_scans(slugs, symbols)
        
        def _calc_pnl_from_scans(row):
            slug = row["slug"]
            side = row["side"]
            strategy = row["strategy"]
            
            p_entry = row["entry_price_poly"]
            b_entry = row["entry_price_binance"]
            size    = row["size_usd"]
            
            scan = latest_scans.get(slug)
            source = "inconnu"
            if scan:
                # Gestion du format hybride (float pour DB, list pour Realtime)
                p_current_data = scan["poly"]
                if isinstance(p_current_data, list):
                    # Realtime : YES=0, NO=1
                    p_live = p_current_data[0] if "yes" in side.lower() else p_current_data[1]
                else:
                    # DB : Prix scanné par le bot
                    p_live = float(p_current_data)
                
                b_live = float(scan["spot"])
                source = scan.get("source", "bot_scan")
            else:
                p_live = p_entry
                b_live = b_entry
            
            # Poly PnL
            p_pnl = (p_live - p_entry) * (size / p_entry) if p_entry > 0 else 0.0
            
            # Binance PnL
            b_pnl = 0.0
            if strategy == "delta_neutral":
                # Si YES Poly -> SHORT Binance
                if "buy_yes" in side.lower():
                    b_pnl = (b_entry - b_live) * (size / b_entry) if b_entry > 0 else 0.0
                # Si NO Poly -> LONG Binance
                else:
                    b_pnl = (b_live - b_entry) * (size / b_entry) if b_entry > 0 else 0.0
                    
            return p_pnl + b_pnl, p_live, b_live, source

        # Apply calculation
        pnl_data = positions_df.apply(_calc_pnl_from_scans, axis=1)
        positions_df["unrealized_pnl"] = [x[0] for x in pnl_data]
        positions_df["current_poly"]   = [x[1] for x in pnl_data]
        positions_df["current_bin"]    = [x[2] for x in pnl_data]
        positions_df["data_source"]    = [x[3] for x in pnl_data]
        live_pnl_total = positions_df["unrealized_pnl"].sum()

    # Summary Metrics
    closed_df = trades_df[trades_df["exit_timestamp"].notna() & (trades_df["exit_timestamp"] != "")] \
                if not trades_df.empty else pd.DataFrame()

    total_realized_pnl = closed_df["realized_pnl"].sum()   if not closed_df.empty else 0.0
    total_fees         = trades_df["fees_paid"].sum()       if not trades_df.empty else 0.0
    win_rate           = (closed_df["realized_pnl"] > 0).mean() if len(closed_df) > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PnL Réalisé",        f"${total_realized_pnl:,.2f}")
    m2.metric("Positions Ouvertes", str(len(positions_df)))
    m3.metric("PnL Latent (Live)",  f"${live_pnl_total:,.2f}", delta=f"{live_pnl_total:+.2f}")
    m4.metric("Frais Totaux",       f"${total_fees:,.2f}", delta_color="inverse")

    st.divider()

    # ── Open Positions ──────────────────────────────────────────────────────
    st.subheader("Positions Ouvertes")
    if positions_df.empty:
        st.info("Aucune position ouverte actuellement.")
    else:
        display_open = positions_df[[
            "timestamp", "asset_pair", "marché", "side", "size_usd",
            "entry_price_poly", "current_poly",
            "entry_price_binance", "current_bin",
            "unrealized_pnl", "data_source"
        ]].copy()

        # [HIGHLIGHT] Styling basé sur la source
        def _style_source(row):
            styles = [""] * len(row)
            # Si la source est "realtime", on met en bleu pour montrer que c'est du frais forcé
            if row["Source"] == "realtime":
                styles[6] = "color: #00d4ff; font-weight: bold;"
                styles[8] = "color: #00d4ff; font-weight: bold;"
            # Si c'est un vieux scan bot (identifié par p_entry == p_live), on grise
            elif row["Prix Poly (In)"] == row["Prix Poly (Live)"]:
                styles[6] = "color: #888888; font-style: italic;"
            return styles

        # Direction lisible
        dir_map = {"buy_yes": "YES (Long Poly)", "buy_no": "NO (Short Poly)"}
        display_open["side"] = display_open["side"].map(lambda d: dir_map.get(d, d))

        display_open.columns = [
            "Ouvert le", "Actif", "Pari", "Direction", "Taille ($)",
            "Prix Poly (In)", "Prix Poly (Live)",
            "Binance (In)", "Binance (Live)",
            "PnL Latent ($)", "Source"
        ]
        
        st.dataframe(
            display_open.style.format({
                "Taille ($)":       "${:,.2f}",
                "Prix Poly (In)":   "{:.4f}",
                "Prix Poly (Live)": "{:.4f}",
                "Binance (In)":     "${:,.2f}",
                "Binance (Live)":   "${:,.2f}",
                "PnL Latent ($)":   "{:+.4f}$",
            }).apply(_style_source, axis=1).applymap(lambda x: "color: #00ff00;" if x > 0 else "color: #ff4b4b;" if x < 0 else "", subset=["PnL Latent ($)"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Les prix 'Live' sont hybrides : Bot Scan si < 15min, sinon Real-time global. PnL Latente Globale: {live_pnl_total:+.4f}$")

    st.divider()

    # ── Trades History ──────────────────────────────────────────────────────
    st.subheader("Historique des Trades Clôturés")
    if closed_df.empty:
        st.info("Aucun trade clôturé pour le moment.")
    else:
        display_closed = closed_df[[
            "timestamp", "asset_pair", "side", "size",
            "entry_price", "exit_price", "exit_timestamp",
            "strategy", "realized_pnl", "fees_paid"
        ]].copy()
        display_closed.columns = [
            "Ouvert le", "Actif", "Direction", "Taille ($)",
            "Prix Entrée", "Prix Sortie", "Clôturé le",
            "Stratégie", "PnL Réalisé ($)", "Frais ($)"
        ]
        st.dataframe(
            display_closed.style.format({
                "Taille ($)":      "${:,.2f}",
                "Prix Entrée":     "{:.4f}",
                "Prix Sortie":     "{:.4f}",
                "PnL Réalisé ($)": "${:,.4f}",
                "Frais ($)":       "${:,.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

