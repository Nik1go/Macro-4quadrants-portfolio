"""Polymarket Arbitrage Dashboard - Refactored for Macro Project alignment."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from .components.backtest_view import render_backtest_view
from .components.position_monitoring import render_position_monitoring
from .components.strategy_explanation import render_strategy_explanation
from .components.scan_activity import render_scan_activity
from .data_loader import load_open_positions, load_trades_history

def render() -> None:
    """Render the Polymarket dashboard integrated into the main app."""
    
    st.title("Polymarket Arbitrage")
    st.markdown("""
    **Objectif :** Exploiter les inefficacités de prix entre Polymarket et les marchés Spot/Perp via des stratégies Delta-Neutral et Directionnelles.
    """)

    # --- TOP SELECTION (Macro Style) ---
    st.subheader("Configuration du Compte")
    account_choice = st.radio(
        "Sélectionner le Compte / Stratégie",
        ["Delta Neutral", "Directionnel"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Map choice to DB path and metadata
    if account_choice == "Delta Neutral":
        db_path = "polymarket_arbitrage/data/dn/arbitrage.db"
        port = 18080
        strategy_name = "Delta Neutral"
    else:
        db_path = "polymarket_arbitrage/data/dir/arbitrage.db"
        port = 18081
        strategy_name = "Pure Polymarket"

    st.divider()

    # --- HEADER KPIs (5 columns like Macro Project) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Load data for KPIs
    open_pos = load_open_positions(db_path=db_path)
    trades_hist = load_trades_history(db_path=db_path)
    
    total_pnl = trades_hist["realized_pnl"].sum() if not trades_hist.empty else 0.0
    active_count = len(open_pos)

    c1.metric("Capital", "$10,000")
    c2.metric("Profit Réalisé", f"${total_pnl:,.2f}", delta=f"{(total_pnl/10000):.2%}" if total_pnl != 0 else None)
    c3.metric("Positions Actives", str(active_count))
    c4.metric("Stratégie", strategy_name)
    c5.metric("Santé Flux", f"Port {port}", delta="Live", delta_color="normal")

    st.divider()

    # --- TAB NAVIGATION (Simple text, no emojis) ---
    tab_explanation, tab_scan, tab_monitoring, tab_backtest = st.tabs(
        [
            "Mécanique Stratégique",
            "Activité du Scanner",
            "Monitoring Positions",
            "Backtesting",
        ]
    )

    with tab_explanation:
        render_strategy_explanation()

    with tab_scan:
        render_scan_activity(db_path=db_path)

    with tab_monitoring:
        render_position_monitoring(db_path=db_path)

    with tab_backtest:
        render_backtest_view()
