"""Polymarket Arbitrage Dashboard - Refactored for Macro Project alignment."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from .components.backtest_view import render_backtest_view
from .components.position_monitoring import render_position_monitoring
from .components.strategy_explanation import render_strategy_explanation
from .components.scan_activity import render_scan_activity
from .data_loader import load_open_positions, load_trades_history

# --- THEME CSS (FROM MACRO PROJECT) ---
BLOOMBERG_CSS = """
    <style>
    /* Tab navigation styling - bigger + separated */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 2px solid #3d4263 !important;
        padding-bottom: 0 !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 !important;
        justify-content: center !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 14px 28px !important;
        color: #8890b5 !important;
        border-right: 2px solid #3d4263 !important;
    }
    .stTabs [data-baseweb="tab"]:last-child {
        border-right: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.08) !important;
        border-bottom: 3px solid #00d4ff !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    </style>
"""

def render() -> None:
    """Render the Polymarket dashboard integrated into the main app."""
    
    st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)
    
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
        strategy_name = "Delta Neutral"
    else:
        db_path = "polymarket_arbitrage/data/dir/arbitrage.db"
        strategy_name = "Pure Polymarket"

    st.divider()

    # --- HEADER KPIs (5 columns like Macro Project) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Load data for KPIs
    open_pos = load_open_positions(db_path=db_path)
    trades_hist = load_trades_history(db_path=db_path)
    
    total_pnl = trades_hist["realized_pnl"].sum() if not trades_hist.empty else 0.0
    active_count = len(open_pos)

    # Calcul du cash restant : Capital Initial + PnL Réalisé - Capital immobilisé (positions ouvertes)
    initial_cap = 10000.0
    invested_cap = open_pos["size_usd"].sum() if not open_pos.empty else 0.0
    remaining_cash = initial_cap + total_pnl - invested_cap

    c1.metric("Capital Initial", f"${initial_cap:,.0f}")
    c2.metric("Profit Réalisé", f"${total_pnl:,.2f}", delta=f"{(total_pnl/initial_cap):.2%}" if total_pnl != 0 else None)
    c3.metric("Positions Actives", str(active_count))
    c4.metric("Mode", strategy_name)
    c5.metric("Cash Restant", f"${remaining_cash:,.2f}")

    st.divider()

    # --- TAB NAVIGATION (Macro Style: Uppercase & Blocks) ---
    tab_explanation, tab_scan, tab_monitoring, tab_backtest = st.tabs(
        [
            "Methodologie",
            "Scanner Activity",
            "Monitoring Positions",
            "Backtest & Perf",
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
