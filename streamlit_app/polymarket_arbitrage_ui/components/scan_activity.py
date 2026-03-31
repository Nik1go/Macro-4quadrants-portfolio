"""Scanner activity view for Streamlit - Macro style."""

from __future__ import annotations
import streamlit as st
from ..data_loader import load_spread_history

def render_scan_activity(db_path: str | None = None) -> None:
    """Render a table of recent market scans and calculated spreads."""
    
    st.header("Activité du Scanner")
    st.markdown("Analyse en temps réel de Bitcoin, Ethereum et Solana pour détecter des écarts de prix.")

    # Load recent spreads
    df = load_spread_history(limit=100, db_path=db_path)
    
    if df.empty:
        st.info("Aucune activité de scan enregistrée pour le moment. Le bot est-il démarré ?")
        return

    # Sort by timestamp descending
    df = df.sort_values("timestamp", ascending=False)

    # Format the dataframe for display
    display_df = df[[
        "timestamp", 
        "asset_pair", 
        "slug", 
        "polymarket_price", 
        "theoretical_prob", 
        "net_spread",
        "is_opportunity"
    ]].copy()
    
    display_df.columns = [
        "Temps", 
        "Actif", 
        "Marché", 
        "Prix Poly", 
        "Prix Théorique", 
        "Spread Net",
        "Signal"
    ]

    st.dataframe(
        display_df.style.format({
            "Prix Poly": "{:.4f}",
            "Prix Théorique": "{:.4f}",
            "Spread Net": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.caption("Les signaux (1) indiquent une opportunité de trade détectée par les modèles alpha.")
