"""Scanner activity view for Streamlit - avec auto-refresh toutes les 30s."""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st
from ..data_loader import load_spread_history


def _fmt_timestamp(series: pd.Series) -> pd.Series:
    """Format UTC timestamps to a readable local (Paris) string."""
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.strftime("%d/%m %H:%M")
    )


def _render_table(db_path: str | None) -> None:
    """Load and display the spread history table (called by fragment or manually)."""
    # [NOUVEAU] On charge directement les 150 derniers signaux (Demande utilisateur)
    df = load_spread_history(limit=150, db_path=db_path, order="DESC", only_opportunities=True)

    st.caption(f"Dernière actualisation : {time.strftime('%H:%M:%S')}")

    if df.empty:
        st.info("Aucun signal détecté. Le bot est-il démarré ?")
        return

    # Plus besoin de filter ici, c'est fait en SQL
    # Format timestamp
    df["timestamp"] = _fmt_timestamp(df["timestamp"])

    # Signal lisible
    df["is_opportunity"] = df["is_opportunity"].map(lambda v: "OUI" if int(v or 0) == 1 else "—")

    display_df = df[[
        "timestamp",
        "asset_pair",
        "slug",
        "polymarket_price",
        "theoretical_prob",
        "net_spread",
        "signal_type",
        "spot_price",
    ]].copy()

    display_df.columns = [
        "Temps", "Actif", "Marché",
        "Prix Poly", "Prix Théo", "Spread Net",
        "Signal", "Prix Binance",
    ]

    st.dataframe(
        display_df.style.format({
            "Prix Poly": "{:.4f}",
            "Prix Théo": "{:.4f}",
            "Spread Net": "{:.2%}",
            "Prix Binance": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Résumé rapide — is_opportunity est la colonne originale (avant rename)
    # On supporte "1", " 1" et "OUI" pour la robustesse
    opp_count = int((pd.to_numeric(df["is_opportunity"].map(
        lambda v: 1 if str(v).strip() in {"1", " 1", "OUI"} else 0
    ), errors="coerce").fillna(0)).sum())
    st.caption(
        f"{len(df)} entrées chargées · {opp_count} signaux détectés · "
        "Signaux = opportunités d'arbitrage filtrées"
    )


def render_scan_activity(db_path: str | None = None) -> None:
    """Render scanner activity with auto-refresh every 30 seconds."""

    st.header("Activité du Scanner")
    st.markdown("Analyse en temps réel de **Bitcoin**, **Ethereum** et **XRP** pour détecter des écarts de prix.")

    # ── Auto-refresh via st.fragment si disponible (Streamlit ≥ 1.33) ──────
    # On tente d'utiliser le décorateur @st.fragment(run_every=30).
    # Si la version est trop ancienne, on bascule sur un bouton manuel.
    try:
        @st.fragment(run_every=30)
        def _auto_table():
            _render_table(db_path)

        _auto_table()

    except (AttributeError, TypeError):
        # Fallback : bouton de refresh manuel
        col_refresh, col_info = st.columns([1, 5])
        with col_refresh:
            if st.button("↺ Actualiser", key="scan_refresh_btn"):
                st.cache_data.clear()
        with col_info:
            st.caption("Auto-refresh non disponible (Streamlit < 1.33) — cliquez sur Actualiser.")

        _render_table(db_path)

    st.divider()
