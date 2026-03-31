"""Backtesting view - Macro style."""

from __future__ import annotations
from datetime import timedelta
import pandas as pd
import streamlit as st

# Root logic imports
from polymarket_arbitrage.backtesting.backtest_engine import run_backtest
from polymarket_arbitrage.backtesting.data_loader import load_historical_data
from polymarket_arbitrage.backtesting.metrics import calculate_metrics

def render_backtest_view() -> None:
    """Render interactive controls and outputs for backtesting."""
    
    st.header("Moteur de Backtesting")

    # Simulation Controls
    st.subheader("Paramètres de Simulation")
    
    default_end = pd.Timestamp.utcnow().date()
    default_start = (pd.Timestamp.utcnow() - timedelta(days=60)).date()

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        strategy = st.selectbox("Stratégie", ["delta_neutral", "pure_polymarket"], key="poly_strat_select")
        min_edge = st.slider("Seuil de spread (%)", 0.1, 5.0, 0.5, 0.1, key="poly_edge_slider") / 100.0
    with col_b:
        start_date = st.date_input("Date de début", value=default_start, key="poly_start_date")
        end_date = st.date_input("Date de fin", value=default_end, key="poly_end_date")
    with col_c:
        max_position = st.slider("Taille max (% capital)", 1, 20, 5, 1, key="poly_max_pos") / 100.0
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Lancer Simulation", type="primary", use_container_width=True, key="poly_run_bt")

    if run_btn:
        if start_date >= end_date:
            st.error("Dates invalides.")
            return

        with st.spinner("Simulation..."):
            historical_data = load_historical_data(start_date=start_date, end_date=end_date)
            if historical_data.empty:
                st.error("Données historiques introuvables.")
                return

            results = run_backtest(
                data=historical_data,
                strategy_name=strategy,
                min_edge=min_edge,
                max_position_size=max_position,
            )
            
            if results.empty:
                st.warning("Aucun trade généré.")
                return

            metrics = calculate_metrics(results)

        st.divider()
        
        # Metrics Row
        st.subheader("Performance du Backtest")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Return Total", f"{metrics['total_return']:.2%}")
        m2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        m3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}", delta_color="inverse")
        m4.metric("Win Rate", f"{metrics['win_rate']:.2%}")

        st.divider()

        # Equity Curve
        st.subheader("Courbe de Capital")
        equity_df = results[["timestamp", "equity"]].copy()
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce")
        equity_df = equity_df.dropna(subset=["timestamp"]).set_index("timestamp")
        st.area_chart(equity_df["equity"], color="#00d4ff")

        st.subheader("Journal des Trades")
        trade_events = results[results["trade_executed"]].copy()
        if trade_events.empty:
            st.info("Aucun trade.")
        else:
            st.dataframe(
                trade_events.style.format({
                    "size": "${:,.2f}",
                    "entry_price": "{:.4f}",
                    "exit_price": "{:.4f}",
                    "pnl": "${:,.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )
