"""
US Macro Four Seasons Strategy & Portfolio - Main Dashboard
Single-page app with session_state navigation.

Run: streamlit run streamlit_app/app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import math

# Add streamlit_app to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, apply_theme, render_sidebar, QUADRANT_NAMES
from views import Monitoring, Backtest, ML_Performance, Methodologie, Correlation_Indicators

def render():
    # Optional button to return to home inside the main page, or side-bar
    with st.sidebar:
        if st.button("⬅️ Retour au Home", use_container_width=True):
            st.session_state.current_page = "Home"
            st.session_state.scroll_to_top = True
            st.rerun()
        st.markdown("---")

    # --- THEME & DATA ---
    apply_theme()
    data = load_data()
    render_sidebar(data)
    
    # --- HEADER KPIs ---
    st.title("US Macro Four Seasons Strategy")
    st.markdown("""
    **Objectif :** Surperformer le marche en adaptant l'allocation d'actifs selon le cycle economique US.
    **Approche :** Modele 4 Quadrants base sur les taux de changement de la **Croissance** et de l'**Inflation**.
    """)
    
    c_header1, c_header2, c_header3, c_header4, c_header5 = st.columns(5)
    
    if data['quadrants'] is not None:
        latest = data['quadrants'].iloc[-1]
    
        raw_q = int(latest.get('assigned_quadrant', 0))
        c_header1.metric("Raw Q (Today)", f"Q{raw_q} {QUADRANT_NAMES.get(raw_q, 'Unknown')}", delta="Volatile Signal", delta_color="off")
    
        if data['backtest'] is not None:
            latest_bt = data['backtest'].iloc[-1]
            smooth_q = int(latest_bt.get('smooth_quadrant', 0))
            c_header2.metric("Model Q (Allocated)", f"Q{smooth_q} {QUADRANT_NAMES.get(smooth_q, 'Unknown')}", delta="Stable Signal", delta_color="normal")
        else:
            c_header2.metric("Model Q", "N/A")
    
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
    
        # RISK REGIME
        if 'MACRO_GROWTH_SCORE' in latest:
            risk_score = latest['MACRO_GROWTH_SCORE']
            risk_prob = sigmoid(risk_score)
            risk_label = "Risk On" if risk_score > 0 else "Risk Off"
            c_header3.metric("Risk Regime", f"{risk_label} ({risk_prob:.0%})")
        elif 'PROB_GROWTH_EMA' in latest:
            risk_prob = latest['PROB_GROWTH_EMA']
            risk_label = "Risk On" if risk_prob > 0.5 else "Risk Off"
            c_header3.metric("Risk Regime", f"{risk_label} ({risk_prob:.0%})")
        else:
            # Legacy fallback
            risk_score = latest.get('score_Q1', 0) - latest.get('score_Q3', 0)
            risk_prob = sigmoid(risk_score / 2.0) # Scale down naive scores for better sigmoid
            risk_label = "Risk On" if risk_score > 0 else "Risk Off"
            c_header3.metric("Risk Regime", f"{risk_label} ({risk_prob:.0%})")
    
        # RATES REGIME
        if 'MACRO_INFLATION_SCORE' in latest:
            rates_score = latest['MACRO_INFLATION_SCORE']
            rates_prob = sigmoid(rates_score)
            rates_label = "Reflation" if rates_score > 0 else "Disinflation"
            c_header4.metric("Rates Regime", f"{rates_label} ({rates_prob:.0%})")
        elif 'PROB_INFLATION_EMA' in latest:
            rates_prob = latest['PROB_INFLATION_EMA']
            rates_label = "Reflation" if rates_prob > 0.5 else "Disinflation"
            c_header4.metric("Rates Regime", f"{rates_label} ({rates_prob:.0%})")
        else:
            # Legacy fallback
            rates_score = latest.get('score_Q2', 0) - latest.get('score_Q4', 0)
            rates_prob = sigmoid(rates_score / 2.0) # Scale down naive scores for better sigmoid
            rates_label = "Reflation" if rates_score > 0 else "Disinflation"
            c_header4.metric("Rates Regime", f"{rates_label} ({rates_prob:.0%})")
    
        last_date = latest['date'].strftime("%Y-%m-%d") if pd.notnull(latest['date']) else "N/A"
        c_header5.metric("Last Update", last_date)
    
    st.divider()
    
    # --- TAB NAVIGATION (no emojis, side by side) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Monitoring",
        "Backtest & Perf",
        "ML Performance",
        "Methodologie",
        "Correlation & Indicators"
    ])
    
    with tab1:
        Monitoring.render(data)
    
    with tab2:
        Backtest.render(data)
    
    with tab3:
        ML_Performance.render(data)
    
    with tab4:
        Methodologie.render(data)
    
    with tab5:
        Correlation_Indicators.render(data)


