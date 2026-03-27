import streamlit as st
import scipy.optimize as sco
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# --- CONFIGURATION ---
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"
CAPITAL_INITIAL = 10000
tickers = ['SPY', 'QQQ', 'GC=F', 'BTC-USD', '^FCHI']

# --- DATA FETCHING ---
@st.cache_data
def fetch_data(tickers):
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, interval=INTERVAL)['Adj Close']
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def optimize_protfolio(data): 
    returns = data.pct_change().dropna()
    sharp = (returns.mean()/returns.std()*np.sqrt(252)) if returns > 0 else 0 
    return{ "sharp: " f"{sharp:.2f}" }




def render():
    st.title("Portfolio Optimizer")
    st.markdown("à des fin d'entrainement je souhaite optimiser un portefeuille multi-actifs.")
    st.markdown("les actifs sont : " + ", ".join(tickers))
    st.markdown("la période est du " + START_DATE + " au " + END_DATE)
    
    st.divider()

    col_data, col_chart = st.columns([1,3])
    with col_chart:
        st.markdown("### Chart")
        data = fetch_data(tickers)
        st.line_chart(data)

    with col_data:
        c1,c2,c3= st.columns(3)
        c1.metrics = (returns.mean()/returns.std()*np.sqrt(252)) if returns > 0 else 0
        st.markdown("l'objecctif va donc de chercher la pondération de chaque actif pour maximiser les rendements")

        
    

