"""
Shared data loader and configuration for the Streamlit multi-page app.
All pages import from here to avoid code duplication.
"""

import streamlit as st
import pandas as pd
import os
import json
import glob
from datetime import datetime


# --- CONSTANTS ---
QUADRANT_NAMES = {1: "Growth", 2: "Inflation", 3: "Stagflation", 4: "Deflation"}
QUADRANT_COLORS = {1: 'green', 2: 'orange', 3: 'red', 4: 'blue'}

ALLOCATIONS = {
    1: {"SP500": 0.40, "NASDAQ_100": 0.40, "US_REIT_VNQ": 0.20},
    2: {"GOLD_OZ_USD": 0.40, "NASDAQ_100": 0.38, "COMMODITIES": 0.11, "SP500": 0.11},
    3: {"USD_JPY": 0.40, "SHORT_SP500": 0.32, "USD_EUR": 0.18, "COMMODITIES": 0.10},
    4: {"TREASURY_10Y": 0.40, "GOLD_OZ_USD": 0.38, "OBLIGATION": 0.22}
}


# --- DATA LOADING ---
@st.cache_data(ttl=300)
def load_data():
    # Resolve paths reliably regardless of where Streamlit is run from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    base_dir = os.path.join(project_root, "data", "US")
    data = {}
    
    try:
        data['backtest'] = pd.read_csv(f"{base_dir}/backtest_results/backtest_timeseries.csv", parse_dates=['date'])
    except:
        data['backtest'] = None
    
    try:
        data['stats'] = pd.read_csv(f"{base_dir}/backtest_results/backtest_stats.csv")
    except:
        data['stats'] = None
    
    try:
        data['quadrants'] = pd.read_csv(f"{base_dir}/output_dag/quadrants.csv", parse_dates=['date'])
    except:
        data['quadrants'] = None
    
    try:
        data['perf'] = pd.read_parquet(f"{base_dir}/output_dag/assets_performance_by_quadrant.parquet")
    except:
        data['perf'] = None
        
    try:
        data['perf_smooth'] = pd.read_csv(f"{base_dir}/backtest_results/assets_performance_by_smooth_quadrant.csv")
    except:
        data['perf_smooth'] = None
    
    try:
        data['forex_perf'] = pd.read_parquet(f"{base_dir}/output_dag/forex_performance_by_quadrant.parquet")
    except:
        data['forex_perf'] = None
    
    try:
        data['perf_target'] = pd.read_parquet(f"{base_dir}/output_dag/assets_performance_by_target_quadrant.parquet")
    except:
        data['perf_target'] = None
    
    try:
        data['forex_perf_target'] = pd.read_parquet(f"{base_dir}/output_dag/forex_performance_by_target_quadrant.parquet")
    except:
        data['forex_perf_target'] = None

    try:
        data['daily_assets'] = pd.read_parquet(f"{base_dir}/output_dag/Assets_daily.parquet")
    except:
        data['daily_assets'] = None

    try:
        data['daily_forex'] = pd.read_parquet(f"{base_dir}/output_dag/Forex_daily.parquet")
    except:
        data['daily_forex'] = None
        
    try:
        data['indicators'] = pd.read_csv(f"{base_dir}/output_dag/combined_indicators.csv", parse_dates=['date'])
    except:
        data['indicators'] = None
    
    # --- Load Raw Indicators (for true publication dates & filtering out daily noise) ---
    raw_indicators = []
    # Only keep these specific macroeconomic metrics (exclude WTI, Copper, Interbank, DXY, VIX, Repos)
    MACRO_ALLOWLIST = [
        "INFLATION", "CONSUMER_SENTIMENT", "HOUSING_PERMITS", "IND_PRODUCTION",
        "INITIAL_CLAIMS", "BREAKEVEN_10Y", "High_Yield_Bond_SPREAD", 
        "10-2Year_Treasury_Yield_Bond", "NFCI", "Real_Gross_Domestic_Product", 
    ]
    
    backup_dir = os.path.join(base_dir, "backup")
    if os.path.exists(backup_dir):
        for ind in MACRO_ALLOWLIST:
            file_path = os.path.join(backup_dir, f"{ind}.csv")
            try:
                if os.path.exists(file_path):
                    df_raw = pd.read_csv(file_path)
                    # Needs at least 2 rows to compute delta
                    if len(df_raw) >= 2:
                        # Drop any NaN rows that might be at the end, then take last 2
                        df_raw = df_raw.dropna(subset=['value'])
                        if len(df_raw) >= 2:
                            last_row = df_raw.iloc[-1]
                            prev_row = df_raw.iloc[-2]
                            
                            last_val = float(last_row['value'])
                            prev_val = float(prev_row['value'])
                            last_date = pd.to_datetime(last_row['date'])
                            
                            pct_change = (last_val - prev_val) / abs(prev_val) * 100 if prev_val != 0 else 0
                            
                            raw_indicators.append({
                                'col': ind,
                                'last_val': last_val,
                                'prev_val': prev_val,
                                'pct_change': pct_change,
                                'last_date': last_date
                            })
            except Exception as e:
                pass
                
    # Sort by descending actual publication date
    raw_indicators.sort(key=lambda x: x["last_date"], reverse=True)
    data['recent_indicators'] = raw_indicators
    
    try:
        with open(f"{base_dir}/output_dag/ml_metrics.json", 'r') as f:
            data['ml_metrics'] = json.load(f)
    except:
        data['ml_metrics'] = None
    
    # --- Load IBKR execution logs ---
    nav_history = []
    orders_history = []
    last_positions = None
    last_portfolio_val = None
    prev_quadrant = None
    
    try:
        log_files = sorted(glob.glob(f"{base_dir}/execution_logs/*.json"))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                ts = pd.to_datetime(log_data.get('timestamp'))
                p_val = log_data.get('portfolio_value')
                current_quadrant = log_data.get('quadrant')
                
                # NAV
                if p_val is not None:
                    nav_history.append({'date': ts, 'nav': p_val, 'quadrant': current_quadrant})
                
                # Orders (Always show even if failed, but don't update positions based on it)
                if log_data.get('orders'):
                    reason = "Changement Quadrant" if prev_quadrant and prev_quadrant != current_quadrant else "Rebalancing"
                    for o in log_data['orders']:
                        orders_history.append({
                            'Date': ts,
                            'Action': o.get('action'),
                            'Ticker': o.get('symbol'),
                            'Asset': o.get('asset'),
                            'Shares': o.get('shares'),
                            'Estimated Value ($)': round(o.get('estimated_value', 0), 2),
                            'Reason': reason,
                            'Status': o.get('status', 'Success' if log_data.get('success') else 'Failed'),
                            'Error': o.get('error', '')
                        })
                
                # Last known positions/weights
                # We update this even on failure if we managed to get weights during that attempt
                weights = log_data.get('current_weights')
                if weights:
                    last_positions = weights
                    last_portfolio_val = log_data.get('portfolio_value')
                    last_update_ts = ts
                
                if current_quadrant:
                    prev_quadrant = current_quadrant
            except Exception as e:
                pass
                
        data['ibkr_nav'] = pd.DataFrame(nav_history) if nav_history else pd.DataFrame(columns=['date', 'nav', 'quadrant'])
        data['ibkr_orders'] = pd.DataFrame(orders_history) if orders_history else pd.DataFrame(columns=['Date', 'Action', 'Ticker', 'Asset', 'Shares', 'Estimated Value ($)', 'Reason'])
        data['ibkr_last_positions'] = last_positions
        data['ibkr_last_portfolio_val'] = last_portfolio_val
        
        # Keep only the last NAV per day for cleaner chart
        if not data['ibkr_nav'].empty:
            data['ibkr_nav']['day'] = data['ibkr_nav']['date'].dt.date
            data['ibkr_nav'] = data['ibkr_nav'].drop_duplicates(subset=['day'], keep='last').drop(columns=['day'])
            
    except Exception as e:
        data['ibkr_nav'] = pd.DataFrame(columns=['date', 'nav', 'quadrant'])
        data['ibkr_orders'] = pd.DataFrame(columns=['Date', 'Action', 'Ticker', 'Asset', 'Shares', 'Estimated Value ($)', 'Reason'])
    
    return data


# --- THEME CSS ---
BLOOMBERG_CSS = """
    <style>
    /* Force dark background everywhere */
    .stApp {
        background-color: #0a0e27 !important;
    }
    .main {
        background-color: #0a0e27 !important;
    }
    
    /* Hide Streamlit top header background but keep it clickable for sidebar expand */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 900 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        min-width: 230px !important;
        max-width: 230px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1a1d35 0%, #0a0e27 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* All text white/light */
    .stMarkdown, .stMarkdown p, .stMarkdown span, p, span, div, label {
        color: #e8e8e8 !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #3d4263 !important;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] details {
        background-color: #0a0e27 !important;
        border: 1px solid #3d4263 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #1a1d35 !important;
        color: #00d4ff !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background-color: #2a2d45 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"], 
    [data-testid="stExpanderDetails"] {
        background-color: #0a0e27 !important;
    }
    
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
    
    /* FIX: Selectbox Dropdown rendering issue (invisible text) */
    div[data-baseweb="select"] > div {
        background-color: #1a1d35 !important;
        color: #e8e8e8 !important;
        border: 1px solid #3d4263 !important;
    }
    div[data-baseweb="select"] span {
        color: #e8e8e8 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #1a1d35 !important;
    }
    div[data-baseweb="popover"] li {
        color: #e8e8e8 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #2a2d45 !important;
    }
    
    /* FIX: Selectbox Label & Help Text styling */
    .stSelectbox label p {
        color: #00d4ff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    .stSelectbox div[data-testid="stMarkdownContainer"] p {
        color: #a0a6cc !important;
        font-size: 13.5px !important;
    }
    .stSelectbox .stTooltipIcon svg {
        fill: #00d4ff !important;
        stroke: #00d4ff !important;
    }
    </style>
"""


def apply_theme():
    """Apply the Bloomberg-style dark theme."""
    st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)


def render_sidebar(data):
    """Render the shared sidebar."""
    with st.sidebar:
        st.subheader("État du Pipeline")
        
        if data['quadrants'] is not None:
            last_date = data['quadrants']['date'].max()
            st.success(f"✅ Quadrants: {last_date.strftime('%Y-%m-%d')}")
        else:
            st.error("❌ Quadrants: Non disponible")
        
        if data['backtest'] is not None:
            st.success("✅ Backtest: Disponible")
        else:
            st.warning("⚠️ Backtest: Non disponible")
        
        if data['ml_metrics'] is not None:
            st.success("✅ ML Metrics: Disponible")
        else:
            st.warning("⚠️ ML Metrics: Non disponible")
        
        st.divider()

        if st.button("🔄 Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.subheader("🔧 Actions")
        st.code("airflow dags trigger dag_us_macro", language="bash")
        
        st.divider()
        
