import sys
import os
import streamlit as st
from streamlit_option_menu import option_menu

# --- PAGE CONFIG (only once) ---
st.set_page_config(
    layout="wide",
    page_title="Quantitative Finance Portfolio",
    initial_sidebar_state="auto"
)

# Add project root to path for core logic imports (bot, models, etc.)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Existing path logic for local pages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize session state for routing
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not installed. Some visualizations may be limited.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from apache_airflow import DAG
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

def apply_home_css():
    st.markdown("""
        <style>
        @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");
        /* Main theme - same as macro page */
        .stApp {
            background-color: #0a0e27 !important;
        }
        .main {
            background-color: #0a0e27 !important;
        }
        
        /* All text white/light */
        .stMarkdown, .stMarkdown p, .stMarkdown span, p, span, div, label {
            color: #e8e8e8 !important;
        }
        
        /* Card styling */
        .project-card {
            background: linear-gradient(135deg, #1e2139 0%, #2a2d4a 100%);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #3d4263;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            margin: 10px 0;
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 420px;
        }
        
        .project-card:hover {
            transform: translateY(-5px);
            border-color: #00d4ff;
        }
        
        /* Card text - white / light grey */
        .project-card h3 {
            color: #00d4ff !important;
        }
        .project-card p, .project-card strong {
            color: #d0d0d0 !important;
        }
        .project-card hr {
            border-color: #3d4263 !important;
        }
        
        /* Metrics styling */
        .metric-container {
            background: rgba(30, 33, 57, 0.6);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #00d4ff;
        }
        .metric-container h4, .metric-container p, .metric-container strong {
            color: #e8e8e8 !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00d4ff !important;
            font-weight: 600;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            min-width: 230px !important;
            max-width: 230px !important;
            background-color: #0a0e27 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #0a0e27 !important;
            background-image: none !important;
        }
        [data-testid="stSidebar"] * {
            color: #e8e8e8 !important;
        }
        
        /* Hide Streamlit top header background but keep it clickable for sidebar expand */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* Button styling - Merge tightly to the project card */
        .stButton>button {
            background: linear-gradient(90deg, #1e2139 0%, #2a2d4a 100%);
            color: #00d4ff !important;
            border: 1px solid #3d4263;
            border-top: none;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
            padding: 12px 24px;
            font-weight: 700;
            margin-top: -10px; /* Pull the button up to attach seamlessly to the card */
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
            color: white !important;
            border-color: #00d4ff;
            transform: scale(1.02);
            z-index: 10;
        }
        
        /* Divider */
        hr {
            border-color: #3d4263 !important;
        }
        
        /* Metric values */
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        [data-testid="stMetricLabel"] {
            color: #b0b0b0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "logo.png")
        if os.path.exists(logo_path):
            import base64
            with open(logo_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f'<div style="text-align: center;">\n'
                f'    <img src="data:image/png;base64,{encoded_string}" style="width: 110px; max-width: 100%; margin-bottom: 20px;">\n'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("<h3 style='text-align: center; margin-top: -10px;'>Finance Portfolio</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        options = ["Home", "Pipeline Macro-Quantitative", "Crypto Momentum Trading", "Monte Carlo Gambling", "DCA Investment Strategy", "Arbitrage Or-Argent", "Equity Research","Portfolio Optimization", "Polymarket Arbitrage"]
        try:
            default_idx = options.index(st.session_state.current_page)
        except ValueError:
            default_idx = 0
            
        def on_menu_change(key):
            st.session_state.current_page = st.session_state[key]
            
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=["house-fill", "safe", "coin", "dice-5", "hourglass-split", "arrows-expand", "robot", "bezier","activity"], 
            #bootstrap icons (7 icons for 7 options)
            menu_icon="cast",
            default_index=default_idx,
            manual_select=default_idx,
            on_change=on_menu_change,
            key="main_menu",
            styles={
                "container": {
                    "padding": "0!important", 
                    "background-color": "#0a0e27",
                    "border-radius": "0"
                },
                "icon": {"color": "#00d4ff", "font-size": "20px"},
                "nav-link": {
                    "color": "#e8e8e8",
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "padding": "10px",
                    "--hover-color": "#1e2139",
                },
                "nav-link-selected": {"background-color": "#00d4ff", "color": "white"},
            }
        )
        
        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("""
        - Python 
        - Streamlit 
        - Pandas & NumPy 
        - Plotly & Matplotlib 
        - VectorBT 
        - Apache Airflow 
        - Scikit-learn 
        """)
        
        st.markdown("---")
        st.caption("Developed by Leo")
        if st.button("Recharger les donnees"):
            st.cache_data.clear()
            st.rerun()

        return selected

def render_home():
    apply_home_css()
    
    st.title("Quantitative Finance Portfolio")
    st.markdown("### Quantitative Analyst | Data Engineer")
    st.markdown("---")
    
    # Profile Section
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "portfoliotete.jpeg")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.image("https://img.icons8.com/fluency/256/000000/financial-analytics.png", width=200)
        
    with col2:
        st.markdown("### About Me")
        st.markdown("""
        Passionate engineer who likes building financial models, 
        data pipelines, and some fun trading strategies.""")

        st.markdown("""
        <div class="metric-container">
            <h4>Profile</h4>
            <p><strong>Location:</strong> Paris et Sud de France </p>
            <p><strong>Focus:</strong> Macro-Quantitative</p>
            <p><strong>Experience:</strong> Python, Data Engineering, Macroeconomie</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    
    # Projects Grid
    st.markdown("### Featured Projects")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-safe"></i> Macro 4 seasons Strategy</h3>
            <p><strong>Tech:</strong> Apache Airflow, Random Forest ML, Spark, IBKR  </p>
            <p>ETL pipeline ingesting and processing macro indicators via Airflow. 
            Orchestrated and backtested strategy based on probability of growth/inflation regime detection.</p>
            <p><strong>Key Achievement:</strong> Portfolio management less volatile than the market.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_macro", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Pipeline Macro-Quantitative"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-coin"></i> Crypto Momentum</h3>
            <p><strong>Tech:</strong> Binance API, VectorBT, Plotly</p>
            <p>Momentum-based crypto live trading strategy detecting "hype cycles" in crypto market to trade altcoins. 
            Main idea is to follow BTC trend: long the strongest altcoins, short the weakest.</p>
            <p><strong>Key Result:</strong> Captured volatility during hype cycles.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_crypto", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Crypto Momentum Trading"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-arrows-expand"></i> Statistical Arbitrage</h3>
            <p><strong>Tech:</strong> Statsmodels, Scikit-learn, Scipy</p>
            <p>Developed pairs trading strategy for Gold-Silver correlation. Applied cointegration tests 
            (Augmented Dickey-Fuller) and linear regression for spread modeling.</p>
            <p><strong>Key Result:</strong> Statistical arbitrage based on mean reversion principles.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_pairs", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Arbitrage Or-Argent"
            st.rerun()
    
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-hourglass-split"></i> DCA Investment Strategy</h3>
            <p><strong>Tech:</strong> Pandas, yFinance, NumPy</p>
            <p>Trying to improve Dollar Cost Averaging backtesting framework for SP500, Gold, and Bitcoin. 
            Implemented bi-weekly rebalancing with z-score optimisation.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_dca", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "DCA Investment Strategy"
            st.rerun()

    
    with col5:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-dice-5"></i> Martingale Gambling</h3>
            <p><strong>Tech:</strong> Python, NumPy, Plotly</p>
            <p>Monte carlo simulation of roulette games using Martingale strategy. 
            Achieved statistical analysis showing high win rate doesn't mean positive expected value.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_mc", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Monte Carlo Gambling"
            st.rerun()
    

    with col6:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-bezier"></i> Portfolio Optimizer</h3>
            <p><strong>Tech:</strong> Python, NumPy, Plotly</p>
            <p>Optimization of a portfolio of assets using Monte Carlo simulation and genetic algorithms. 
            Achieved statistical analysis showing high win rate doesn't mean positive expected value.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_opti", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Portfolio Optimizer"
            st.rerun()
        
    
    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-activity"></i> Polymarket Arbitrage</h3>
            <p>l'objectif est de décelé grace a black-scholes des opportunité d'arbitrage entre polymarket et Binance.
            En partant du principe que polymarket propose des contrat forward binaire 
            </p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_poly", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Polymarket Arbitrage"
            st.rerun()
    
   
    st.markdown("---")
    
    # Contact Section
    st.markdown("### Let's Connect")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("[![GitHub](https://img.icons8.com/fluency/48/000000/github.png)](https://github.com/Nik1go)")
        st.markdown("[**GitHub**](https://github.com/Nik1go)")
    
    with col2:
        st.markdown("[![LinkedIn](https://img.icons8.com/fluency/48/000000/linkedin.png)](https://www.linkedin.com/in/leo-bertrand-link/)")
        st.markdown("[**LinkedIn**](https://www.linkedin.com/in/leo-bertrand-link/)")
    

# --- ROUTING LOGIC ---
def _render_nav_sidebar(current_page_name):
    """Renders the sidebar option_menu."""
    render_sidebar()

if __name__ == "__main__":
    if st.session_state.current_page == "Home":
        _render_nav_sidebar("Home")
        render_home()

    elif st.session_state.current_page == "Pipeline Macro-Quantitative":
        import macro_projet.app_macro as app_macro
        app_macro.render()

    elif st.session_state.current_page == "Monte Carlo Gambling":
        _render_nav_sidebar("Monte Carlo Gambling")
        try:
            apply_home_css()
            import montecarlo_gambling.app_montecarlo as app_montecarlo
            app_montecarlo.render()
        except Exception as e:
            st.error(f"Could not load the Monte Carlo app: {str(e)}")

    elif st.session_state.current_page == "DCA Investment Strategy":
        _render_nav_sidebar("DCA Investment Strategy")
        try:
            apply_home_css()
            import dca_strat.app_dca as app_dca
            app_dca.render()
        except Exception as e:
            st.error(f"Could not load the DCA Strategy app: {str(e)}")

    elif st.session_state.current_page == "Crypto Momentum Trading":
        _render_nav_sidebar("Crypto Momentum Trading")
        try:
            apply_home_css()
            import momentum_BTC.app_momentum as app_momentum
            app_momentum.render()
        except Exception as e:
            st.error(f"Could not load the Crypto Momentum Trading app: {str(e)}")

    elif st.session_state.current_page == "Arbitrage Or-Argent":
        _render_nav_sidebar("Arbitrage Or-Argent")
        try:
            apply_home_css()
            import arbitrage.app_arbitrage as app_arbitrage
            app_arbitrage.render()
        except Exception as e:
            st.error(f"Could not load the Arbitrage app: {str(e)}")
    
    elif st.session_state.current_page == "Portfolio Optimizer":
        _render_nav_sidebar("Portfolio Optimizer")
        try:
            apply_home_css()
            import portfolio_optimizer.app_optimizer as app_optimizer
            app_optimizer.render()
        except Exception as e:
            st.error(f"Could not load the Portfolio Optimizer app: {str(e)}")


    elif st.session_state.current_page == "Polymarket Arbitrage":
        _render_nav_sidebar("Polymarket Arbitrage")
        try:
            apply_home_css()
            import polymarket_arbitrage_ui.app_poly_arb as app_polymarket
            app_polymarket.render()
        except Exception as e:
            st.error(f"Could not load the Polymarket Arbitrage app: {str(e)}")
            st.info("Check if polymarket_arbitrage_ui is correctly installed in streamlit_app/.")

    elif st.session_state.current_page == "Equity Research":
        _render_nav_sidebar("Equity Research")
        st.title("Equity Research")
        st.write("Page under construction...")
    

    else:
        st.session_state.current_page = "Home"
        st.rerun()

