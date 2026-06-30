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

        /* Fix Expander Background */
        [data-testid="stExpander"] {
            background-color: #1e2139 !important;
            border: 1px solid #3d4263 !important;
            border-radius: 12px !important;
            padding: 0px 10px 5px 10px !important;
        }
        [data-testid="stExpander"] summary {
            background-color: transparent !important;
            color: #00d4ff !important;
            font-weight: 600 !important;
        }
        /* Ensure content stays dark when expanded */
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"], 
        [data-testid="stExpander"] div[role="region"] {
            background-color: transparent !important;
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
        
        options = ["Home", "Macro 4 seasons strategy", "Crypto Momentum Trading", "Monte Carlo Gambling", "DCA Investment Strategy", "Pairs trading", "Equity Research","Portfolio Optimizer", "Polymarket Arbitrage", "Mon Portefeuille"]
        try:
            default_idx = options.index(st.session_state.current_page)
        except ValueError:
            default_idx = 0
            
        def on_menu_change(key):
            st.session_state.current_page = st.session_state[key]
            
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=["house-fill", "safe", "coin", "dice-5", "hourglass-split", "arrows-expand", "robot", "bezier","activity", "wallet2"], 
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
    col1, col2 = st.columns([1.4, 2])
    
    with col1:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "portfoliotete.jpeg")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.image("https://img.icons8.com/fluency/256/000000/financial-analytics.png", width=200)
        
    with col2:
        st.markdown("<h3 style='margin-top: 0;'>About Me</h3>", unsafe_allow_html=True)
        st.markdown("""
        **Passionate engineering student with an insatiable curiosity and a love for learning, particularly in financial markets and macroeconomics.** 
        I enjoy bridging the gap between technology and finance by building data pipelines, quantitative analysis models, and algorithmic strategies. 
        
        As I approach the end of my engineering degree (July 2026), I am fully committed to transitioning into the financial industry. **I am currently in the interview process for a Master's in Finance at top French business schools (NEOMA, Albert School).** 
        
        **I am actively seeking an apprenticeship position for the 2026-2027 academic year** to apply my engineering background while developing advanced financial expertise.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-container">
        <h4>Profile</h4>
        <p><strong>Location:</strong> Paris & South of France</p>
        <p><strong>Focus:</strong> Quantitative Finance, Data Engineering & Macroeconomic Analysis</p>
        <p><strong>Core Skills:</strong> Python, Data Pipelines, Machine Learning</p>
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
            st.session_state.current_page = "Macro 4 seasons strategy"
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
            <h3><i class="bi bi-arrows-expand"></i> Pairs trading Statistical Arbitrage</h3>
            <p><strong>Tech:</strong> Statsmodels, Scikit-learn, Scipy</p>
            <p>Developed pairs trading strategy for Gold-Silver correlation. Applied cointegration tests 
            (Augmented Dickey-Fuller) and linear regression for spread modeling.</p>
            <p><strong>Key Result:</strong> Pairs trading strategy based on mean reversion principles.</p>
            <hr>   
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_pairs", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Pairs trading"
            st.rerun()
    
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-hourglass-split"></i> DCA Investment Strategy</h3>
            <p><strong>Tech:</strong> Pandas, yFinance, NumPy</p>
            <p>Backtesting and optimization of Dollar Cost Averaging framework for SP500 and Gold. 
            Implemented bi-weekly rebalancing with z-score optimization to improve entry points.</p>
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
            <p>Portfolio optimization using Markowitz theory and SciPy. Includes Monte Carlo 
            simulations to visualize the Efficient Frontier and find the optimal Risk/Reward balance.</p>
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
            <p><strong>Tech:</strong> Python, Black-Scholes, WebSockets</p>
            <p>The objective is to detect arbitrage opportunities between Polymarket and Binance using the Black-Scholes model, assuming Polymarket offers binary forward contracts.</p>
            <p><em>(Building in progress)</em></p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_poly", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Polymarket Arbitrage"
            st.rerun()
            
    with col8:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3><i class="bi bi-wallet2"></i> Mon Portefeuille</h3>
            <p><strong>Tech:</strong> Python, Streamlit, yFinance</p>
            <p>Portfolio tracker. Live price tracking and performance calculations.</p>
            <hr>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_portfolio", use_container_width=True):
            st.session_state.pop("main_menu", None)
            st.session_state.current_page = "Mon Portefeuille"
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

    elif st.session_state.current_page == "Macro 4 seasons strategy":
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


    elif st.session_state.current_page == "Pairs trading":
        _render_nav_sidebar("Pairs trading")
        try:
            apply_home_css()
            import arbitrage.app_arbitrage as app_arbitrage
            app_arbitrage.render()
        except Exception as e:
            st.error(f"Could not load the Arbitrage app: {str(e)}")

    elif st.session_state.current_page == "Crypto Momentum Trading":
        _render_nav_sidebar("Crypto Momentum Trading")
        try:
            apply_home_css()
            import momentum_BTC.app_momentum as app_momentum
            app_momentum.render()
        except Exception as e:
            st.error(f"Could not load the Crypto Momentum Trading app: {str(e)}")
    
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
    

    elif st.session_state.current_page == "Mon Portefeuille":
        _render_nav_sidebar("Mon Portefeuille")
        try:
            apply_home_css()
            import portfolio_tracker.app_portfolio as app_portfolio
            app_portfolio.render()
        except Exception as e:
            st.error(f"Could not load the Portfolio Tracker app: {str(e)}")

    else:
        st.session_state.current_page = "Home"
        st.rerun()

