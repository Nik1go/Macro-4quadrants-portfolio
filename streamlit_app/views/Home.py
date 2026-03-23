import streamlit as st
from streamlit_option_menu import option_menu

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
            background-color: #0a0e27 !important;
            background-image: none !important;
        }
        [data-testid="stSidebar"] * {
            color: #e8e8e8 !important;
        }
        
        /* Hide Streamlit top header and top margin padding */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* Option menu - blend into sidebar with rounded white border */
        .nav-link {
            color: #e8e8e8 !important;
            border-radius: 10px !important;
        }
        div[data-testid="stSidebar"] .css-j7qwjs,
        div[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(> iframe) {
            background-color: transparent !important;
        }
        nav.menu {
            background-color: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            padding: 8px !important;
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
        st.image("https://img.icons8.com/fluency/96/000000/python.png", width=80)
        st.markdown("### 📊 Quantitative Finance Portfolio")
        st.markdown("---")
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Pipeline Macro-Quantitative", "Equity Research"],
            icons=["house-fill", "graph-up-arrow", "file-earmark-text", "journal-code"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#00d4ff", "font-size": "20px"},
                "nav-link": {
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
        st.markdown("### 🔧 Tech Stack")
        st.markdown("""
        - Python 🐍
        - Streamlit 🎈
        - Pandas & NumPy 📊
        - Plotly & Matplotlib 📈
        - VectorBT 📉
        - Apache Airflow 🔄
        - Scikit-learn 🤖
        """)
        
        st.markdown("---")
        st.caption("Developed by Leo")
        if st.button("Recharger les donnees"):
            st.cache_data.clear()
            st.rerun()

        return selected

def render_home():
    apply_home_css()
    
    st.title(" Quantitative Finance Portfolio")
    st.markdown("###  Quantitative Analyst | Data Engineer")
    st.markdown("---")
    
    # Profile Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/fluency/256/000000/financial-analytics.png", width=200)
        st.markdown("""
        <div class="metric-container">
            <h4>📍 Profile</h4>
            <p><strong>Location:</strong> Paris, Sud de</p>
            <p><strong>Focus:</strong> Quantitative Finance</p>
            <p><strong>Experience:</strong> Python, Data Engineering, Financial Modeling</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💻 About Me")
        st.markdown("""
        Passionate engineer who likes building financial models, 
        data pipelines, and some fun trading strategies. Every thing i know is 

        
        **Core Competencies:**
        - 📊 Macro quantitative Analysis & Statistical Modeling
        - 🤖 Algorithmic Trading & Backtesting
        - 🔄 Data Engineering & ETL Pipelines (Apache Airflow)
        - 📈 Financial Derivatives & Risk Management
        - 🐍 Python Development
        """)
        

    st.markdown("---")
    
    # Projects Grid
    st.markdown("### 📂 Featured Projects")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>🌍 Macro Strategy Pipeline</h3>
            <p><strong>Tech:</strong> Apache Airflow, Random Forest, Spark, IBKR  </p>
            <p>Designed ETL pipeline ingesting macro indicators via Airflow DAGs and procces them with Random Forest ML model. 
            Orchestrated "4 Seasons" strategy based on probability of growth/inflation regime detection.</p>
            <p><strong>Key Achievement:</strong> Automated portfolio management less volatile than the market with real-time streamlit dashboards.</p>
            <hr>
            <p>🔄 Airflow | 📊 ETL | 📡 Macro ML</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet Macro", key="btn_macro", use_container_width=True):
            st.session_state.current_page = "Pipeline Macro-Quantitative"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>🎲 Monte Carlo Gambling Strategies</h3>
            <p><strong>Tech:</strong> NumPy, Matplotlib, Plotly</p>
            <p>Simulated 1000+ roulette games using Martingale strategy with vectorized operations. 
            Achieved statistical analysis showing 82% win rate but negative expected value.</p>
            <p><strong>Key Result:</strong> Demonstrated the mathematical impossibility of beating the house edge.</p>
            <hr>
            <p>📊 Monte Carlo | 🎯 Risk Analysis </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_mc", use_container_width=True):
            pass  # placeholder
    
    with col3:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>💰 DCA Investment Strategy</h3>
            <p><strong>Tech:</strong> Pandas, yFinance, VectorBT</p>
            <p>Built a Dollar Cost Averaging backtesting framework for SP500, Gold, and Bitcoin. 
            Implemented bi-weekly rebalancing with Sharpe ratio optimization.</p>
            <p><strong>Key Result:</strong> Systematized long-term investment strategy with risk-adjusted returns.</p>
            <hr>
            <p>💵 Portfolio Mgmt | 📉 Backtesting | ⚖️ Risk/Reward</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_dca", use_container_width=True):
            pass # placeholder
    
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>🥇 Pairs Trading Arbitrage</h3>
            <p><strong>Tech:</strong> Statsmodels, Scikit-learn, Scipy</p>
            <p>Developed pairs trading strategy for Gold-Silver correlation. Applied cointegration tests 
            (Augmented Dickey-Fuller) and linear regression for spread modeling.</p>
            <p><strong>Key Result:</strong> Statistical arbitrage based on mean reversion principles.</p>
            <hr>
            <p>🔗 Correlation | 📊 Cointegration | 🎯 Arbitrage</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_pairs", use_container_width=True):
            pass # placeholder

    with col5:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>🚀 Crypto Momentum Trading</h3>
            <p><strong>Tech:</strong> Binance API, VectorBT, Plotly</p>
            <p>Momentum-based crypto strategy detecting "hype cycles" using rolling z-scores. 
            Trades top performer among 15 cryptos when BTC shows strength.</p>
            <p><strong>Key Result:</strong> Captured volatility during bullish crypto trends.</p>
            <hr>
            <p>₿ Crypto | 📈 Momentum | 🔥 Trend Following</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_crypto", use_container_width=True):
            pass # placeholder
    
    
    with col6:
        st.markdown("""
        <div class="project-card" style="margin-bottom: 0px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <h3>📊 Financial Modeling Suite</h3>
            <p><strong>Tech:</strong> NumPy, Pandas, LaTeX (for formulas)</p>
            <p>Built comprehensive equity valuation models including DCF, multiples analysis, 
            and options pricing (Black-Scholes). Interactive sensitivity analysis with Plotly.</p>
            <p><strong>Key Result:</strong> End-to-end equity research framework.</p>
            <hr>
            <p>💼 DCF | 📐 Valuation | 🎓 Financial Theory</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Voir le Projet", key="btn_finmod", use_container_width=True):
            st.session_state.current_page = "Equity Research"
            st.rerun()
    
    st.markdown("---")
    
    # Contact Section
    st.markdown("### 📬 Let's Connect")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("[![GitHub](https://img.icons8.com/fluency/48/000000/github.png)](https://github.com)")
        st.markdown("[**GitHub**](https://github.com)")
    
    with col2:
        st.markdown("[![LinkedIn](https://img.icons8.com/fluency/48/000000/linkedin.png)](https://linkedin.com)")
        st.markdown("[**LinkedIn**](https://linkedin.com)")
    
    with col3:
        st.markdown("[![Email](https://img.icons8.com/fluency/48/000000/email.png)](mailto:contact@example.com)")
        st.markdown("[**Email**](mailto:contact@example.com)")
