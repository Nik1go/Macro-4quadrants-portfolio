"""
IBKR Module - Configuration
===========================
Configuration settings for Interactive Brokers API connection.
"""

# ===== CONNECTION SETTINGS =====
import os
from ibkr.live_universe import (
    selected_contract_details,
    selected_etf_mapping,
    selected_yfinance_mapping,
)

HOST = "127.0.0.1"          # Localhost (IB Gateway runs on the same VPS via Docker)
PAPER_PORT = 4002           # Paper trading port (IB Gateway default)
LIVE_PORT = 4001            # Live trading port (IB Gateway default)
CLIENT_ID = 1               # Unique client ID for this application
CONNECTION_TIMEOUT = 90     # Timeout in seconds for API connection
ACCOUNT_ID = os.getenv("IBKR_ACCOUNT_ID", "")  # Specific account to manage (e.g., DUO809117)

# Use paper trading by default for safety
CURRENT_PORT = PAPER_PORT

# ===== ETF SYMBOL MAPPING (UCITS / IBKR LIVE UNIVERSE) =====
# Single source of truth lives in ibkr/live_universe.py.
# This keeps execution, Yahoo live-compatible prices, and contract details aligned.
ETF_MAPPING = selected_etf_mapping()
CONTRACT_DETAILS = selected_contract_details()
YFINANCE_MAPPING = selected_yfinance_mapping()

# ===== SAFETY LIMITS (Paper Trading) =====
MAX_ORDER_VALUE_USD = 1000000    # Maximum value per single order (for 1M portfolio)
MAX_TOTAL_POSITION_USD = 1500000  # Maximum total portfolio value
MIN_ORDER_SIZE_USD = 50         # Minimum order to avoid dust

# ===== EXECUTION SETTINGS =====
DRY_RUN_DEFAULT = False         # Execute orders on paper account
ORDER_TYPE = "MKT"              # Market orders for simplicity
REBALANCE_THRESHOLD = 0.02      # 2% - Only rebalance if delta > threshold

# ===== SCHEDULING (Market Hours) =====
# US Market Hours (Eastern Time)
US_MARKET_OPEN_ET = "09:30"     # 9:30 AM ET = 15:30 CET
US_MARKET_CLOSE_ET = "16:00"    # 4:00 PM ET = 22:00 CET

# DAG Schedule: 16:00 CET (15:00 UTC) - 30 min after US market open
# Good liquidity, reasonable spreads
DAG_SCHEDULE_CRON = "0 15 * * 1-5"  # Mon-Fri at 15:00 UTC = 16:00 CET

# Execution timing options:
# - "immediate": Execute right after backtest (pre-market if before open)
# - "market_open": Wait for US market open (15:35 CET)
# - "manual": Don't auto-execute, wait for manual trigger
EXECUTION_TIMING = "market_open"

# Skip execution on weekends/holidays
SKIP_WEEKENDS = True
SKIP_US_HOLIDAYS = True

# ===== DATA PATHS (Fallback) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DATA_PATH = os.path.join(BASE_DIR, "data", "US", "output_dag", "Assets_daily.parquet")
FOREX_DATA_PATH = os.path.join(BASE_DIR, "data", "US", "output_dag", "Forex_daily.parquet")
LIVE_ASSETS_DATA_PATH = os.path.join(BASE_DIR, "data", "US", "backtest_results", "ibkr_live_prices.csv")


