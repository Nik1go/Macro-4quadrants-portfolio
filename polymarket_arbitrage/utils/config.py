"""Centralized configuration for the Polymarket arbitrage system.

All runtime parameters are loaded from environment variables with robust
validation and sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

dotenv_file = os.getenv("DOTENV_FILE", ".env")
load_dotenv(dotenv_file, override=True)


class Config:
    """Static configuration holder for the trading and visualization services."""

    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))
    BACKTEST_DIR: Path = Path(os.getenv("BACKTEST_DIR", str(BASE_DIR / "backtests")))

    DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "arbitrage.db"))
    TRADES_CSV_PATH: str = os.getenv("TRADES_CSV_PATH", str(DATA_DIR / "trades.csv"))
    METRICS_JSON_PATH: str = os.getenv("METRICS_JSON_PATH", str(DATA_DIR / "metrics.json"))
    MACRO_DATA_PATH: str = os.getenv(
        "MACRO_DATA_PATH",
        str(BASE_DIR.parent / "data" / "US" / "backup" / "TAUX_FED.csv"),
    )

    STATE_FILE_PATH: str = os.getenv("STATE_FILE_PATH", str(DATA_DIR / "bot_state.json"))
    LOCK_FILE_PATH: str = os.getenv("LOCK_FILE_PATH", str(DATA_DIR / "bot.lock"))

    PAPER_TRADE: bool = os.getenv("PAPER_TRADE", "true").lower() in {"true", "1", "yes"}
    STRATEGY_NAME: str = os.getenv("STRATEGY_NAME", "delta_neutral").strip().lower()
    ENABLE_NOTIFIER: bool = os.getenv("ENABLE_NOTIFIER", "false").lower() in {"true", "1", "yes"}

    EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
    BINANCE_DEFAULT_TYPE: str = os.getenv("BINANCE_DEFAULT_TYPE", "future")

    TARGET_ASSETS: List[str] = [
        item.strip()
        for item in os.getenv("TARGET_ASSETS", "Bitcoin,Ethereum,XRP").split(",")
        if item.strip()
    ]
    ASSET_TO_SYMBOL: Dict[str, str] = {
        "Bitcoin": os.getenv("SYMBOL_BITCOIN", "BTC/USDT"),
        "Ethereum": os.getenv("SYMBOL_ETHEREUM", "ETH/USDT"),
        "XRP": os.getenv("SYMBOL_XRP", "XRP/USDT"),
    }

    SCAN_YEAR: str = os.getenv("SCAN_YEAR", "2026")
    SCAN_INTERVAL: int = max(int(os.getenv("SCAN_INTERVAL", "60")), 5)
    MAX_MARKETS_PER_ASSET: int = max(int(os.getenv("MAX_MARKETS_PER_ASSET", "100")), 1)
    MAX_MARKETS_PER_DAY: int = max(int(os.getenv("MAX_MARKETS_PER_DAY", "15")), 1)
    MIN_TTM_DAYS: int = max(int(os.getenv("MIN_TTM_DAYS", "0")), 0)
    MAX_TTM_DAYS: int = max(int(os.getenv("MAX_TTM_DAYS", "7")), 1)

    MIN_BATCH_EDGE: float = float(os.getenv("MIN_BATCH_EDGE", "0.04"))
    MAX_TRADES_PER_ROUND: int = max(int(os.getenv("MAX_TRADES_PER_ROUND", "5")), 1)

    MIN_SPREAD_THRESHOLD: float = float(os.getenv("MIN_SPREAD_THRESHOLD", "0.02"))
    DELTA_NEUTRAL_MIN_EDGE: float = float(os.getenv("DELTA_NEUTRAL_MIN_EDGE", "0.015"))
    PURE_POLY_MIN_EDGE: float = float(os.getenv("PURE_POLY_MIN_EDGE", "0.025"))
    DELTA_REBALANCE_THRESHOLD: float = float(os.getenv("DELTA_REBALANCE_THRESHOLD", "0.15"))
    MAX_FUNDING_RATE_ABS: float = float(os.getenv("MAX_FUNDING_RATE_ABS", "0.005"))
    MIN_POLY_LIQUIDITY_USD: float = max(float(os.getenv("MIN_POLY_LIQUIDITY_USD", "200.0")), 0.0)
    MAX_SLIPPAGE_TO_EDGE_RATIO: float = max(float(os.getenv("MAX_SLIPPAGE_TO_EDGE_RATIO", "0.5")), 0.0)

    MAX_POSITION_SIZE: float = max(float(os.getenv("MAX_POSITION_SIZE", "1000")), 1.0)
    MIN_POSITION_SIZE: float = max(float(os.getenv("MIN_POSITION_SIZE", "10")), 0.0)
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.5"))
    MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.5"))
    INITIAL_CAPITAL: float = max(float(os.getenv("INITIAL_CAPITAL", "10000")), 100.0)
    MAX_POSITION_PER_ASSET_PCT: float = float(os.getenv("MAX_POSITION_PER_ASSET_PCT", "0.35"))

    EXCHANGE_TAKER_FEE: float = float(os.getenv("EXCHANGE_TAKER_FEE", "0.0004"))
    POLYMARKET_FEE: float = float(os.getenv("POLYMARKET_FEE", "0.0"))
    GAS_FEE_MATIC: float = float(os.getenv("GAS_FEE_MATIC", "0.01"))

    RETRY_ATTEMPTS: int = max(int(os.getenv("RETRY_ATTEMPTS", "5")), 1)
    RETRY_BASE_DELAY: float = max(float(os.getenv("RETRY_BASE_DELAY", "0.35")), 0.01)
    RETRY_MAX_DELAY: float = max(float(os.getenv("RETRY_MAX_DELAY", "5.0")), 0.1)
    RETRY_JITTER: float = max(float(os.getenv("RETRY_JITTER", "0.25")), 0.0)
    CIRCUIT_BREAKER_THRESHOLD: int = max(int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")), 1)
    CIRCUIT_BREAKER_COOLDOWN: int = max(int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "45")), 5)

    OUTLIER_ZSCORE_THRESHOLD: float = max(float(os.getenv("OUTLIER_ZSCORE_THRESHOLD", "3.5")), 0.1)
    OUTLIER_IQR_MULTIPLIER: float = max(float(os.getenv("OUTLIER_IQR_MULTIPLIER", "1.5")), 0.1)

    HEALTH_HOST: str = os.getenv("HEALTH_HOST", "0.0.0.0")
    HEALTH_PORT: int = max(int(os.getenv("HEALTH_PORT", "8080")), 1024)

    STREAMLIT_HOST: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    STREAMLIT_PORT: int = max(int(os.getenv("STREAMLIT_SERVER_PORT", "8501")), 8501)
    STREAMLIT_BACKTEST_MIN_DAYS: int = max(int(os.getenv("STREAMLIT_BACKTEST_MIN_DAYS", "30")), 7)
    STREAMLIT_MAX_ROWS: int = max(int(os.getenv("STREAMLIT_MAX_ROWS", "10000")), 100)

    BACKTEST_INITIAL_CAPITAL: float = max(float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000")), 100.0)
    BACKTEST_MAX_DRAWDOWN: float = float(os.getenv("BACKTEST_MAX_DRAWDOWN", "0.2"))
    BACKTEST_STOP_LOSS: float = float(os.getenv("BACKTEST_STOP_LOSS", "0.05"))

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    EMAIL_HOST: str = os.getenv("EMAIL_HOST", "")
    EMAIL_PORT: int = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USERNAME: str = os.getenv("EMAIL_USERNAME", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_TO: str = os.getenv("EMAIL_TO", "")

    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")

    POLYMARKET_RPC_URL: str = os.getenv("POLYMARKET_RPC_URL", "")
    POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    POLYMARKET_EXCHANGE_ADDRESS: str = os.getenv("POLYMARKET_EXCHANGE_ADDRESS", "")
    POLYMARKET_CHAIN_ID: int = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

    @classmethod
    def validate(cls) -> None:
        """Validate startup configuration and create required directories."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

        if cls.STRATEGY_NAME not in {"delta_neutral", "pure_polymarket"}:
            raise ValueError(
                "STRATEGY_NAME must be one of {'delta_neutral', 'pure_polymarket'}."
            )

        if not (0 < cls.MAX_DRAWDOWN_PCT < 1):
            raise ValueError("MAX_DRAWDOWN_PCT must be in (0, 1).")

        if not (0 < cls.STOP_LOSS_PCT < 1):
            raise ValueError("STOP_LOSS_PCT must be in (0, 1).")

        if cls.MIN_SPREAD_THRESHOLD < 0:
            raise ValueError("MIN_SPREAD_THRESHOLD cannot be negative.")

        if cls.MAX_POSITION_PER_ASSET_PCT <= 0 or cls.MAX_POSITION_PER_ASSET_PCT > 1:
            raise ValueError("MAX_POSITION_PER_ASSET_PCT must be in (0, 1].")

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Expose safe runtime configuration for logging and observability."""
        return {
            "paper_trade": cls.PAPER_TRADE,
            "strategy_name": cls.STRATEGY_NAME,
            "exchange_id": cls.EXCHANGE_ID,
            "scan_interval": cls.SCAN_INTERVAL,
            "min_spread_threshold": cls.MIN_SPREAD_THRESHOLD,
            "delta_neutral_min_edge": cls.DELTA_NEUTRAL_MIN_EDGE,
            "pure_poly_min_edge": cls.PURE_POLY_MIN_EDGE,
            "max_funding_rate_abs": cls.MAX_FUNDING_RATE_ABS,
            "min_poly_liquidity_usd": cls.MIN_POLY_LIQUIDITY_USD,
            "max_slippage_to_edge_ratio": cls.MAX_SLIPPAGE_TO_EDGE_RATIO,
            "initial_capital": cls.INITIAL_CAPITAL,
            "max_position_size": cls.MAX_POSITION_SIZE,
            "max_position_per_asset_pct": cls.MAX_POSITION_PER_ASSET_PCT,
            "db_path": cls.DB_PATH,
            "min_batch_edge": cls.MIN_BATCH_EDGE,
            "max_trades_per_round": cls.MAX_TRADES_PER_ROUND,
            "min_ttm_days": cls.MIN_TTM_DAYS,
            "max_ttm_days": cls.MAX_TTM_DAYS,
            "backtest_dir": str(cls.BACKTEST_DIR),
        }
