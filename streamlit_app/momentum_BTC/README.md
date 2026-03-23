# Crypto Momentum Project

## Core Objective
Quantitative trading strategy exploiting BTC-driven fashion/wealth effects on altcoins. High-conviction strategy (100% equity in one position) based on relative momentum and trend regimes.

## Architecture & Data Lifecycle
- **Automation**: Apache Airflow (Daily at 00H05 UTC).
- **Processing**: Apache Spark for feature engineering and performance optimization.
- **Execution**: Interactive Brokers API (IBKR) for automated order placement.
- **Data Sources**: FRED (Macro), Binance API (Crypto OHLCV), Yahoo Finance.

## Project Structure & File Map
- `dags/`: Airflow DAGs orchestrated via Apache Airflow.
    - `dag_crypto_momentum.py`: Daily pipeline entry point.
- `spark_jobs/`: Distributed data processing logic (feature engineering, backtests).
- `ibkr/`: Interactive Brokers (Trader Workstation) integration logic.
- `data/crypto/`: Project data storage.
    - `ALT_USDT/` / `ALT_BTC/`: OHLCV CSV persistence.
    - `backtest_results/`: Serialized backtest performance and parameters.
- `momentum_BTC/`: Main project logic directory (Streamlit + logic helpers).
    - `app_momentum.py`: Streamlit dashboard for monitoring and backtest visualization.
    - `momentum_utils.py`: Backtesting logic (VectorBT), data downloading, and rolling universe management.
    - `data_fetcher.py`: Standardized data ingestion pipeline.
    - `indicators/calc_indicators.py`: Primitive indicators (SMA, Skewness, ATR).
    - `signals/generate_signals.py`: Entry/Exit signal logic and asset selection criteria.
    - `execution/`: IBKR order execution and state management.
    - `backtest/`: Storage for `backtest_summary.json` (optimal params) and CSV results.

## AI Navigation & Maintenance
- **Logic Updates**: To modify entry/exit conditions, investigate `signals/generate_signals.py`.
- **Parameter Tuning**: Current "Best" params (SMA length, lookbacks) are stored in `data/crypto/backtest_results/backtest_summary.json`.
- **Data Schema**: All crypto data is stored in `data/crypto/ALT_USDT/` (pairs) or `ALT_BTC/` (relative strength).
- **Monitoring**: Check `state.json` and `nav_history.csv` in `data/crypto/` for live status.
