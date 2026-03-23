"""
Airflow DAG: Crypto Momentum Trading Pipeline

Orchestrates the daily crypto momentum strategy:
1. fetch_data        — Download/update Top 20 + BTC klines from Binance (USDT + BTC pairs)
2. calc_indicators   — Compute BTC SMA, 5D returns, rolling stats, ALT/BTC SMAs
3. generate_signals  — Evaluate Long/Short entry & exit conditions
4. execute_orders    — Send orders to Binance Futures Testnet (dry-run by default)
5. update_monitoring — Update NAV history, execution logs, portfolio state

Schedule: Daily at 00:05 UTC (after daily candle close)
Trigger:  airflow dags trigger dag_crypto_momentum
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# === Dynamic Paths Configuration ===
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME')
PROJECT_ROOT = os.path.abspath(os.path.join(AIRFLOW_HOME, '..'))

# Add project paths for module imports
MOMENTUM_DIR = os.path.join(PROJECT_ROOT, 'streamlit_app', 'momentum_BTC')
if MOMENTUM_DIR not in sys.path:
    sys.path.insert(0, MOMENTUM_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === Strategy Parameters ===
SMA_PERIOD = 50
ROLL_LOOKBACK = 600
START_DATE = "2020-01-01"
DRY_RUN = True  # Set to False to execute on Binance Testnet

# === DAG Default Args ===
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


# === Task Functions ===

def task_fetch_data(**kwargs):
    """Task 1: Fetch crypto data from Binance."""
    from data_fetcher import fetch_all_crypto_data

    print("=" * 60)
    print("📡 TASK 1: Fetching crypto data from Binance...")
    print("=" * 60)

    results = fetch_all_crypto_data(start_date=START_DATE)

    n_usdt = len(results.get("ALT_USDT", {}))
    n_btc = len(results.get("ALT_BTC", {}))
    print(f"\n✅ Fetch complete: {n_usdt} USDT pairs, {n_btc} BTC pairs")

    return results


def task_calc_indicators(**kwargs):
    """Task 2: Calculate all technical indicators."""
    from indicators.calc_indicators import compute_all_indicators

    print("=" * 60)
    print("📊 TASK 2: Computing indicators...")
    print("=" * 60)

    indicators = compute_all_indicators(
        sma_period=SMA_PERIOD,
        roll_lookback=ROLL_LOOKBACK
    )

    # Log summary
    btc_close = indicators["btc_close"].iloc[-1]
    btc_sma = indicators["btc_sma"].iloc[-1]
    btc_ret = indicators["btc_ret_5d"].iloc[-1]
    btc_med = indicators["btc_median"].iloc[-1]
    btc_std = indicators["btc_std"].iloc[-1]
    above = indicators["btc_above_sma_2d"].iloc[-1]
    below = indicators["btc_below_sma_2d"].iloc[-1]

    long_threshold = btc_med + 0.5 * btc_std
    short_threshold = btc_med - 0.5 * btc_std

    print(f"\n📈 BTC Close: ${btc_close:,.0f}")
    print(f"📉 BTC SMA({SMA_PERIOD}): ${btc_sma:,.0f}")
    print(f"📊 BTC 5D Return: {btc_ret:.2%}")
    print(f"📏 Rolling Median (5D): {btc_med:.2%}")
    print(f"📐 Rolling StdDev (5D): {btc_std:.2%}")
    print(f"🟢 Long Threshold (Med + 0.5σ): {long_threshold:.2%}  {'✅ TRIGGERED' if btc_ret > long_threshold else '❌ Not met'}")
    print(f"🔴 Short Threshold (Med - 0.5σ): {short_threshold:.2%}  {'✅ TRIGGERED' if btc_ret < short_threshold else '❌ Not met'}")
    print(f"🟢 BTC > SMA (2d): {above}")
    print(f"🔴 BTC < SMA (2d): {below}")

    # ── Preview: which altcoins would be selected ──
    import pandas as pd
    today_idx = len(indicators["btc_ret_5d"]) - 1
    today = indicators["btc_ret_5d"].index[today_idx]
    alt_btc_cols = list(indicators["alt_btc_closes"].columns)

    # Long candidate preview
    long_conditions_met = (btc_ret > long_threshold) and above
    print(f"\n{'─' * 50}")
    if long_conditions_met:
        long_filtered = []
        for sym in alt_btc_cols:
            if sym in indicators["alt_btc_above_sma_2d"].columns:
                if indicators["alt_btc_above_sma_2d"].at[today, sym]:
                    usdt_sym = sym.replace("BTC", "USDT")
                    if usdt_sym in indicators["ret_3d"].columns:
                        ret_val = indicators["ret_3d"].at[today, usdt_sym]
                        if pd.notna(ret_val):
                            long_filtered.append((usdt_sym, ret_val))
        if long_filtered:
            long_filtered.sort(key=lambda x: x[1], reverse=True)
            print(f"🟢 LONG SIGNAL PREVIEW — {len(long_filtered)} alts pass filter:")
            for sym, ret in long_filtered:
                marker = "⭐" if sym == long_filtered[0][0] else "  "
                print(f"   {marker} {sym:<12s}  3D ret: {ret:+.2%}")
            print(f"   → Crypto ⭐ star du moment: {long_filtered[0][0]} ({long_filtered[0][1]:+.2%})")
        else:
            print("🟢 LONG: BTC conditions ✅ mais aucune alt ne passe le filtre ALT/BTC > SMA (2d)")
    else:
        print("🟢 LONG: Pas de signal (conditions BTC non remplies)")

    # Short candidate preview
    short_conditions_met = (btc_ret < short_threshold) and below
    if short_conditions_met:
        short_filtered = []
        for sym in alt_btc_cols:
            if sym in indicators["alt_btc_below_sma_2d"].columns:
                if indicators["alt_btc_below_sma_2d"].at[today, sym]:
                    usdt_sym = sym.replace("BTC", "USDT")
                    if usdt_sym in indicators["ret_3d"].columns:
                        ret_val = indicators["ret_3d"].at[today, usdt_sym]
                        if pd.notna(ret_val):
                            short_filtered.append((usdt_sym, ret_val))
        if short_filtered:
            short_filtered.sort(key=lambda x: x[1])
            print(f"🔴 SHORT SIGNAL PREVIEW — {len(short_filtered)} alts pass filter:")
            for sym, ret in short_filtered:
                marker = "💀" if sym == short_filtered[0][0] else "  "
                print(f"   {marker} {sym:<12s}  3D ret: {ret:+.2%}")
            print(f"   → Crypto 💀 absente du moment: {short_filtered[0][0]} ({short_filtered[0][1]:+.2%})")
        else:
            print("🔴 SHORT: BTC conditions ✅ mais aucune alt ne passe le filtre ALT/BTC < SMA (2d)")
    else:
        print("🔴 SHORT: Pas de signal (conditions BTC non remplies)")
    print(f"{'─' * 50}")

    # Store indicators in XCom (serializable summary only)
    kwargs['ti'].xcom_push(key='btc_close', value=float(btc_close))
    kwargs['ti'].xcom_push(key='btc_sma', value=float(btc_sma))

    return "indicators_computed"


def task_generate_signals(**kwargs):
    """Task 3: Generate Long/Short trading signals."""
    from indicators.calc_indicators import compute_all_indicators
    from signals.generate_signals import generate_daily_signals

    print("=" * 60)
    print("🎯 TASK 3: Generating signals...")
    print("=" * 60)

    indicators = compute_all_indicators(
        sma_period=SMA_PERIOD,
        roll_lookback=ROLL_LOOKBACK
    )
    report, state = generate_daily_signals(indicators)

    # Log summary
    print(f"\n📅 Date: {report['date']}")
    print(f"🚪 Exits: {len(report['exits'])}")
    for ex in report['exits']:
        print(f"   → {ex['side'].upper()} EXIT {ex['symbol']} ({ex['reason']})")

    print(f"🎯 Entries: {len(report['entries'])}")
    for entry in report['entries']:
        print(f"   → {entry['side'].upper()} ENTRY {entry['symbol']} @ ${entry['entry_price']:,.2f}")

    print(f"💰 Cash remaining: ${state['cash']:,.2f}")
    print(f"📦 Open positions: {len(state['positions'])}")

    # Push report path for next task
    kwargs['ti'].xcom_push(key='signal_report', value=report)

    return report


def task_execute_orders(**kwargs):
    """Task 4: Execute orders on Binance Testnet."""
    from execution.binance_executor import execute_signals

    print("=" * 60)
    print(f"⚡ TASK 4: Executing orders ({'DRY RUN' if DRY_RUN else 'LIVE TESTNET'})...")
    print("=" * 60)

    ti = kwargs['ti']
    signal_report = ti.xcom_pull(task_ids='generate_signals', key='signal_report')

    if not signal_report:
        print("⚠️ No signal report found — skipping execution")
        return None

    execution_log = execute_signals(signal_report, dry_run=DRY_RUN)

    n_orders = len(execution_log.get("orders", []))
    print(f"\n✅ Execution complete: {n_orders} orders processed")

    kwargs['ti'].xcom_push(key='execution_log', value=execution_log)
    return execution_log


def task_update_monitoring(**kwargs):
    """Task 5: Update monitoring data (NAV, logs)."""
    from indicators.calc_indicators import compute_all_indicators
    from monitoring.portfolio_tracker import update_monitoring

    print("=" * 60)
    print("📊 TASK 5: Updating monitoring...")
    print("=" * 60)

    ti = kwargs['ti']
    signal_report = ti.xcom_pull(task_ids='generate_signals', key='signal_report')
    execution_log = ti.xcom_pull(task_ids='execute_orders', key='execution_log')

    if not signal_report:
        print("⚠️ No signal report — skipping monitoring update")
        return

    # Reload indicators for current prices
    indicators = compute_all_indicators(
        sma_period=SMA_PERIOD,
        roll_lookback=ROLL_LOOKBACK
    )

    nav = update_monitoring(signal_report, execution_log or {}, indicators)
    print(f"\n✅ Monitoring updated — NAV: ${nav:,.2f}")

    return nav


# === DAG Definition ===
with DAG(
    dag_id='dag_crypto_momentum',
    default_args=default_args,
    description='Crypto Momentum Trading Pipeline (Long & Short) — Top 20 Altcoins',
    schedule='5 0 * * *',  # Daily at 00:05 UTC
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'momentum', 'trading'],
) as dag:

    fetch_data = PythonOperator(
        task_id='fetch_data',
        python_callable=task_fetch_data,
    )

    calc_indicators = PythonOperator(
        task_id='calc_indicators',
        python_callable=task_calc_indicators,
    )

    generate_signals = PythonOperator(
        task_id='generate_signals',
        python_callable=task_generate_signals,
    )

    execute_orders = PythonOperator(
        task_id='execute_orders',
        python_callable=task_execute_orders,
    )

    update_monitor = PythonOperator(
        task_id='update_monitoring',
        python_callable=task_update_monitoring,
    )

    # Weekly offline backtest generator (Runs on Sundays by default or when externally triggered via UI)
    run_offline_backtest = BashOperator(
        task_id='generate_offline_backtest',
        bash_command=f'python {os.path.join(PROJECT_ROOT, "streamlit_app", "momentum_BTC", "backtest", "run_backtest.py")}',
    )

    # Task dependencies: linear pipeline
    fetch_data >> calc_indicators >> generate_signals >> execute_orders >> update_monitor >> run_offline_backtest
