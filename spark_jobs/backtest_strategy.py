import os
import sys
import shutil
import pandas as pd
import numpy as np



"""
backtest_strategy.py - Daily Native Version (Vectorized)
Core Strategy:
1. Macro Smoothing: 20-day Rolling Mode for stable quadrant allocation.
2. Trend Following: MA150 overlay for SP500/GOLD with 5-day streak confirmation.
3. Transaction Costs: 0.10% on any allocation change.
4. Annualized Stats: 252 trading days.

dans le venv active  
python spark_jobs/backtest_strategy.py data/US/output_dag/quadrants.csv data/US/output_dag/Assets_daily.parquet data/US/output_dag/Forex_daily.parquet data/US/output_dag/combined_indicators.csv 1000 data/US/backtest_results
on peu ajt une start-date a la fin de la cmd YYYY-MM-JJ

"""

TRANSACTION_COST = 0.0010  # 0.10%
TRADING_DAYS = 252
MA_WINDOW = 200 # MA200 for trend following
# ML model already applies EMA smoothing (span=5) in compute_quadrants.py
# No additional rolling mode smoothing needed
N_STREAK = 5  # Days below/above MA to trigger action

# UCITS ETFs TER (Total Expense Ratio) - Real costs for European investors
TER = {
    'SP500': 0.0007,         # SXR8: 0.07% (iShares Core S&P 500 UCITS)
    'GOLD_OZ_USD': 0.0012,   # SGLD: 0.12% (iShares Physical Gold ETC)
    'SmallCAP': 0.0035,      # IUSN: 0.35% (iShares MSCI World Small Cap UCITS)
    'US_REIT_VNQ': 0.0040,   # IUSP: 0.40% (iShares US Property Yield UCITS)
    'TREASURY_10Y': 0.0010,  # IDTL: 0.10% (iShares $ Treasury Bond 7-10yr UCITS)
    'OBLIGATION': 0.0020,    # LQDE: 0.20% (iShares $ Corp Bond UCITS)
    'NASDAQ_100': 0.0033,    # SXRV: 0.33% (iShares Nasdaq 100 UCITS)
    'COMMODITIES': 0.0046,   # EXXY: 0.46% (iShares Diversified Commodity Swap UCITS)
    'SHORT_SP500': 0.0089    # SH: 0.89% - ProShares Short S&P500 (Inverse ETF)
}


# The assets available in the daily data
ASSETS_CORE = ['SP500', 'GOLD_OZ_USD', 'SmallCAP', 'US_REIT_VNQ', 'OBLIGATION', 'TREASURY_10Y', 'NASDAQ_100', 'COMMODITIES', 'SHORT_SP500']
FOREX_CORE = ['USD_EUR', 'USD_JPY', 'USD_CAD', 'USD_AUD', 'USD_BRL']
ASSETS = ASSETS_CORE + FOREX_CORE

# Import optimization functions
from optimization_engine import get_carry_adjusted_returns_wide, optimize_for_metric

# We will fill WEIGHTS dynamically during startup
WEIGHTS = {}


# rolling_mode removed: ML model handles smoothing via EMA in compute_quadrants.py


def max_drawdown(wealth_series):
    """Calculate maximum drawdown."""
    running_max = wealth_series.cummax()
    drawdown = (running_max - wealth_series) / running_max
    return drawdown.max()


def calculate_stats(returns, wealth, label):
    """Calculate annualized stats."""
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    sharpe_annual = (mean_ret / std_ret) * np.sqrt(TRADING_DAYS) if std_ret > 0 else np.nan
    vol_annual = std_ret * np.sqrt(TRADING_DAYS)
    md = max_drawdown(wealth)
    avg_year_ret = mean_ret * TRADING_DAYS
    return {
        f"{label}_vol_annual": vol_annual,
        f"{label}_sharpe_annual": sharpe_annual,
        f"{label}_max_drawdown": md,
        f"{label}_avg_year_return": avg_year_ret
    }


def main():
    if len(sys.argv) < 7:
        print("Usage: backtest_strategy.py <quadrants.csv> <Assets_daily.parquet> <Forex_daily.parquet> <Indicators.csv> <initial_capital> <output_dir> [start_date]")
        sys.exit(1)

    quadrants_csv, assets_parquet, forex_parquet, indicators_csv, initial_capital, output_dir = sys.argv[1:7]
    initial_capital = float(initial_capital)
    
    # Start date (default: 2009-01-01 = first OOS year from walk-forward with min_train_years=4)
    if len(sys.argv) >= 8:
        start_date = pd.to_datetime(sys.argv[7])
    else:
        start_date = pd.to_datetime('2009-01-01')
    print(f" Backtest starting from: {start_date.strftime('%Y-%m-%d')}")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ========== 1. LOAD & DEDUPLICATE ==========
    df_q = pd.read_csv(quadrants_csv, parse_dates=['date'])
    df_q = df_q.drop_duplicates(subset=['date']).sort_values('date')
    
    # FIX: Shift quadrant by 1 day to avoid look-ahead bias
    # Signal at Close(T) applies to Return(T+1)
    # We shift the columns (excluding date) or just use shift on the specific column later
    # Here we shift the entire dataframe's data columns relative to date
    cols_to_shift = [c for c in df_q.columns if c != 'date']
    df_q[cols_to_shift] = df_q[cols_to_shift].shift(1)
    
    df_q = df_q.set_index('date').sort_index()
    df_q = df_q.dropna(thresh=1)  # Drop rows that became NaN due to shift

    # ── Load OOS (Walk-Forward) quadrants ────────────────────────────────────
    # quadrants_oos.csv is exported by train_model.py from the walk-forward OOS
    # predictions. At each date T, the quadrant was assigned by a model trained
    # only on data strictly before T. This gives an unbiased view of live perf.
    oos_csv_path = os.path.join(os.path.dirname(quadrants_csv), 'quadrants_oos.csv')
    df_q_oos = None
    if os.path.exists(oos_csv_path):
        df_q_oos = pd.read_csv(oos_csv_path, parse_dates=['date'])
        df_q_oos = df_q_oos.drop_duplicates(subset=['date']).sort_values('date')

        # Shift T+1 (same anti-look-ahead as main signals)
        cols_to_shift_oos = [c for c in df_q_oos.columns if c != 'date']
        df_q_oos[cols_to_shift_oos] = df_q_oos[cols_to_shift_oos].shift(1)
        df_q_oos = df_q_oos.set_index('date').sort_index()

        # OOS is weekly (W-FRI from walk-forward resample) → ffill to business days
        full_daily_idx = pd.date_range(df_q_oos.index.min(), df_q_oos.index.max(), freq='B')
        df_q_oos = df_q_oos.reindex(full_daily_idx).ffill()
        df_q_oos.index.name = 'date'
        df_q_oos = df_q_oos.dropna(thresh=1)

        print(f"   ✅ OOS quadrants loaded: {len(df_q_oos)} business days")
        print(f"      Range: {df_q_oos.index.min().date()} → {df_q_oos.index.max().date()}")
    else:
        print(f"   ⚠️ quadrants_oos.csv not found at {oos_csv_path} — OOS curve skipped")

    df_a = pd.read_parquet(assets_parquet)
    df_a['date'] = pd.to_datetime(df_a['date'])
    df_a = df_a.drop_duplicates(subset=['date']).set_index('date').sort_index()

    df_f = pd.read_parquet(forex_parquet)
    df_f['date'] = pd.to_datetime(df_f['date'])
    df_f = df_f.drop_duplicates(subset=['date']).set_index('date').sort_index()
    
    df_ind = pd.read_csv(indicators_csv, parse_dates=['date']).set_index('date').sort_index()

    # Pre-calculate optimized weight allocation for the Backtest using Monte-Carlo ZScore Custom Custom Function
    df_returns_all = get_carry_adjusted_returns_wide(df_a.reset_index(), df_f.reset_index(), df_ind.reset_index())
    df_returns_all = df_returns_all.dropna(how='all')
    
    # Combine assets + forex
    df_a_combined = pd.merge(df_a, df_f, left_index=True, right_index=True, how='outer').ffill()

    # Option 1: Poids Bloqués par Régime (Locked Weights) au lieu d'optimisation Monte Carlo dynamique continuelle
    # Ceci empêche l'overfitting (suroptimisation) et réduit les frictions inutiles pour l'exécuteur.
    print("Application des Poids fixes (Locked Weights Option 1) en cours...")
    
    LOCKED_WEIGHTS = {
        1: {"SP500": 0.40, "NASDAQ_100": 0.40, "US_REIT_VNQ": 0.20},
        2: {"GOLD_OZ_USD": 0.40, "NASDAQ_100": 0.38, "COMMODITIES": 0.11, "SP500": 0.11},
        3: {"USD_JPY": 0.40, "SHORT_SP500": 0.32, "USD_EUR": 0.18, "COMMODITIES": 0.10},
        4: {"TREASURY_10Y": 0.40, "GOLD_OZ_USD": 0.38, "OBLIGATION": 0.22}
    }
    
    for q in [1, 2, 3, 4]:
        WEIGHTS[q] = {}
        # Ensure all ASSETS are present (fill missing with 0.0)
        for a in ASSETS:
            WEIGHTS[q][a] = LOCKED_WEIGHTS[q].get(a, 0.0)
        print(f"Poids fixes appliqués pour le Quadrant {q}: {WEIGHTS[q]}")

    # ========== 2. INNER JOIN ==========
    df = df_a_combined[ASSETS].join(df_q[['assigned_quadrant', 'PROB_GROWTH_RAW', 'PROB_INFLATION_RAW']], how='inner')
    df = df.dropna(subset=['assigned_quadrant', 'PROB_GROWTH_RAW', 'PROB_INFLATION_RAW'])
    df['assigned_quadrant'] = df['assigned_quadrant'].astype(int)
    
    # Apply start date filter
    df = df[df.index >= start_date]
    print(f"   Filtered to {len(df)} trading days from {start_date.strftime('%Y-%m-%d')}")

    # ========== 3. DAILY RETURNS ==========
    for asset in ASSETS:
        # Use our pre-calculated carry adjusted returns where applicable!
        if asset in df_returns_all.columns:
            df[f'{asset}_ret'] = df_returns_all[asset]
        else:
            df[f'{asset}_ret'] = df[asset].pct_change().fillna(0.0)


    # ========== 4. USE ML QUADRANTS DIRECTLY ==========
    # ML model (compute_quadrants.py) already applies EMA(5) smoothing on probabilities
    df['smooth_quadrant'] = df['assigned_quadrant']  # column name kept for backward compat

    # ========== 5. BASE ALLOCATION FROM SMOOTH QUADRANT ==========
    for asset in ASSETS:
        df[f'{asset}_base_weight'] = df['smooth_quadrant'].map(lambda q: WEIGHTS.get(q, {}).get(asset, 0.0))

    # ========== 6. TREND FOLLOWING OVERLAY (MA150 + 5-Day Streak) ==========
    # Applied to SP500, GOLD_OZ_USD, NASDAQ_100, and USD_JPY for downside protection
    for asset in ['SP500', 'GOLD_OZ_USD', 'NASDAQ_100', 'USD_JPY']:
        df[f'{asset}_MA'] = df[asset].rolling(MA_WINDOW, min_periods=1).mean()

        # Below MA streak
        below = (df[asset] < df[f'{asset}_MA']).astype(int)
        df[f'{asset}_below_streak'] = below.rolling(N_STREAK, min_periods=N_STREAK).sum() >= N_STREAK

        # Above MA streak
        above = (df[asset] > df[f'{asset}_MA']).astype(int)
        df[f'{asset}_above_streak'] = above.rolling(N_STREAK, min_periods=N_STREAK).sum() >= N_STREAK

    # ========== 7. RISK-OFF STATE MACHINE (Vectorized) ==========
    # For SP500, GOLD, and NASDAQ: track risk_off state using expanding logic
    for asset in ['SP500', 'GOLD_OZ_USD', 'NASDAQ_100', 'USD_JPY']:
        risk_off = pd.Series(False, index=df.index)

        # We need to iterate here due to the state machine nature
        # But we'll do it efficiently with numpy
        below_streak = df[f'{asset}_below_streak'].values
        above_streak = df[f'{asset}_above_streak'].values
        base_weight = df[f'{asset}_base_weight'].values
        quadrant_changed = df['smooth_quadrant'].diff().fillna(0).values != 0

        risk_off_arr = np.zeros(len(df), dtype=bool)

        for i in range(1, len(df)):
            # Quadrant changed -> reset risk_off, follow new quadrant rules
            if quadrant_changed[i]:
                risk_off_arr[i] = False
            # Currently risk_off, check if can go back risk_on
            elif risk_off_arr[i - 1]:
                if above_streak[i] and base_weight[i] > 0:
                    risk_off_arr[i] = False  # Back to risk_on
                else:
                    risk_off_arr[i] = True  # Stay risk_off
            # Currently risk_on, check if need to go risk_off
            else:
                if below_streak[i] and base_weight[i] > 0:
                    risk_off_arr[i] = True
                else:
                    risk_off_arr[i] = False

        df[f'{asset}_risk_off'] = risk_off_arr

    # ========== 8. FINAL WEIGHTS (After Trend Overlay) ==========
    for asset in ASSETS:
        df[f'{asset}_weight'] = df[f'{asset}_base_weight'].copy()

    # Apply risk-off: move weight to Treasury
    treasury_boost = pd.Series(0.0, index=df.index)
    for asset in ['SP500', 'GOLD_OZ_USD', 'NASDAQ_100', 'USD_JPY']:
        risk_off_mask = df[f'{asset}_risk_off']
        weight_to_move = df.loc[risk_off_mask, f'{asset}_weight'].copy()
        df.loc[risk_off_mask, f'{asset}_weight'] = 0.0
        treasury_boost.loc[risk_off_mask] += weight_to_move

    df['TREASURY_10Y_weight'] = df['TREASURY_10Y_weight'] + treasury_boost

    # ========== 9. TRANSACTION COSTS ==========
    weight_cols = [f'{a}_weight' for a in ASSETS]
    turnover = df[weight_cols].diff().abs().sum(axis=1).fillna(0.0)
    df['transaction_cost'] = turnover * TRANSACTION_COST

    # ========== 10. TER COSTS (Daily) ==========
    ter_daily = pd.Series(0.0, index=df.index)
    for asset in ASSETS:
        ter_daily += df[f'{asset}_weight'] * (TER.get(asset, 0.0) / TRADING_DAYS)
    df['ter_cost'] = ter_daily

    # ========== 11. PORTFOLIO RETURN ==========
    df['portfolio_return'] = 0.0
    for asset in ASSETS:
        df['portfolio_return'] += df[f'{asset}_weight'] * df[f'{asset}_ret']

    # Subtract costs
    df['portfolio_return'] = df['portfolio_return'] - df['ter_cost'] - df['transaction_cost']

    # ========== 12. WEALTH SERIES ==========
    df['wealth'] = initial_capital * (1 + df['portfolio_return']).cumprod()
    df['SP500_wealth'] = initial_capital * (1 + df['SP500_ret']).cumprod()
    df['GOLD_wealth'] = initial_capital * (1 + df['GOLD_OZ_USD_ret']).cumprod()

    # ========== 12.A OOS STRATEGY (Walk-Forward — honest unbiased curve) ==========
    # Uses quadrants_oos.csv: for each date T, the model only knew data < T.
    # No future information leaks in. This is the "real" live performance estimate.
    df_oos_wealth = None
    df_oos = None          # will hold OOS DataFrame if successfully computed
    oos_computed = False   # flag: True when OOS wealth is ready
    if df_q_oos is not None:
        oos_required_cols = ['assigned_quadrant', 'PROB_GROWTH_RAW', 'PROB_INFLATION_RAW']
        available_oos_cols = [c for c in oos_required_cols if c in df_q_oos.columns]

        df_oos = df_a_combined[ASSETS].join(
            df_q_oos[available_oos_cols], how='inner'
        ).dropna(subset=['assigned_quadrant'])
        df_oos['assigned_quadrant'] = df_oos['assigned_quadrant'].astype(int)
        df_oos = df_oos[df_oos.index >= start_date]

        if len(df_oos) > 10:
            # Returns (carry-adjusted where available)
            for asset in ASSETS:
                if asset in df_returns_all.columns:
                    df_oos[f'{asset}_ret'] = df_returns_all[asset].reindex(df_oos.index)
                else:
                    df_oos[f'{asset}_ret'] = df_oos[asset].pct_change().fillna(0.0)

            # Weights from same LOCKED_WEIGHTS (no leakage: weights are fixed rules)
            df_oos['smooth_quadrant'] = df_oos['assigned_quadrant']
            for asset in ASSETS:
                df_oos[f'{asset}_weight'] = df_oos['smooth_quadrant'].map(
                    lambda q: WEIGHTS.get(q, {}).get(asset, 0.0)
                )

            # Transaction costs
            oos_w_cols = [f'{a}_weight' for a in ASSETS]
            df_oos['transaction_cost'] = df_oos[oos_w_cols].diff().abs().sum(axis=1).fillna(0.0) * TRANSACTION_COST

            # TER costs
            oos_ter = pd.Series(0.0, index=df_oos.index)
            for asset in ASSETS:
                oos_ter += df_oos[f'{asset}_weight'] * (TER.get(asset, 0.0) / TRADING_DAYS)
            df_oos['ter_cost'] = oos_ter

            # Portfolio return & wealth
            df_oos['oos_return'] = sum(
                df_oos[f'{a}_weight'] * df_oos[f'{a}_ret'].fillna(0.0) for a in ASSETS
            ) - df_oos['ter_cost'] - df_oos['transaction_cost']
            df_oos['oos_wealth'] = initial_capital * (1 + df_oos['oos_return']).cumprod()

            df_oos_wealth = df_oos[['oos_wealth', 'oos_return', 'smooth_quadrant']].copy()
            df_oos_wealth.index.name = 'date'
            df_oos_wealth.to_csv(f"{output_dir}/backtest_oos_timeseries.csv")
            oos_computed = True   # stats will be added after stats={} in step 13
        else:
            print("   ⚠️ Not enough OOS rows after date filter — OOS wealth skipped")

    # ========== 12.B HIGH CONVICTION STRATEGY ==========
    # Initialize weights
    for asset in ASSETS:
        df[f'{asset}_hc_weight'] = 0.0

    # Q1: Growth > 65% AND Inflation < 35%
    # SP500, US REIT, Obligation
    q1_hc_mask = (df['PROB_GROWTH_RAW'] > 0.65) & (df['PROB_INFLATION_RAW'] < 0.35)
    df.loc[q1_hc_mask, 'SP500_hc_weight'] = 0.334
    df.loc[q1_hc_mask, 'US_REIT_VNQ_hc_weight'] = 0.333
    df.loc[q1_hc_mask, 'OBLIGATION_hc_weight'] = 0.333

    # Q2: Growth > 65% AND Inflation > 65%
    # NASDAQ_100, SP500
    q2_hc_mask = (df['PROB_GROWTH_RAW'] > 0.65) & (df['PROB_INFLATION_RAW'] > 0.65)
    df.loc[q2_hc_mask, 'NASDAQ_100_hc_weight'] = 0.5
    df.loc[q2_hc_mask, 'SP500_hc_weight'] = 0.5

    # Q4: Growth < 35% AND Inflation < 35%
    # Treasury 10Y, GOLD
    q4_hc_mask = (df['PROB_GROWTH_RAW'] < 0.35) & (df['PROB_INFLATION_RAW'] < 0.35)
    df.loc[q4_hc_mask, 'TREASURY_10Y_hc_weight'] = 0.5
    df.loc[q4_hc_mask, 'GOLD_OZ_USD_hc_weight'] = 0.5

    # Any other condition implies Cash (all weights remain 0.0)

    # Returns for HC
    df['hc_return'] = 0.0
    for asset in ASSETS:
        df['hc_return'] += df[f'{asset}_hc_weight'] * df[f'{asset}_ret']

    # Costs for HC
    hc_weight_cols = [f'{a}_hc_weight' for a in ASSETS]
    hc_turnover = df[hc_weight_cols].diff().abs().sum(axis=1).fillna(0.0)
    df['hc_transaction_cost'] = hc_turnover * TRANSACTION_COST

    hc_ter_daily = pd.Series(0.0, index=df.index)
    for asset in ASSETS:
        hc_ter_daily += df[f'{asset}_hc_weight'] * (TER.get(asset, 0.0) / TRADING_DAYS)
    df['hc_ter_cost'] = hc_ter_daily

    df['hc_return'] = df['hc_return'] - df['hc_ter_cost'] - df['hc_transaction_cost']
    df['hc_wealth'] = initial_capital * (1 + df['hc_return']).cumprod()

    # ========== 12.C HC 2X LEVERAGED ==========
    # Gross leverage x2 on every HC trade.
    # Borrowing cost: 5% annual on the leveraged notional, charged on active days only.
    HC_LEVERAGE       = 2.0
    HC_BORROW_ANNUAL  = 0.05   # 5% p.a. financing cost (SOFR-like)
    HC_BORROW_DAILY   = HC_BORROW_ANNUAL / TRADING_DAYS

    hc_active = (df[[f'{a}_hc_weight' for a in ASSETS]].sum(axis=1) > 0.01).astype(float)

    # Gross leveraged return = leverage * base_return − financing cost (when active)
    df['hc2x_return'] = (
        df['hc_return'] * HC_LEVERAGE
        - hc_active * HC_BORROW_DAILY * (HC_LEVERAGE - 1)   # only on borrowed fraction
    )
    df['hc2x_wealth'] = initial_capital * (1 + df['hc2x_return']).cumprod()
    print(f"HC 2x Leveraged: Final Wealth={df['hc2x_wealth'].iloc[-1]:.2f} | "
          f"Total Return={(df['hc2x_wealth'].iloc[-1]/initial_capital - 1)*100:.1f}%")

    stats = {}
    stats.update(calculate_stats(df['portfolio_return'], df['wealth'], 'strategy'))
    stats.update(calculate_stats(df['hc_return'], df['hc_wealth'], 'strategy_hc'))
    stats.update(calculate_stats(df['SP500_ret'], df['SP500_wealth'], 'SP500'))
    stats.update(calculate_stats(df['GOLD_OZ_USD_ret'], df['GOLD_wealth'], 'GOLD'))

    stats.update(calculate_stats(df['hc2x_return'], df['hc2x_wealth'], 'strategy_hc2x'))
    stats['cum_transaction_cost'] = df['transaction_cost'].sum()
    stats['cum_ter_cost'] = df['ter_cost'].sum()
    stats['initial_capital'] = initial_capital
    stats['final_wealth'] = df['wealth'].iloc[-1]
    stats['hc_final_wealth'] = df['hc_wealth'].iloc[-1]
    stats['hc2x_final_wealth'] = df['hc2x_wealth'].iloc[-1]
    stats['total_return'] = (df['wealth'].iloc[-1] / initial_capital) - 1
    stats['hc_total_return'] = (df['hc_wealth'].iloc[-1] / initial_capital) - 1
    stats['hc2x_total_return'] = (df['hc2x_wealth'].iloc[-1] / initial_capital) - 1

    # Count risk-off switches (including NASDAQ-100)
    for asset in ['SP500', 'GOLD_OZ_USD', 'NASDAQ_100', 'USD_JPY']:
        stats[f'nb_switch_{asset.lower()}'] = df[f'{asset}_risk_off'].diff().fillna(0).abs().sum() / 2

    # OOS stats (walk-forward honest curve) — added here because stats={} is in step 13
    if oos_computed and df_oos is not None:
        stats.update(calculate_stats(df_oos['oos_return'], df_oos['oos_wealth'], 'strategy_oos'))
        stats['oos_final_wealth'] = float(df_oos['oos_wealth'].iloc[-1])
        stats['oos_total_return'] = float((df_oos['oos_wealth'].iloc[-1] / initial_capital) - 1)
        print(f"OOS Backtest: Final Wealth={stats['oos_final_wealth']:.2f} | "
              f"Total Return={stats['oos_total_return']*100:.1f}% | "
              f"Sharpe={stats.get('strategy_oos_sharpe_annual', 0):.2f}")

    # ========== 14. EXPORT ==========
    pd.DataFrame([stats]).to_csv(f"{output_dir}/backtest_stats.csv", index=False)

    # Export Performance by Smooth Quadrant (Model Selection)
    # Calculate annualized return per asset per smooth quadrant
    perf_rows = []
    for q in [1, 2, 3, 4]:
        mask = df['smooth_quadrant'] == q
        if mask.any():
            # Annualized return for this quadrant
            days_in_q = mask.sum()
            years_in_q = days_in_q / 252.0
            
            for asset in ASSETS:
                # Total return in this quadrant
                returns_q = df.loc[mask, f'{asset}_ret']
                total_ret = (1 + returns_q).prod() - 1
                # Annualized
                ann_ret = (1 + total_ret) ** (1 / years_in_q) - 1 if years_in_q > 0 else 0
                
                # Sharpe
                mean_ret = returns_q.mean()
                std_ret = returns_q.std(ddof=1)
                sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

                perf_rows.append({
                    'quadrant': q,
                    'asset': asset,
                    'annual_return': ann_ret,
                    'sharpe': sharpe,
                    'days': days_in_q
                })
    
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(f"{output_dir}/assets_performance_by_smooth_quadrant.csv", index=False)

    # Timeseries
    base_weight_cols = [f'{a}_base_weight' for a in ASSETS]
    out_cols = [
        'smooth_quadrant', 'portfolio_return', 'wealth',
        'hc_return', 'hc_wealth',
        'hc2x_return', 'hc2x_wealth',
        'SP500_wealth', 'GOLD_wealth',
        'transaction_cost', 'ter_cost'
    ] + weight_cols + base_weight_cols
    df_out = df[out_cols].copy()
    df_out.index.name = 'date'
    df_out.to_csv(f"{output_dir}/backtest_timeseries.csv")

    # Costs breakdown
    df_costs = df[['transaction_cost', 'ter_cost']].copy()
    df_costs['cum_transaction_cost'] = df_costs['transaction_cost'].cumsum()
    df_costs['cum_ter_cost'] = df_costs['ter_cost'].cumsum()
    df_costs.to_csv(f"{output_dir}/backtest_costs.csv")

    print(f"Backtest terminé. Final Wealth: {stats['final_wealth']:.2f}, Sharpe: {stats['strategy_sharpe_annual']:.2f}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()