import os
import sys
import pandas as pd
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
momentum_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(os.path.dirname(momentum_dir))

if momentum_dir not in sys.path:
    sys.path.insert(0, momentum_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the backtest logic from streamlit app
import momentum_utils as mu
import data_fetcher as dfetch

def run_offline_backtest():
    """
    Executes the full VectorBT heatmap optimization offline (via Airflow).
    Saves the results into data/crypto/backtest/ for Streamlit to consume instantly.
    """
    print("Starting Offline Momentum Backtest Optimization...")
    
    # 1. Output directory
    crypto_data_dir = os.path.join(project_root, "data", "crypto")
    backtest_out_dir = os.path.join(crypto_data_dir, "backtest_results")
    os.makedirs(backtest_out_dir, exist_ok=True)
    
    # 2. Parameters
    start_date = "2020-01-01"
    
    # 3. Dynamic Universe (Top 20 Cryptos by 24h Volume + BTC)
    print("Fetching dynamic universe for backtest (Top 20 by Volume)...")
    top_symbols, _ = dfetch.fetch_top_symbols(n=80)  # Wide universe for rolling volume filter
    print(f"Universe selected: {top_symbols}")
    
    # 4. Execute Grid Search
    heatmap_df, best_params, pf, btc_close = mu.run_heatmap_simulation(
        symbols=top_symbols,
        start_date=start_date,
        sma_periods=[20, 30, 40, 50, 60, 70, 80, 90, 100],
        roll_lookbacks=[30, 60, 90, 120, 180, 240, 300, 350, 400, 450, 500, 550, 600],
        fees_bps=6,
        slippage_bps=10
    )
    
    if pf is None:
        print("Backtest failed: No data or no trades triggered.")
        return
        
    # 4. Save Heatmap Data
    heatmap_path = os.path.join(backtest_out_dir, "heatmap.csv")
    heatmap_df.to_csv(heatmap_path)
    print(f"Saved: {heatmap_path}")
    
    # 5. Save Equity Curve Data
    equity_series = pf.value()
    bh_series = (btc_close / btc_close.iloc[0]) * 10000.0
    
    equity_df = pd.DataFrame({
        "Momentum_Equity": equity_series,
        "Buy_and_Hold_BTC": bh_series
    })
    equity_path = os.path.join(backtest_out_dir, "equity_curves.csv")
    equity_df.to_csv(equity_path)
    print(f"Saved: {equity_path}")
    
    # 6. Save Stats & Best Params
    stats = pf.stats()
    results_dict = {
        "best_sma": int(best_params[0]),
        "best_lookback": int(best_params[1]),
        "start_date": start_date,
        "tot_ret_pct": float(stats.get("Total Return [%]", 0.0)),
        "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        "max_dd_pct": float(stats.get("Max Drawdown [%]", 0.0))
    }
    
    res_path = os.path.join(backtest_out_dir, "backtest_summary.json")
    with open(res_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved: {res_path}")
    
    # 7. Save Trades Log
    trades_df = pf.trades.records_readable
    if not trades_df.empty:
        # VectorBT uses Direction 0 for Long and 1 for Short
        if "Direction" in trades_df.columns:
            trades_df["Side"] = trades_df["Direction"].map({0: "Long", 1: "Short", "Long": "Long", "Short": "Short"}).fillna("Long/Short")
        else:
            trades_df["Side"] = "Long/Short"

        # Attach indicator values that triggered the trade (T-1)
        try:
            import indicators.calc_indicators as calc
            indicators = calc.compute_btc_indicators(btc_close, sma_period=int(best_params[0]), roll_lookback=int(best_params[1]))
            
            sig_dates = pd.to_datetime(trades_df["Entry Timestamp"]) - pd.Timedelta(days=1)
            # Map index values using series
            trades_df["BTC Skew"] = sig_dates.map(indicators["btc_skew"]).round(4)
            trades_df["BTC 5D Ret"] = sig_dates.map(indicators["btc_ret_5d"]).round(4)
            trades_df["BTC Median"] = sig_dates.map(indicators["btc_median"]).round(4)
            trades_df["BTC Std"] = sig_dates.map(indicators["btc_std"]).round(4)
            
            def format_cond(row):
                if pd.isna(row["BTC 5D Ret"]): return ""
                if row["Side"] == "Long":
                    return f"{row['BTC 5D Ret']} > {row['BTC Median']} + {row['BTC Std']}"
                elif row["Side"] == "Short":
                    return f"{row['BTC 5D Ret']} < {row['BTC Median']} - {row['BTC Std']}"
                return ""
                
            trades_df["Extrême Condition"] = trades_df.apply(format_cond, axis=1)
        except Exception as e:
            print(f"Failed to attach indicators: {e}")
            
        trades_path = os.path.join(backtest_out_dir, "trades_log.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Saved: {trades_path}")
        
    print("Offline Backtest Optimization completed successfully!")

if __name__ == "__main__":
    run_offline_backtest()
