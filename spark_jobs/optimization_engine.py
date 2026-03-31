import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def get_carry_adjusted_returns_wide(df_assets, df_forex, df_indicators):
    """
    Returns a wide dataframe (index=date, cols=assets) of daily returns.
    For Forex, it includes the carry yield.
    """
    df_assets = df_assets.copy()
    df_assets['date'] = pd.to_datetime(df_assets['date'])
    df_assets = df_assets.set_index('date').sort_index()
    
    # Calculate price return for assets
    df_ret_assets = df_assets.pct_change().dropna(how='all')
    
    if df_forex is not None and not df_forex.empty:
        df_forex = df_forex.copy()
        df_forex['date'] = pd.to_datetime(df_forex['date'])
        df_forex = df_forex.set_index('date').sort_index()
        
        # Forex price return
        df_ret_forex = df_forex.pct_change().dropna(how='all')
        
        # Calculate Carry Yield
        if df_indicators is not None and not df_indicators.empty:
            df_ind = df_indicators.copy()
            df_ind['date'] = pd.to_datetime(df_ind['date'])
            df_ind = df_ind.set_index('date').sort_index().ffill()
            
            CARRY_MAPPING = {
                'USD_EUR': ('TAUX_FED', 'TAUX_ECB'),
                'USD_JPY': ('TAUX_FED', 'TAUX_BOJ'),
                'USD_CAD': ('TAUX_FED', 'TAUX_BOC'),
                'USD_AUD': ('TAUX_FED', 'TAUX_RBA'),
                'USD_BRL': ('TAUX_FED', 'TAUX_BCB'),
            }
            
            for pair, (l_r, s_r) in CARRY_MAPPING.items():
                if pair in df_ret_forex.columns and l_r in df_ind.columns and s_r in df_ind.columns:
                    # Align dates
                    idx = df_ret_forex.index.intersection(df_ind.index)
                    lr_series = df_ind.loc[idx, l_r].fillna(0)
                    sr_series = df_ind.loc[idx, s_r].fillna(0)
                    daily_carry = (lr_series - sr_series) / 100.0 / 252.0
                    df_ret_forex.loc[idx, pair] += daily_carry
                    
            # Inversion (e.g. BRL/USD instead of USD/BRL)
            # To keep things simple, we keep the direct pairs as they are to test the allocator.
            
        df_all = pd.merge(df_ret_assets, df_ret_forex, left_index=True, right_index=True, how='outer')
        return df_all.fillna(0)
    
    return df_ret_assets.fillna(0)

def portfolio_performance_vectorized(weights, returns, rf_rate):
    # weights: (n_sims, n_assets)
    # returns: (n_obs, n_assets)
    # port_returns: (n_obs, n_sims)
    port_returns = returns.values.dot(weights.T)
    
    # Yearly metrics
    mean_ret = port_returns.mean(axis=0) * 252
    vol = port_returns.std(axis=0) * np.sqrt(252)
    
    # Sharpe
    vol_safe = np.where(vol < 1e-6, 1e-6, vol)
    sharpe = (mean_ret - rf_rate) / vol_safe
    
    # Sortino
    # Downside deviation
    downside = np.where(port_returns < 0, port_returns, 0)
    downside_vol = downside.std(axis=0) * np.sqrt(252)
    downside_vol_safe = np.where(downside_vol < 1e-6, 1e-6, downside_vol)
    sortino = (mean_ret - rf_rate) / downside_vol_safe
    
    # Calmar (Max Drawdown)
    cumprod = (1 + port_returns).cumprod(axis=0)
    running_max = np.maximum.accumulate(cumprod, axis=0)
    drawdown = (cumprod - running_max) / running_max
    max_dd = -drawdown.min(axis=0)
    max_dd_safe = np.where(max_dd < 1e-6, 1e-6, max_dd)
    calmar = mean_ret / max_dd_safe
    
    return mean_ret, vol_safe, sharpe, sortino, calmar

def run_monte_carlo(returns, rf_rate, n_sims=25000):
    # Set seed for consistency between backend and UI
    np.random.seed(42)
    n_assets = returns.shape[1]
    max_weight = 0.405
    min_weight = 0.05
    
    # Generate large pool to handle constraints
    alpha_val = max(1.0, 5.0 - n_assets)
    weights = np.random.dirichlet(np.full(n_assets, alpha_val), size=n_sims * 20)
    
    # 1. Enforce Min Weight (Zero out dust)
    weights[weights < min_weight] = 0.0
    
    # 2. Re-normalize to 100%
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
    
    # 3. Filter valid portfolios (Sum must be 1.0 and no weight > max_weight)
    valid_mask = (np.abs(weights.sum(axis=1) - 1.0) < 1e-6) & (np.max(weights, axis=1) <= max_weight)
    weights = weights[valid_mask]
    
    if len(weights) > n_sims:
        weights = weights[:n_sims]
    elif len(weights) == 0:
        # Fallback to equal weight if no valid portfolios found
        weights = np.full((1, n_assets), 1.0 / n_assets)
        
    mean_ret, vol, sharpe, sortino, calmar = portfolio_performance_vectorized(weights, returns, rf_rate)
    
    # Stats for Custom Z-Score
    stats = {
        'sharpe_m': np.mean(sharpe), 'sharpe_s': np.std(sharpe) if np.std(sharpe)>0 else 1,
        'sortino_m': np.mean(sortino), 'sortino_s': np.std(sortino) if np.std(sortino)>0 else 1,
        'calmar_m': np.mean(calmar), 'calmar_s': np.std(calmar) if np.std(calmar)>0 else 1
    }
    
    # Z-scores
    zs = (sharpe - stats['sharpe_m']) / stats['sharpe_s']
    zso = (sortino - stats['sortino_m']) / stats['sortino_s']
    zc = (calmar - stats['calmar_m']) / stats['calmar_s']
    custom_z = (zs + zso + zc) / 3.0
    
    res = {
        'weights': weights,
        'mean_ret': mean_ret,
        'vol': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'custom': custom_z
    }
    return stats, res

def optimize_for_metric(returns, rf_rate, metric="custom", stats=None, max_weight=0.40, res=None):
    if res is None:
        stats, res = run_monte_carlo(returns, rf_rate, 25000)
        
    metric_array = res[metric]
    best_idx = np.argmax(metric_array)
    opt_weights = res['weights'][best_idx]
    
    opt_weights = np.round(opt_weights, 4)
    opt_weights = opt_weights / np.sum(opt_weights)
    return dict(zip(returns.columns, opt_weights))

def run_efficient_frontier_points(returns, rf_rate, n_sims=8000):
    # Reduced sims for UI rendering speed
    stats, res = run_monte_carlo(returns, rf_rate, n_sims)
    
    mc_data = {
        'returns': res['mean_ret'],
        'volatilities': res['vol'],
        'sharpes': res['sharpe'],
        'sortinos': res['sortino'],
        'calmars': res['calmar'],
        'customs': res['custom'],
        'weights': res['weights']
    }
    
    # Find min vol portfolio
    min_vol_idx = np.argmin(res['vol'])
    min_vol_w = res['weights'][min_vol_idx]
    
    # Optimizations simply pluck the argmax from the monte carlo! Perfectly robust.
    opt_sharpe = optimize_for_metric(returns, 0, "sharpe", res=res)
    opt_sortino = optimize_for_metric(returns, 0, "sortino", res=res)
    opt_calmar = optimize_for_metric(returns, 0, "calmar", res=res)
    opt_custom = optimize_for_metric(returns, 0, "custom", res=res)
    
    return {
        'mc_data': mc_data,
        'min_vol_weights': dict(zip(returns.columns, min_vol_w)),
        'opt_sharpe': opt_sharpe,
        'opt_sortino': opt_sortino,
        'opt_calmar': opt_calmar,
        'opt_custom': opt_custom,
        'stats': stats
    }
