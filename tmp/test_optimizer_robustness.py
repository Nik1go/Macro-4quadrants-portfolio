import pandas as pd
import numpy as np
import scipy.optimize as sco

# Mock data to simulate yfinance output with fewer tickers than requested
def mock_fetch_data(requested_tickers):
    # Simulate only 3 out of 5 tickers being available
    available_tickers = requested_tickers[:3]
    dates = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame(
        np.random.randn(100, 3) + 100, 
        index=dates, 
        columns=available_tickers
    )
    return data

def portfolio_performance(weights, returns, cov_matrix):
    mean_returns = returns.mean()
    p_ret = np.sum(mean_returns * weights) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return p_ret, p_vol

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    sharpe = (p_ret - risk_free_rate) / p_vol
    return -sharpe

def optimize_portfolio(data): 
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    # USE THE FIX: num_assets from data.columns, not static list
    num_assets = len(data.columns) 

    initial_weights = np.array(num_assets * [1. / num_assets])
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    result = sco.minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result, returns, cov_matrix

# Test Execution
requested = ['SPY', 'QQQ', 'GC=F', 'BTC-USD', '^FCHI']
data = mock_fetch_data(requested)
active_tickers = data.columns.tolist()

print(f"Requested: {len(requested)}")
print(f"Active: {len(active_tickers)} ({active_tickers})")

try:
    opt_result, returns, cov_matrix = optimize_portfolio(data)
    optimal_weights = opt_result.x
    print("Optimization successful!")
    print(f"Weights: {optimal_weights}")
    print(f"Weights length: {len(optimal_weights)}")
    
    if len(optimal_weights) == len(active_tickers):
        print("SUCCESS: Weights match active tickers count.")
    else:
        print("FAILURE: Weights mismatch!")

except Exception as e:
    print(f"CRASH: {e}")
