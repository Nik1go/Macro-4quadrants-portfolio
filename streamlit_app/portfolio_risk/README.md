# Portfolio Risk Engine — Streamlit app

Interactive risk engine for a multi-asset portfolio valued in euros, built for the
*Portfolio Risk Management with Python* project (MSc Finance & Big Data, NEOMA).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. It starts from the saved Yahoo Finance dataset in `data/`
(2 Jan 2019 – 8 Oct 2026). New tickers outside that file are downloaded automatically with
`yfinance` for the fixed study period. Existing prices are read from the saved dataset.

The study uses a fixed **€10,000 portfolio**, a fixed date range of **1 Jan 2019 – 8 Oct 2026 inclusive**,
and a fixed **2% annual risk-free rate** for the Sharpe ratio. The first common price date for the
default portfolio is 2 Jan 2019. The editable portfolio displays its total weight immediately;
weights must add up to 100% before an analysis can be applied. Monte Carlo offers an explicit choice
between normal returns and Student-t returns with 5 degrees of freedom.

## What is in it

| Tab | Content |
|---|---|
| Overview | Holdings, capital vs risk allocation, prices in EUR |
| Data | Source, calendar alignment, currency conversion, CSV downloads |
| Performance | Return, volatility, Sharpe, skewness, kurtosis, drawdown, rolling volatility |
| Correlation | Correlation matrix by regime (calm 2019, Covid, 2022, 2025), rolling correlation, diversification ratio |
| VaR & ES | Historical, parametric and Monte Carlo VaR/ES at 1 day and N days, fat-tail variants, MC convergence, risk contributions |
| Backtesting | Rolling out-of-sample backtest, Kupiec, Christoffersen, Basel traffic light, exception list |
| Stress tests | Historical episodes (+ custom), editable per-asset shocks, beta-driven market shock, correlation and volatility stress, reverse stress test |
| Concentration | VaR/ES as the portfolio is moved into one asset; comparison with equal weights |
| Findings & export | Answers to the 8 project questions (computed live) and an Excel export of every table |

All inputs are validated: weights must be non-negative and sum to 100%, tickers must be unique,
at least one confidence level is required, and the estimation window must fit in the available history.

## Python notebook

At the bottom of the page, download **the notebook (.ipynb)** and **its data (.csv)** into the same folder.
The notebook download includes the last valid portfolio and risk settings; its stress tests use the
predefined project scenarios. Run its cells in order in Jupyter or VS Code. Short explanations in French
walk through the calculations using NumPy, pandas, SciPy and Matplotlib, without depending on Streamlit
or `risk_engine.py`. The final cell exports the main results to `notebook_results.xlsx`.

The repository's `portfolio_risk.ipynb` uses the default portfolio and reads the saved CSV from `data/`.

## Files

- `app.py` — Streamlit interface
- `risk_engine.py` — calculations (no Streamlit), reusable from a notebook
- `portfolio_risk.ipynb` — commented, step-by-step Python walkthrough
- `data/yahoo_adj_close.csv` + `data/yahoo_adj_close_meta.json` — Yahoo Finance adjusted closes and source metadata
- `.streamlit/config.toml` — theme
