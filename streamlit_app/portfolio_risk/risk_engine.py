"""
Portfolio Risk Engine — computation module.

Pure functions (no Streamlit) so that the same code can be used by the app, a notebook or tests.
Conventions: simple returns for P&L/VaR, log returns for annualised performance, VaR/ES reported
as positive euro losses (absolute VaR, drift included).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

BASE_CCY = "EUR"
TRADING_DAYS = 252

SUFFIX_CCY = {".PA": "EUR", ".DE": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR", ".BR": "EUR", ".F": "EUR",
              ".L": "GBP", ".SW": "CHF", ".T": "JPY", ".HK": "HKD", ".TO": "CAD", ".AX": "AUD"}

KNOWN_CLASS = {"AAPL": "US equity", "NVDA": "US equity", "MSFT": "US equity", "AMZN": "US equity",
               "MC.PA": "EU equity", "SPY": "US equity index", "QQQ": "US equity index", "TLT": "US Treasuries 20y+",
               "IEF": "US Treasuries 7-10y", "GLD": "Gold", "VNQ": "US real estate", "BTC-USD": "Crypto",
               "ETH-USD": "Crypto", "EFA": "Intl. equity index", "AGG": "US aggregate bonds"}

EPISODES = {
    "gfc2008": ("Global financial crisis", "2007-10-09", "2009-03-09"),
    "covid": ("Covid crash", "2020-02-19", "2020-03-23"),
    "rates2022": ("2022 rate shock", "2022-01-03", "2022-10-12"),
    "tariffs2025": ("2025 tariff shock", "2025-02-19", "2025-04-08"),
}

REGIMES = {
    "Calm year 2019": ("2019-01-01", "2019-12-31"),
    "Covid crash": ("2020-02-19", "2020-04-30"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock": ("2025-02-19", "2025-04-30"),
}


# --------------------------------------------------------------------------- data
def guess_ccy(ticker: str) -> str:
    if ticker.endswith("-USD"):
        return "USD"
    if ticker.endswith("-EUR"):
        return "EUR"
    for suf, c in SUFFIX_CCY.items():
        if ticker.endswith(suf):
            return c
    return "USD"


def is_24_7(ticker: str) -> bool:
    return ticker.endswith("-USD") or ticker.endswith("-EUR")


def _yf_download(tickers, start, end):
    import yfinance as yf
    end_excl = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    raw = yf.download(list(tickers), start=start, end=end_excl, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(list(tickers)[0])
    raw = raw.dropna(how="all")
    ccy = {}
    for t in tickers:
        try:
            ccy[t] = yf.Ticker(t).fast_info["currency"].upper()
        except Exception:
            ccy[t] = guess_ccy(t)
    return raw, ccy


def load_prices(tickers, start, end, data_dir, refresh=False):
    """Return raw adjusted closes (local currencies + needed FX), currency map and metadata.

    Uses the cached CSV when it covers the request; otherwise downloads with yfinance and updates the cache.
    """
    data_file = os.path.join(data_dir, "yahoo_adj_close.csv")
    meta_file = os.path.join(data_dir, "yahoo_adj_close_meta.json")
    meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    cached = pd.read_csv(data_file, index_col=0, parse_dates=True) if os.path.exists(data_file) else None
    ccy = dict(meta.get("currencies", {}))
    for t in tickers:
        ccy.setdefault(t, guess_ccy(t))
    fx_needed = sorted({f"EUR{ccy[t]}=X" for t in tickers if ccy[t] != BASE_CCY})
    wanted = list(tickers) + fx_needed
    missing = [t for t in wanted if cached is None or t not in cached.columns]
    covers = cached is not None and cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7) \
        and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)
    downloaded = False
    if refresh or missing or not covers:
        px, ccy_new = _yf_download(tickers, start, end)
        ccy.update(ccy_new)
        fx_needed = sorted({f"EUR{ccy[t]}=X" for t in tickers if ccy[t] != BASE_CCY})
        if fx_needed:
            fx, _ = _yf_download(fx_needed, start, end)
            for f in fx_needed:
                ccy[f] = f[3:6]
            px = px.join(fx, how="outer")
        px.index.name = "Date"
        if cached is not None and not refresh:     # merge with the existing cache
            keep = [c for c in cached.columns if c not in px.columns]
            px = px.join(cached[keep], how="outer")
        os.makedirs(data_dir, exist_ok=True)
        px.to_csv(data_file)
        meta = {"source": "Yahoo Finance via yfinance (auto_adjust=True: adjusted close)",
                "downloaded": str(date.today()), "start": str(px.index.min().date()),
                "end": str(px.index.max().date()), "currencies": ccy}
        json.dump(meta, open(meta_file, "w"), indent=2)
        cached = px
        downloaded = True
    cols = list(tickers) + fx_needed
    missing_cols = [c for c in cols if c not in cached.columns or cached[c].dropna().empty]
    if missing_cols:
        raise ValueError(f"No price data returned for {', '.join(missing_cols)}. Check the Yahoo Finance symbols.")
    meta = dict(meta, downloaded_now=downloaded)
    return cached.loc[start:end, cols], ccy, meta


def build_eur_prices(raw, tickers, ccy, ffill_limit=3):
    """Common calendar (intersection of exchange trading days), forward-fill small gaps, convert to EUR."""
    exchange = [t for t in tickers if not is_24_7(t)]
    cal = raw.index[raw.index.dayofweek < 5]
    for t in (exchange or tickers):
        cal = cal.intersection(raw[t].dropna().index)
    first_common = max(raw[t].first_valid_index() for t in tickers)
    cal = cal[cal >= first_common]
    filled = raw.ffill(limit=ffill_limit)
    px = filled.loc[cal].copy()
    nan_after = int(px[list(tickers)].isna().sum().sum())
    px = px.dropna()
    eur = pd.DataFrame(index=px.index)
    for t in tickers:
        eur[t] = px[t] if ccy[t] == BASE_CCY else px[t] / px[f"EUR{ccy[t]}=X"]
    own_days = {t: int(raw[t].loc[eur.index.min():].notna().sum()) for t in tickers}
    report = {
        "raw_rows": int(raw.shape[0]), "common_days": int(len(cal)), "nan_after_alignment": nan_after,
        "final_days": int(len(eur)), "first_common_date": str(first_common.date()),
        "excluded_days": {t: own_days[t] - int(len(eur)) for t in tickers},
    }
    return eur, report


# --------------------------------------------------------------------------- statistics
def max_drawdown(simple_returns: pd.Series):
    wealth = (1 + simple_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min()), dd.idxmin(), dd


def summary_stats(R: pd.DataFrame, LR: pd.DataFrame, rf: float) -> pd.DataFrame:
    ann_ret = LR.mean() * TRADING_DAYS
    ann_vol = LR.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({
        "Ann. return": ann_ret, "Ann. volatility": ann_vol, "Sharpe": (ann_ret - rf) / ann_vol,
        "Skewness": R.skew(), "Excess kurtosis": R.kurt(), "Worst day": R.min(), "Best day": R.max(),
        "Max drawdown": R.apply(lambda s: max_drawdown(s)[0]),
    })


def avg_offdiag(c: pd.DataFrame) -> float:
    m = c.values
    n = len(m)
    return float((m.sum() - n) / (n * (n - 1)))


def rolling_avg_corr(R: pd.DataFrame, window=63) -> pd.Series:
    out = {}
    vals = R.values
    for i in range(window, len(R) + 1):
        c = np.corrcoef(vals[i - window:i], rowvar=False)
        n = c.shape[0]
        out[R.index[i - 1]] = (np.nansum(c) - n) / (n * (n - 1))
    return pd.Series(out)


# --------------------------------------------------------------------------- VaR / ES
def var_es_from_pnl(pnl, alpha):
    pnl = np.asarray(pnl)
    q = np.quantile(pnl, 1 - alpha)
    return float(-q), float(-pnl[pnl <= q].mean())


def historical_var_es(R_win, w, V, alpha, h=1):
    rp = np.asarray(R_win) @ np.asarray(w)
    if h == 1:
        return var_es_from_pnl(V * rp, alpha)
    gross = pd.Series(1 + rp).rolling(h).apply(np.prod, raw=True).dropna().values
    return var_es_from_pnl(V * (gross - 1), alpha)


def parametric_var_es(mu, Sigma, w, V, alpha, h=1):
    w = np.asarray(w)
    m = float(w @ np.asarray(mu)) * h
    s = float(np.sqrt(w @ np.asarray(Sigma) @ w)) * np.sqrt(h)
    z = stats.norm.ppf(alpha)
    return (z * s - m) * V, (s * stats.norm.pdf(z) / (1 - alpha) - m) * V


def simulate_returns(mu, Sigma, n, h=1, seed=42, dist="normal", nu=5):
    """Simulated h-day compounded asset returns, shape (n, k). Student-t draws are scaled to unit variance."""
    rng = np.random.default_rng(seed)
    S = np.asarray(Sigma)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(S + np.eye(len(S)) * 1e-12)
    k = len(mu)
    gross = np.ones((n, k))
    for _ in range(h):
        z = rng.standard_normal((n, k))
        if dist == "t":
            chi = rng.chisquare(nu, size=(n, 1))
            z = z / np.sqrt(chi / nu) * np.sqrt((nu - 2) / nu)
        gross *= 1 + (np.asarray(mu) + z @ L.T)
    return gross - 1


def mc_var_es(mu, Sigma, w, V, alpha, h=1, n=100_000, seed=42, dist="normal"):
    sims = simulate_returns(mu, Sigma, n, h, seed, dist)
    return var_es_from_pnl(V * sims @ np.asarray(w), alpha)


def cornish_fisher_z(alpha, S, K):
    z = stats.norm.ppf(alpha)
    return z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36


def ewma_sigma(rp, lam=0.94, seed_obs=30):
    v = float(np.var(rp[:seed_obs]))
    for r in rp:
        v = lam * v + (1 - lam) * r * r
    return float(np.sqrt(v))


@dataclass
class VarInputs:
    est: pd.DataFrame          # estimation window of simple returns
    w: np.ndarray
    V: float
    conf: tuple
    horizons: tuple
    n_sims: int
    seed: int
    mc_dist: str = "normal"


def var_table(inp: VarInputs) -> pd.DataFrame:
    mu, Sigma = inp.est.mean().values, inp.est.cov().values
    rows = []
    for h in inp.horizons:
        for a in inp.conf:
            hv, he = historical_var_es(inp.est.values, inp.w, inp.V, a, h)
            pv, pe = parametric_var_es(mu, Sigma, inp.w, inp.V, a, h)
            mv, me = mc_var_es(mu, Sigma, inp.w, inp.V, a, h, inp.n_sims, inp.seed, inp.mc_dist)
            rows += [("Historical", h, a, hv, he), ("Parametric", h, a, pv, pe), ("Monte Carlo", h, a, mv, me)]
            if h > 1:
                h1v, h1e = historical_var_es(inp.est.values, inp.w, inp.V, a, 1)
                rows.append(("Historical (√h rule)", h, a, h1v * np.sqrt(h), h1e * np.sqrt(h)))
    return pd.DataFrame(rows, columns=["Method", "Horizon", "Confidence", "VaR", "ES"])


def pivot_var(res: pd.DataFrame, h: int) -> pd.DataFrame:
    sub = res[res.Horizon == h]
    p = sub.pivot(index="Method", columns="Confidence", values=["VaR", "ES"])
    p.columns = [f"{m} {c * 100:g}%" for m, c in p.columns]
    order = [m for m in ["Historical", "Historical (√h rule)", "Parametric", "Monte Carlo"] if m in p.index]
    return p.loc[order]


def parametric_variants(est, w, V, conf, n_sims, seed):
    rp = est.values @ w
    mu_p, sig_p = float(rp.mean()), float(np.std(rp, ddof=1))
    S_, K_ = float(stats.skew(rp)), float(stats.kurtosis(rp))
    nu = float(stats.t.fit(rp)[0])
    sig_ew = ewma_sigma(rp)
    mu, Sigma = est.mean().values, est.cov().values
    sig_p = float(np.sqrt(w @ Sigma @ w)); mu_p = float(w @ mu)
    rows = []
    for a in conf:
        z = stats.norm.ppf(a)
        zt = stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu) if nu > 2 else np.nan
        rows.append({"Confidence": a,
                     "Normal": (z * sig_p - mu_p) * V,
                     f"Student-t (ν = {nu:.1f})": (zt * sig_p - mu_p) * V,
                     "Cornish-Fisher": (cornish_fisher_z(a, S_, K_) * sig_p - mu_p) * V,
                     "EWMA λ = 0.94": (z * sig_ew - mu_p) * V,
                     "MC Student-t (ν = 5)": mc_var_es(mu, Sigma, w, V, a, 1, n_sims, seed, "t")[0]})
    info = {"skew": S_, "exkurt": K_, "nu": nu, "sigma_ewma": sig_ew, "sigma_p": sig_p, "mu_p": mu_p}
    return pd.DataFrame(rows).set_index("Confidence"), info


def mc_convergence(est, w, V, alpha=0.99, sizes=(1_000, 10_000, 100_000), runs=25, seed=42):
    mu, Sigma = est.mean().values, est.cov().values
    rows = []
    for n in sizes:
        vals = [mc_var_es(mu, Sigma, w, V, alpha, 1, n, seed + i)[0] for i in range(runs)]
        rows.append({"Simulations": n, "Mean VaR": np.mean(vals), "Std. error": np.std(vals, ddof=1),
                     "Min": np.min(vals), "Max": np.max(vals)})
    return pd.DataFrame(rows).set_index("Simulations")


def risk_contributions(est, w, V, alpha):
    tick = list(est.columns)
    mu, Sigma = est.mean().values, est.cov().values
    sig_p = float(np.sqrt(w @ Sigma @ w))
    z = stats.norm.ppf(alpha)
    comp_var = V * w * (z * (Sigma @ w) / sig_p - mu)
    rp = est.values @ w
    v_h, _ = var_es_from_pnl(V * rp, alpha)
    tail = V * rp <= -v_h
    comp_es = -V * (est.values[tail] * w).mean(axis=0)
    df = pd.DataFrame({"Weight": w, "Ann. volatility": np.sqrt(np.diag(Sigma) * TRADING_DAYS),
                       "Component VaR (param.)": comp_var, "Share of VaR": comp_var / comp_var.sum(),
                       "Component ES (hist.)": comp_es, "Share of ES": comp_es / comp_es.sum(),
                       "Marginal VaR": z * (Sigma @ w) / sig_p - mu}, index=tick)
    return df


# --------------------------------------------------------------------------- backtesting
def kupiec(x, T, p):
    ph = x / T
    if x == 0:
        lr = -2 * T * np.log(1 - p)
    elif x == T:
        lr = -2 * T * np.log(p)
    else:
        lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p) - (T - x) * np.log(1 - ph) - x * np.log(ph))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(exc):
    e = np.asarray(exc, dtype=int)
    n00 = np.sum((e[:-1] == 0) & (e[1:] == 0)); n01 = np.sum((e[:-1] == 0) & (e[1:] == 1))
    n10 = np.sum((e[:-1] == 1) & (e[1:] == 0)); n11 = np.sum((e[:-1] == 1) & (e[1:] == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(pp, a, b):
        return (a * np.log(1 - pp) if a and pp < 1 else 0) + (b * np.log(pp) if b and pp > 0 else 0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def traffic_light(x, T, p):
    if T == 250 and abs(p - 0.01) < 1e-9:
        return "Green" if x <= 4 else ("Yellow" if x <= 9 else "Red")
    cdf = stats.binom.cdf(x - 1, T, p) if x > 0 else 0.0
    return "Green" if cdf < 0.95 else ("Yellow" if cdf < 0.9999 else "Red")


def run_backtest(R, w, V, alpha, win=500, include_mc=True, mc_sims=5_000, seed=42):
    Rv, w = R.values, np.asarray(w)
    rp_all = Rv @ w
    z = stats.norm.ppf(alpha)
    ew = rp_all[:win].var()
    out = []
    for t in range(win, len(Rv)):
        X = Rv[t - win:t]; rp = rp_all[t - win:t]
        m, S = X.mean(0), np.cov(X, rowvar=False)
        s = np.sqrt(w @ S @ w)
        ew = 0.94 * ew + 0.06 * rp_all[t - 1] ** 2
        row = [R.index[t], rp_all[t] * V, -np.quantile(rp, 1 - alpha) * V, (z * s - w @ m) * V,
               (z * np.sqrt(ew) - w @ m) * V]
        if include_mc:
            sims = simulate_returns(m, S, mc_sims, 1, seed=seed + t) @ w
            row.append(-np.quantile(sims, 1 - alpha) * V)
        out.append(row)
    cols = ["Date", "P&L", "Historical", "Parametric", "EWMA parametric"] + (["Monte Carlo"] if include_mc else [])
    return pd.DataFrame(out, columns=cols).set_index("Date")


def backtest_summary(bt: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for m in [c for c in bt.columns if c != "P&L"]:
        exc = (-bt["P&L"] > bt[m]).astype(int)
        T, x = len(exc), int(exc.sum())
        lr_k, p_k = kupiec(x, T, 1 - alpha)
        lr_c, p_c = christoffersen(exc.values)
        last = int(exc.iloc[-250:].sum())
        rows.append({"Model": m, "Days": T, "Expected": T * (1 - alpha), "Exceptions": x, "Rate": x / T,
                     "Kupiec LR": lr_k, "Kupiec p": p_k, "Christoffersen p": p_c,
                     "Zone (all days)": traffic_light(x, T, 1 - alpha),
                     "Exc. last 250d": last, "Zone last 250d": traffic_light(last, 250, 1 - alpha)})
    return pd.DataFrame(rows).set_index("Model")


# --------------------------------------------------------------------------- stress
def replay_episode(prices, W, V, d0, d1, proxy):
    if pd.Timestamp(d0) < prices.index.min() or pd.Timestamp(d0) > prices.index.max():
        return None
    win = prices.loc[d0:d1]
    if len(win) < 2:
        return None
    ret, used = {}, {}
    for t in W.index:
        src = t if not np.isnan(win[t].iloc[0]) else proxy
        ret[t] = win[src].iloc[-1] / win[src].iloc[0] - 1
        used[t] = src
    ret = pd.Series(ret)
    path = (win[W.index] / win[W.index].iloc[0]) @ W
    return {"asset_returns": ret, "loss": -float(W @ ret) * V, "path": path * V,
            "max_dd": float((path / path.cummax() - 1).min()), "days": len(win),
            "start": str(win.index[0].date()), "end": str(win.index[-1].date()),
            "contrib": W * ret * V, "proxies": {k: v for k, v in used.items() if v != k}}


def stressed_cov(corr, vols, rho=None, k=1.0):
    C = np.asarray(corr).copy()
    if rho is not None:
        C = np.full_like(C, rho)
        np.fill_diagonal(C, 1.0)
    D = np.diag(np.asarray(vols) * k)
    S = D @ C @ D
    lo = np.linalg.eigvalsh(S).min()
    if lo <= 0:
        S = S + np.eye(len(S)) * (abs(lo) + 1e-12)
    return S


def reverse_stress(W, Sigma, betas, L):
    w = W.values
    Sg = np.asarray(Sigma)
    sig_p = float(np.sqrt(w @ Sg @ w))
    r_star = -L * (Sg @ w) / (w @ Sg @ w)
    beta_p = float(W @ betas)
    return {"single": pd.Series(L / w, index=W.index), "beta_p": beta_p, "market": -L / beta_p,
            "r_star": pd.Series(r_star, index=W.index), "d1": L / sig_p, "d10": L / (sig_p * np.sqrt(10)),
            "d_ann": L / (sig_p * np.sqrt(TRADING_DAYS)), "p_ann": float(stats.norm.sf(L / (sig_p * np.sqrt(TRADING_DAYS))))}


def concentration_path(est, w, V, k_asset, steps=11):
    tick = list(est.columns)
    e_k = np.array([1.0 if t == k_asset else 0.0 for t in tick])
    Sigma, mu = est.cov().values, est.mean().values
    vols = np.sqrt(np.diag(Sigma))
    rows = []
    for lam in np.linspace(0, 1, steps):
        wl = (1 - lam) * w + lam * e_k
        hv, he = historical_var_es(est.values, wl, V, 0.99)
        pv, _ = parametric_var_es(mu, Sigma, wl, V, 0.99)
        sp = np.sqrt(wl @ Sigma @ wl)
        rows.append({"Shift λ": lam, "Weight in asset": wl[tick.index(k_asset)], "Effective N": 1 / np.sum(wl**2),
                     "Ann. volatility": sp * np.sqrt(TRADING_DAYS), "Diversification ratio": (wl @ vols) / sp,
                     "Hist. VaR 99%": hv, "Hist. ES 99%": he, "Param. VaR 99%": pv})
    return pd.DataFrame(rows)
