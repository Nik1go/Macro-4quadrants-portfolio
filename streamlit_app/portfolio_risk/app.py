"""
Portfolio Risk Engine — Streamlit interface.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

import risk_engine as rk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
PORTFOLIO_VALUE = 10_000.0
STUDY_START = date(2019, 1, 1)
STUDY_END = date(2026, 10, 8)
RISK_FREE_RATE = 0.02

st.set_page_config(page_title="Portfolio Risk Engine", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================== design tokens
INK, SLATE, LINE, PANEL = "#f4f6ff", "#b8c1e3", "#3d4263", "#1e2139"
RISK, DEEP, TEAL, AMBER, GREEN = "#ff7f8a", "#ffb6c1", "#50dfc2", "#ffcf75", "#7cdb9b"
ASSET_COLORS = ["#00d4ff", "#50dfc2", "#ffcf75", "#ff7f8a", "#b59bff", "#75baff", "#c6d96f", "#a8b7d1",
                "#f49ac2", "#66d9e8", "#d7a58f", "#a5c4d0"]
METHOD_COLORS = {"Historical": "#00d4ff", "Parametric": "#ffcf75", "Monte Carlo": "#50dfc2",
                 "EWMA parametric": "#b59bff", "Historical (√h rule)": "#a8b7d1"}
ZONE_COLORS = {"Green": "#214a3c", "Yellow": "#574521", "Red": "#592d40"}

st.markdown("""
<style>
/* This stylesheet is rendered by Risk only. The portfolio keeps its old CSS. */
html body .stApp, html body .stApp .main { background:#0a0e27 !important; }
html body .stApp .block-container { padding-top:2.2rem !important;padding-bottom:3rem;max-width:1380px; }
html body .stApp h1, html body .stApp h2, html body .stApp h3 { color:#00d4ff !important; }
html body section[data-testid="stSidebar"] {
  background:#0a0e27 !important;border-right:1px solid #3d4263;
  min-width:min(400px,100vw) !important;max-width:400px !important;
}
html body section[data-testid="stSidebar"] > div:first-child {background:#0a0e27 !important;}
.stApp .subtitle, .stApp .small-note {color:#b8c1e3 !important;}
.stApp .subtitle {font-size:.98rem;margin:.25rem 0 1.1rem;max-width:80ch;}
.kpi-strip {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid #3d4263;
  border-radius:12px;background:linear-gradient(135deg,#1e2139,#2a2d4a);margin-bottom:1.2rem;}
.kpi {padding:.85rem 1rem;border-right:1px solid #3d4263;min-width:0;}
.kpi:last-child {border-right:none;}
.stApp .kpi .lab, .stApp .kpi .note {color:#b8c1e3 !important;font-size:.76rem;}
.stApp .kpi .lab {font-size:.78rem;font-weight:500;}
.stApp .kpi .val {color:#f4f6ff !important;font-size:1.55rem;font-weight:600;line-height:1.25;margin-top:.15rem;}
.stApp .kpi .val.risk {color:#ff7f8a !important;}
.reading {border-left:3px solid #00d4ff;background:#1e2139;padding:.75rem 1rem;margin:.4rem 0 1rem;
  color:#e8e8e8;font-size:.93rem;line-height:1.55;border-radius:0 8px 8px 0;}
.stApp .reading, .stApp .reading b {color:#e8e8e8 !important;}
.answer {border:1px solid #3d4263;border-radius:12px;background:#1e2139;padding:.9rem 1.1rem;margin-bottom:.8rem;line-height:1.6;}
.stApp .answer .q {font-weight:600;color:#00d4ff !important;margin-bottom:.35rem;font-size:1rem;}
.zone {display:inline-block;padding:.05rem .5rem;border-radius:3px;font-size:.8rem;font-weight:600;}
html body .stApp .stTabs [data-baseweb="tab-list"] {gap:0;border-bottom:2px solid #3d4263;overflow-x:auto;}
html body .stApp .stTabs [data-baseweb="tab"] {padding:12px 15px;color:#b8c1e3 !important;font-weight:600;white-space:nowrap;}
html body .stApp .stTabs [data-baseweb="tab"] p {color:inherit !important;}
html body .stApp .stTabs [data-baseweb="tab"][aria-selected="true"] {color:#00d4ff !important;background:#00d4ff14;}
html body .stApp .stTabs [data-baseweb="tab-highlight"] {background:#00d4ff;}
html body .stApp :is(.stButton,.stDownloadButton,.stFormSubmitButton) button {
  background:linear-gradient(90deg,#1e2139,#2a2d4a);color:#00d4ff !important;
  border:1px solid #3d4263 !important;border-radius:8px !important;padding:10px 16px;
  margin-top:0 !important;box-shadow:none;transform:none;
}
html body .stApp :is(.stButton,.stDownloadButton,.stFormSubmitButton) button * {color:inherit !important;}
html body .stApp :is(.stButton,.stDownloadButton,.stFormSubmitButton) button:hover {background:#163c54;border-color:#00d4ff !important;transform:none;}
html body .stApp :is(.stButton,.stDownloadButton,.stFormSubmitButton) button[kind="primary"] {
  background:linear-gradient(90deg,#00bfe8,#0099cc);border-color:#00d4ff !important;color:#081426 !important;
}
html body .stApp :is(button,input):focus-visible {outline:2px solid #00d4ff;outline-offset:3px;}
html body .stApp :is([data-baseweb="select"] > div,[data-baseweb="input"],[data-baseweb="base-input"],textarea) {
  background:#1e2139 !important;color:#e8e8e8 !important;border-color:#3d4263 !important;
}
html body .stApp :is(input,[data-baseweb="select"] span) {color:#e8e8e8 !important;}
html body :is([data-baseweb="popover"],[data-baseweb="popover"] ul,[role="listbox"]) {background:#1e2139 !important;}
html body :is([data-baseweb="popover"] li,[role="option"]) {color:#e8e8e8 !important;}
html body .stApp [data-testid="stMetric"] {background:#1e2139;border:1px solid #3d4263;border-radius:8px;padding:12px;}
html body .stApp [data-testid="stMetricValue"] {font-size:1.35rem;}
@media(max-width:1200px) {
  .kpi-strip {grid-template-columns:repeat(3,minmax(0,1fr));}
  .kpi:nth-child(3) {border-right:none;}.kpi:nth-child(-n+3) {border-bottom:1px solid #3d4263;}
}
@media(max-width:640px) {
  .kpi-strip {grid-template-columns:repeat(2,minmax(0,1fr));}
  .kpi {border-bottom:1px solid #3d4263;}.kpi:nth-child(even) {border-right:none;}
  html body .stApp .block-container {padding-left:1rem;padding-right:1rem;}
}
</style>
""", unsafe_allow_html=True)


def eur(x, d=0):
    return f"€{x:,.{d}f}" if pd.notna(x) else "—"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%" if pd.notna(x) else "—"


def reading(html):
    st.markdown(f'<div class="reading">{html}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380, title=None, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=46 if title else 16, b=70 if legend else 10),
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12, color=INK),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK)) if title else None,
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(font_family="IBM Plex Sans", bgcolor="#1e2139", font_color=INK), paper_bgcolor="#0a0e27", plot_bgcolor="#1e2139",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, linecolor=LINE, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def zone_style(v):
    return f"background-color: {ZONE_COLORS.get(v, 'white')}; font-weight: 600"


# =============================================================================== sidebar inputs
DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker": ["AAPL", "NVDA", "MC.PA", "SPY", "TLT", "GLD", "VNQ", "BTC-USD"],
    "Weight (%)": [12.0, 8.0, 10.0, 25.0, 15.0, 15.0, 10.0, 5.0],
    "Asset class": ["US equity", "US equity", "EU equity (LVMH)", "US equity index", "US Treasuries 20y+",
                    "Gold", "US REIT", "Crypto"],
})

with st.sidebar:
    st.markdown("### Portfolio Risk Engine")
    st.caption("Set the portfolio and the risk parameters, then run the analysis. Stress scenarios are set in the Stress tests tab.")
    st.markdown("## Portfolio")
    st.caption("Yahoo Finance symbols (e.g. MC.PA for LVMH in Paris, BTC-USD for Bitcoin). Add or delete rows freely.")
    port_df = st.data_editor(
        DEFAULT_PORTFOLIO, num_rows="dynamic", width="stretch", hide_index=True, key="port_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True, max_chars=15),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Asset class": st.column_config.TextColumn("Asset class"),
        })
    has_ticker = port_df["Ticker"].fillna("").astype(str).str.strip().ne("")
    total_weight = pd.to_numeric(port_df.loc[has_ticker, "Weight (%)"], errors="coerce").sum()
    st.markdown(f"**Total weight: {total_weight:.1f}% / 100%**")
    if abs(total_weight - 100) > 0.01:
        st.caption("Adjust the weights to reach 100% before running the analysis.")
    st.markdown("**Portfolio value: €10,000 (fixed)**")
    st.caption("Fixed study period: 1 January 2019 – 8 October 2026, inclusive. "
               "The saved portfolio's common trading-day prices start on 2 January 2019. "
               "Risk estimates use the most recent trading days selected below within this period.")
    with st.form("settings", border=False):
        st.markdown("## Risk settings")
        st.caption("Confidence levels")
        cc = st.columns(4)
        conf_flags = {0.90: cc[0].checkbox("90%", False), 0.95: cc[1].checkbox("95%", True),
                      0.975: cc[2].checkbox("97.5%", False), 0.99: cc[3].checkbox("99%", True)}
        horizon = st.number_input("Long horizon (trading days)", min_value=2, max_value=60, value=10, step=1,
                                  help="VaR and ES are always computed at 1 day and at this horizon.")
        window = st.slider("Estimation window (trading days)", 250, 2500, 1250, step=10,
                           help="Number of most recent daily returns used to estimate VaR and ES. 1,250 days ≈ 5 years.")
        n_sims = st.number_input("Monte Carlo simulations", min_value=1_000, max_value=1_000_000, value=100_000, step=10_000)
        mc_model = st.radio("Monte Carlo return model", ["Normal distribution", "Student-t distribution (5 degrees of freedom)"])
        st.caption("Normal: the standard bell curve. Student-t: extreme gains and losses are more frequent. "
                   "The degrees of freedom (ν) control the thickness of the tails: a smaller ν means more extreme outcomes. "
                   "Here ν is fixed at 5, with the same estimated volatility as the normal model.")
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)

        st.markdown("## Backtesting")
        bt_window = st.slider("Rolling estimation window (days)", 250, 1000, 500, step=10)
        bt_mc = st.toggle("Include Monte Carlo VaR in the backtest", value=True, help="Re-simulates every day (5,000 draws); slower.")
        st.caption("The Monte Carlo backtest uses the normal model as a fixed comparison benchmark.")

        submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")


# =============================================================================== validation
def validate(df, conf_flags):
    errs, warns = [], []
    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[(df["Ticker"] != "") & (df["Ticker"] != "NONE") & (df["Ticker"] != "NAN")]
    df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
    if len(df) < 2:
        errs.append("Enter at least two tickers.")
    dup = df["Ticker"][df["Ticker"].duplicated()].tolist()
    if dup:
        errs.append(f"Duplicate tickers: {', '.join(dup)}. Merge them into one row.")
    if df["Weight (%)"].isna().any():
        errs.append("Every ticker needs a weight.")
    elif (df["Weight (%)"] < 0).any():
        errs.append("Weights cannot be negative: short positions are not supported.")
    else:
        tot = df["Weight (%)"].sum()
        if tot <= 0:
            errs.append("Weights add up to 0%.")
        elif abs(tot - 100) > 0.01:
            errs.append(f"Weights add up to {tot:.2f}%, not 100%. Adjust the weights in the portfolio table.")
    conf = tuple(sorted(a for a, on in conf_flags.items() if on))
    if not conf:
        errs.append("Tick at least one confidence level.")
    return df.reset_index(drop=True), conf, errs, warns


if "config" not in st.session_state or submitted:
    pdf, conf, errs, warns = validate(port_df, conf_flags)
    if errs:
        if "config" not in st.session_state:
            st.title("Portfolio Risk Engine")
        for e in errs:
            st.error(e)
        if "config" not in st.session_state:
            st.stop()
        st.info("The results below still show the last valid run.")
    else:
        st.session_state.config = dict(
            tickers=tuple(pdf["Ticker"]), weights=tuple(pdf["Weight (%)"] / 100),
            classes=tuple(pdf["Asset class"].fillna("").astype(str)),
            conf=conf, horizon=int(horizon), window=int(window), n_sims=int(n_sims),
            dist="t" if mc_model.startswith("Student-t") else "normal",
            seed=int(seed), bt_window=int(bt_window), bt_mc=bool(bt_mc),
            run_at=datetime.now().strftime("%d %b %Y, %H:%M"), warns=warns)

cfg = st.session_state.config
# Apply the study constants even to a session opened before these settings were fixed.
cfg.update(value=PORTFOLIO_VALUE, start=str(STUDY_START), end=str(STUDY_END), rf=RISK_FREE_RATE)
cfg.pop("refresh", None)
for w_ in cfg["warns"]:
    st.warning(w_)


# =============================================================================== computations (cached)
@st.cache_data(show_spinner=False)
def get_data(tickers, start, end):
    raw, ccy, meta = rk.load_prices(list(tickers), start, end, DATA_DIR)
    prices, report = rk.build_eur_prices(raw, list(tickers), ccy)
    return raw, ccy, meta, prices, report


@st.cache_data(show_spinner=False)
def core(prices, weights, value, window, conf, horizon, n_sims, seed, dist, rf):
    W = pd.Series(weights, index=prices.columns)
    w = W.values
    R = prices.pct_change().dropna()
    LR = np.log(prices).diff().dropna()
    Rp, LRp = R @ w, np.log1p(R @ w)
    est = R.iloc[-window:]
    stats_tbl = rk.summary_stats(R.assign(Portfolio=Rp), LR.assign(Portfolio=LRp), rf)
    res = rk.var_table(rk.VarInputs(est, w, value, conf, (1, horizon), n_sims, seed, dist))
    variants, vinfo = rk.parametric_variants(est, w, value, conf, n_sims, seed)
    contrib = {a: rk.risk_contributions(est, w, value, a) for a in conf}
    pnl_hist = value * (est.values @ w)
    pnl_mc = value * rk.simulate_returns(est.mean().values, est.cov().values, min(n_sims, 200_000), 1, seed, dist) @ w
    return dict(R=R, LR=LR, Rp=Rp, est=est, stats=stats_tbl, res=res, variants=variants, vinfo=vinfo,
                contrib=contrib, pnl_hist=pnl_hist, pnl_mc=pnl_mc)


@st.cache_data(show_spinner=False)
def convergence(est, weights, value, seed):
    return rk.mc_convergence(est, np.array(weights), value, 0.99, seed=seed)


@st.cache_data(show_spinner=False)
def backtest(R, weights, value, conf, win, include_mc, seed):
    out = {}
    for a in conf:
        bt = rk.run_backtest(R, np.array(weights), value, a, win, include_mc, 5_000, seed)
        out[a] = (bt, rk.backtest_summary(bt, a))
    return out


@st.cache_data(show_spinner=False)
def rolling_corr(R):
    return rk.rolling_avg_corr(R, 63)


try:
    with st.spinner("Loading prices and aligning calendars…"):
        raw, CCY, META, prices, REPORT = get_data(cfg["tickers"], cfg["start"], cfg["end"])
except Exception as exc:
    st.title("Portfolio Risk Engine")
    st.error(f"Prices could not be loaded: {exc}")
    st.markdown("Check the ticker symbols on finance.yahoo.com and your internet connection, then run the analysis again.")
    st.stop()

if len(prices) < cfg["window"] + 20:
    st.error(f"Only {len(prices):,} common trading days are available for this portfolio, fewer than the estimation "
             f"window of {cfg['window']:,} days. Shorten the window or choose assets with more history in the fixed study period.")
    st.stop()

with st.spinner("Estimating VaR and Expected Shortfall…"):
    C = core(prices, cfg["weights"], cfg["value"], cfg["window"], cfg["conf"], cfg["horizon"], cfg["n_sims"],
             cfg["seed"], cfg["dist"], cfg["rf"])

TICK = list(cfg["tickers"])
W = pd.Series(cfg["weights"], index=TICK)
w = W.values
V, H, CONF = cfg["value"], cfg["horizon"], cfg["conf"]
R, est, ST, RES = C["R"], C["est"], C["stats"], C["res"]
A_TOP = max(CONF)
CLASS = {t: (c or rk.KNOWN_CLASS.get(t, "—")) for t, c in zip(TICK, cfg["classes"])}
COLOR = {t: ASSET_COLORS[i % len(ASSET_COLORS)] for i, t in enumerate(TICK)}
Sigma, mu = est.cov(), est.mean()
sig_p = float(np.sqrt(w @ Sigma.values @ w)); mu_p = float(w @ mu.values)
vols = np.sqrt(np.diag(Sigma.values))
div_ratio = float(w @ vols / sig_p)
P1, PH = rk.pivot_var(RES, 1), rk.pivot_var(RES, H)
lab = lambda a: f"{a * 100:g}%"
hv1, he1 = P1.loc["Historical", f"VaR {lab(A_TOP)}"], P1.loc["Historical", f"ES {lab(A_TOP)}"]
hvH = PH.loc["Historical", f"VaR {lab(A_TOP)}"]
mdd, mdd_date, dd = rk.max_drawdown(C["Rp"])

# =============================================================================== header
st.title("Portfolio Risk Engine")
st.markdown(
    f'<div class="subtitle">{eur(V)} portfolio in {len(TICK)} positions, valued in euros from Yahoo Finance adjusted prices '
    f'({prices.index.min():%d %b %Y} to {prices.index.max():%d %b %Y}). Risk estimated on the last {cfg["window"]:,} trading days; '
    f'run on {cfg["run_at"]}.</div>', unsafe_allow_html=True)

kpis = [
    (f"1-day VaR {lab(A_TOP)} (historical)", eur(hv1), f"{pct(hv1 / V, 2)} of the portfolio", True),
    (f"1-day ES {lab(A_TOP)} (historical)", eur(he1), f"{he1 / hv1:.2f} × VaR", True),
    (f"{H}-day VaR {lab(A_TOP)} (historical)", eur(hvH), f"{pct(hvH / V, 1)} of the portfolio", True),
    ("Annualised volatility", pct(ST.loc["Portfolio", "Ann. volatility"]), f"diversification ratio {div_ratio:.2f}", False),
    ("Sharpe ratio", f"{ST.loc['Portfolio', 'Sharpe']:.2f}", "Risk-free rate: 2% / year (fixed)", False),
    ("Maximum drawdown", pct(mdd), f"trough on {mdd_date:%d %b %Y}", False),
]
st.markdown('<div class="kpi-strip">' + "".join(
    f'<div class="kpi"><div class="lab">{l}</div><div class="val{" risk" if r else ""}">{v}</div><div class="note">{n}</div></div>'
    for l, v, n, r in kpis) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Data", "Performance", "Correlation", "VaR & ES", "Backtesting", "Stress tests",
                "Concentration", "Findings & export"])

# =============================================================================== 1. overview
with tabs[0]:
    ctr = C["contrib"][A_TOP]
    st.subheader("Holdings")
    if True:
        hold = pd.DataFrame({"Asset class": [CLASS[t] for t in TICK], "Currency": [CCY[t] for t in TICK],
                             "Weight": w, "Amount": w * V, "Ann. volatility": ST.loc[TICK, "Ann. volatility"].values,
                             f"Share of VaR {lab(A_TOP)}": ctr["Share of VaR"].values,
                             f"Share of ES {lab(A_TOP)}": ctr["Share of ES"].values}, index=TICK)
        st.dataframe(hold.style.format({"Weight": "{:.1%}", "Amount": "€{:,.0f}", "Ann. volatility": "{:.1%}",
                                        f"Share of VaR {lab(A_TOP)}": "{:.1%}", f"Share of ES {lab(A_TOP)}": "{:.1%}"}),
                     width="stretch")
        ccys = sorted({CCY[t] for t in TICK})
        st.markdown(f'<span class="small-note">{len(TICK)} positions quoted in {", ".join(ccys)}. Non-euro positions are not hedged, '
                    f'so exchange-rate moves are part of the risk.</span>',
                    unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Capital versus risk")
        fig = go.Figure()
        fig.add_bar(y=TICK, x=w, orientation="h", name="Share of capital", marker_color="#B8C4D3")
        fig.add_bar(y=TICK, x=ctr["Share of VaR"].values, orientation="h", name=f"Share of VaR {lab(A_TOP)}", marker_color=RISK)
        fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"), xaxis=dict(tickformat=".0%"))
        show(style_fig(fig, 330))
    over = ctr["Share of VaR"] - W
    top_over = over.idxmax()
    with right:
      st.subheader("What it shows")
      reading(f"<b>Reading.</b> {top_over} weighs {pct(W[top_over], 0)} of the capital but carries {pct(ctr['Share of VaR'][top_over], 0)} "
            f"of the {lab(A_TOP)} VaR, the largest gap between capital and risk. Positions whose red bar is shorter than the grey one "
            f"({', '.join(over[over < -0.02].index) or 'none'}) diversify the portfolio: they bring less risk than capital.")

    st.subheader("Prices in euros")
    log_scale = st.toggle("Logarithmic scale", value=True, key="logscale")
    reb = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for t in TICK:
        fig.add_scatter(x=reb.index, y=reb[t], name=t, line=dict(width=1.3, color=COLOR[t]))
    pv = (1 + C["Rp"]).cumprod() * 100
    fig.add_scatter(x=pv.index, y=pv, name="Portfolio", line=dict(width=2.6, color=INK))
    fig.update_yaxes(type="log" if log_scale else "linear", title="Base 100")
    show(style_fig(fig, 420))

# =============================================================================== 2. data
with tabs[1]:
    st.subheader("Source")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Common trading days", f"{REPORT['final_days']:,}")
    c2.metric("First common date", REPORT["first_common_date"])
    c3.metric("Last date", f"{prices.index.max():%Y-%m-%d}")
    c4.metric("Gaps left after filling", REPORT["nan_after_alignment"])
    st.markdown(f"**Provider.** {META.get('source', 'Yahoo Finance')}. Dataset downloaded on {META.get('downloaded', '—')}"
                f"{' (downloaded during this run)' if META.get('downloaded_now') else ' and reloaded from the saved file'}. "
                "Adjusted close prices include dividends and splits.")
    reading("<b>How the data are prepared.</b> Paris, New York and the 24/7 crypto market follow different calendars. The engine keeps "
            "only the days on which <b>every exchange-traded asset</b> traded, so that a holiday never creates an artificial zero return; "
            "crypto-assets and exchange rates are sampled on those days (a weekend Bitcoin move shows up in Monday's return). Gaps of up "
            "to 3 days are forward-filled. Every non-euro price is converted with Yahoo's EUR exchange rate "
            "(P<sub>EUR</sub> = P<sub>local</sub> / EURxxx), so the euro investor's currency risk is included.")
    cl, cr = st.columns([1.35, 1])
    with cl:
        st.markdown("**Assets and calendars**")
        info = pd.DataFrame({"Currency": [CCY[t] for t in TICK],
                             "First price": [raw[t].first_valid_index().date() for t in TICK],
                             "Own trading days": [int(raw[t].loc[prices.index.min():].notna().sum()) for t in TICK],
                             "Dropped days": [REPORT["excluded_days"][t] for t in TICK]}, index=TICK)
        st.dataframe(info, width="stretch")
    with cr:
        st.markdown("**Last prices (local currency and EUR)**")
        last = pd.DataFrame({"Local price": [raw[t].dropna().iloc[-1] for t in TICK], "Currency": [CCY[t] for t in TICK],
                             "EUR price": prices.iloc[-1].values}, index=TICK)
        st.dataframe(last.style.format({"Local price": "{:,.2f}", "EUR price": "€{:,.2f}"}), width="stretch")
    st.markdown("**Download the datasets**")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Raw Yahoo prices (CSV)", raw.to_csv().encode(), "yahoo_adj_close.csv", "text/csv", width="stretch")
    d2.download_button("Aligned EUR prices (CSV)", prices.to_csv().encode(), "prices_eur_aligned.csv", "text/csv", width="stretch")
    d3.download_button("Daily EUR returns (CSV)", R.to_csv().encode(), "returns_eur.csv", "text/csv", width="stretch")
    with st.expander("Preview the aligned EUR prices"):
        st.dataframe(prices.tail(250).sort_index(ascending=False).style.format("{:,.2f}"), width="stretch", height=320)

# =============================================================================== 3. performance
with tabs[2]:
    st.subheader("Return, volatility and risk-adjusted performance")
    st.dataframe(ST.style.format({"Ann. return": "{:.1%}", "Ann. volatility": "{:.1%}", "Sharpe": "{:.2f}", "Skewness": "{:.2f}",
                                  "Excess kurtosis": "{:.1f}", "Worst day": "{:.1%}", "Best day": "{:.1%}", "Max drawdown": "{:.1%}"})
                 .bar(subset=["Ann. volatility"], color="#6b2d46", vmin=0).bar(subset=["Sharpe"], color=["#6b2d46", "#245747"], align="zero"),
                 width="stretch")
    st.caption(f"Full sample in EUR. Annual return = mean daily log return × 252; volatility = daily standard deviation × √252; "
               "Sharpe = (annual return − 2%) / annual volatility. Risk-free rate fixed at 2% per year.")
    s_ = ST.drop("Portfolio")
    most_vol, least_vol = s_["Ann. volatility"].idxmax(), s_["Ann. volatility"].idxmin()
    best_sh = s_["Sharpe"].idxmax()
    reading(f"<b>Reading.</b> The most volatile holding is <b>{most_vol}</b> ({pct(s_.loc[most_vol, 'Ann. volatility'])} a year), "
            f"the least volatile {least_vol} ({pct(s_.loc[least_vol, 'Ann. volatility'])}). The portfolio's volatility "
            f"({pct(ST.loc['Portfolio', 'Ann. volatility'])}) is far below the average of its components, the effect of diversification. "
            f"The best risk-adjusted performance among the holdings belongs to <b>{best_sh}</b> (Sharpe {ST.loc[best_sh, 'Sharpe']:.2f}); "
            f"the portfolio reaches {ST.loc['Portfolio', 'Sharpe']:.2f}. Every series shows "
            f"excess kurtosis between {s_['Excess kurtosis'].min():.1f} and {s_['Excess kurtosis'].max():.1f}: extreme days are far more "
            f"frequent than a normal distribution allows, which matters for the parametric VaR.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for t in TICK:
            fig.add_scatter(x=[ST.loc[t, "Ann. volatility"]], y=[ST.loc[t, "Ann. return"]], mode="markers",
                            marker=dict(size=11 + 40 * W[t], color=COLOR[t]), name=t,
                            hovertemplate=f"{t}<br>volatility %{{x:.1%}}<br>return %{{y:.1%}}<extra></extra>")
        fig.add_scatter(x=[ST.loc["Portfolio", "Ann. volatility"]], y=[ST.loc["Portfolio", "Ann. return"]], mode="markers",
                        name="Portfolio", marker=dict(size=17, color=INK, symbol="star"))
        fig.update_xaxes(title="Annualised volatility", tickformat=".0%")
        fig.update_yaxes(title="Annualised log return", tickformat=".0%")
        show(style_fig(fig, 430, "Risk and return (marker size = weight)"))
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=RISK, width=1), fillcolor="rgba(179,38,30,.25)",
                        name="Drawdown")
        fig.update_yaxes(tickformat=".0%")
        show(style_fig(fig, 430, f"Portfolio drawdown (maximum {pct(mdd)} on {mdd_date:%d %b %Y})", legend=False))
    c1, c2 = st.columns(2)
    with c1:
        rv = C["Rp"].rolling(63).std() * np.sqrt(252)
        fig = go.Figure(go.Scatter(x=rv.index, y=rv, line=dict(color=INK, width=1.4), name="63-day volatility"))
        fig.add_scatter(x=[rv.index[0], rv.index[-1]], y=[ST.loc["Portfolio", "Ann. volatility"]] * 2, mode="lines",
                        line=dict(dash="dot", color=SLATE, width=1.2), name="Full-sample average")
        fig.update_yaxes(tickformat=".0%")
        show(style_fig(fig, 360, "Rolling 3-month volatility of the portfolio"))
    with c2:
        rp_ = C["Rp"].values
        fig = go.Figure()
        fig.add_histogram(x=rp_, nbinsx=120, histnorm="probability density", marker_color="#8FA3BC", name="Daily returns")
        xs = np.linspace(rp_.min(), rp_.max(), 400)
        fig.add_scatter(x=xs, y=stats.norm.pdf(xs, rp_.mean(), rp_.std()), line=dict(color=INK, width=1.5), name="Normal fit")
        fig.update_xaxes(tickformat=".0%")
        show(style_fig(fig, 340, "Distribution of daily portfolio returns"))
    with st.expander("Return and volatility formulas"):
        st.latex(r"R_t=\frac{P_t-P_{t-1}}{P_{t-1}},\qquad r_t=\ln\frac{P_t}{P_{t-1}},\qquad R_{p,t}=\sum_i w_i R_{i,t}")
        st.latex(r"\sigma=\sqrt{\tfrac{1}{n-1}\sum_t (r_t-\bar r)^2},\qquad \sigma_{ann}=\sigma_{daily}\sqrt{252},\qquad "
                 r"\text{Sharpe}=\frac{R_p-R_f}{\sigma_p}")
        st.markdown("Simple returns aggregate across assets (portfolio return = weighted sum), log returns aggregate over time. "
                    "The portfolio is rebalanced daily to constant weights.")

# =============================================================================== 4. correlation
with tabs[3]:
    st.subheader("How the assets move together")
    regimes = {"Estimation window": (est.index.min(), est.index.max()), "Full sample": (R.index.min(), R.index.max())}
    regimes.update({k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in rk.REGIMES.items()
                    if pd.Timestamp(v[0]) >= R.index.min() - pd.Timedelta(days=10) and pd.Timestamp(v[1]) <= R.index.max()})
    reg_corr = {k: R.loc[v[0]:v[1]].corr() for k, v in regimes.items()}
    choice = st.segmented_control("Period", list(regimes), default="Estimation window", key="corr_regime") or "Estimation window"
    c1, c2 = st.columns([1.05, 1])
    with c1:
        cm_ = reg_corr[choice]
        fig = go.Figure(go.Heatmap(z=cm_.values, x=TICK, y=TICK, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=np.round(cm_.values, 2), texttemplate="%{text}", textfont=dict(size=11),
                                   colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(style_fig(fig, 460, f"Correlation matrix: {choice} ({regimes[choice][0]:%b %Y} to {regimes[choice][1]:%b %Y})",
                       legend=False))
    with c2:
        rows = []
        for k, cm2 in reg_corr.items():
            r_ = {"Avg ρ": rk.avg_offdiag(cm2)}
            if "SPY" in TICK and "TLT" in TICK:
                r_["ρ SPY–TLT"] = cm2.loc["SPY", "TLT"]
            r_["σp at base vols"] = float(np.sqrt(w @ (np.outer(vols, vols) * cm2.values) @ w) * np.sqrt(252))
            rows.append(pd.Series(r_, name=k))
        reg_tbl = pd.DataFrame(rows)
        st.markdown("**Correlation regimes**")
        st.dataframe(reg_tbl.style.format({c: ("{:.1%}" if c == "σp at base vols" else "{:+.2f}") for c in reg_tbl.columns}),
                     width="stretch")
        st.caption("σp at base vols: portfolio volatility with the period's correlations and today's volatilities, isolating the "
                   "correlation effect.")
        st.markdown(f"**Diversification over the estimation window.** Portfolio volatility {pct(sig_p * np.sqrt(252))} against "
                    f"{pct(w @ vols * np.sqrt(252))} for the weighted average of individual volatilities: diversification ratio "
                    f"**{div_ratio:.2f}**.")
    rc = rolling_corr(R)
    fig = go.Figure(go.Scatter(x=rc.index, y=rc, line=dict(color=INK, width=1.3), name="63-day average correlation"))
    for k, (a0, a1) in rk.REGIMES.items():
        if k != "Calm year 2019" and pd.Timestamp(a0) >= rc.index.min():
            fig.add_vrect(x0=a0, x1=a1, fillcolor=RISK, opacity=0.08, line_width=0, annotation_text=k,
                          annotation_position="top left", annotation_font_size=10)
    show(style_fig(fig, 330, "Rolling 3-month average pairwise correlation (shaded: stress periods)", legend=False))
    calm = reg_tbl.loc["Calm year 2019", "Avg ρ"] if "Calm year 2019" in reg_tbl.index else np.nan
    worst_reg = reg_tbl.drop(["Estimation window", "Full sample"], errors="ignore")["Avg ρ"].idxmax() \
        if len(reg_tbl) > 2 else "Full sample"
    txt = (f"<b>Reading.</b> Diversification is weakest when it is needed most. The average correlation rises from "
           f"{calm:.2f} in calm 2019 to {reg_tbl.loc[worst_reg, 'Avg ρ']:.2f} during the {worst_reg.lower()}, which on its own lifts "
           f"portfolio volatility to {pct(reg_tbl.loc[worst_reg, 'σp at base vols'])}. ") if pd.notna(calm) else "<b>Reading.</b> "
    if "ρ SPY–TLT" in reg_tbl.columns and "Covid crash" in reg_tbl.index and "2022 rate shock" in reg_tbl.index:
        txt += (f"The stock–bond hedge depends on the type of crisis: Treasuries offset equities in the Covid crash "
                f"(ρ = {reg_tbl.loc['Covid crash', 'ρ SPY–TLT']:+.2f}) but fell with them in the 2022 inflation shock "
                f"(ρ = {reg_tbl.loc['2022 rate shock', 'ρ SPY–TLT']:+.2f}). ")
    txt += "Crisis windows are calendar-defined, because correlations measured only on the worst days are biased."
    reading(txt)

# =============================================================================== 5. VaR & ES
with tabs[4]:
    st.subheader("Value at Risk and Expected Shortfall")
    st.markdown(f"Absolute VaR and ES in euros, estimated on {cfg['window']:,} trading days "
                f"({est.index.min():%d %b %Y} to {est.index.max():%d %b %Y}); Monte Carlo with {cfg['n_sims']:,} "
                f"{'Student-t' if cfg['dist'] == 't' else 'normal'} draws, seed {cfg['seed']}.")
    c1, c2 = st.columns(2)
    fmt_e = lambda df: df.style.format("€{:,.0f}")
    with c1:
        st.markdown("**1-day horizon**")
        st.dataframe(fmt_e(P1), width="stretch")
    with c2:
        st.markdown(f"**{H}-day horizon**")
        st.dataframe(fmt_e(PH), width="stretch")
    st.caption(f"Portfolio value {eur(V)}. Source: Yahoo Finance adjusted close. {H}-day historical figures use overlapping "
               f"{H}-day compounded returns; the √h row scales the 1-day figures for comparison.")

    st.markdown("#### Loss distribution")
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        dist_m = st.radio("Distribution", ["Historical", "Monte Carlo"], key="dist_m")
        show_normal = st.checkbox("Overlay the normal fit", value=True)
        show_es = st.checkbox("Show Expected Shortfall", value=True)
    with cc2:
        pnl = C["pnl_hist"] if dist_m == "Historical" else C["pnl_mc"]
        fig = go.Figure()
        fig.add_histogram(x=pnl / 1e3, nbinsx=140, histnorm="probability density", marker_color="#8FA3BC", name="Daily P&L")
        if show_normal:
            xs = np.linspace(pnl.min(), pnl.max(), 400)
            fig.add_scatter(x=xs / 1e3, y=stats.norm.pdf(xs, mu_p * V, sig_p * V) * 1e3, line=dict(color=INK, width=1.4), name="Normal fit")
        dens, _ = np.histogram(pnl / 1e3, bins=140, density=True)
        ytop = dens.max() * 1.08
        shades = {A_TOP: (RISK, DEEP)}
        for i, a in enumerate(CONF):
            v_, e_ = rk.var_es_from_pnl(pnl, a)
            cv, ce = shades.get(a, (AMBER, "#d4aa65"))
            fig.add_scatter(x=[-v_ / 1e3] * 2, y=[0, ytop], mode="lines", line=dict(color=cv, dash="dash", width=1.6),
                            name=f"VaR {lab(a)}: {eur(v_)}")
            if show_es:
                fig.add_scatter(x=[-e_ / 1e3] * 2, y=[0, ytop], mode="lines", line=dict(color=ce, dash="dot", width=1.6),
                                name=f"ES {lab(a)}: {eur(e_)}")
        fig.update_yaxes(range=[0, ytop], title="Density")
        fig.update_xaxes(title="1-day P&L (€ thousands)")
        show(style_fig(fig, 440))
    vi = C["vinfo"]
    hv_ = P1.loc["Historical", f"VaR {lab(A_TOP)}"]; pv_ = P1.loc["Parametric", f"VaR {lab(A_TOP)}"]
    mv_ = P1.loc["Monte Carlo", f"VaR {lab(A_TOP)}"]
    reading(f"<b>Reading.</b> With {lab(A_TOP)} confidence, the portfolio should not lose more than <b>{eur(hv_)}</b> in one day "
            f"(historical); a larger loss is expected about once every {1 / (1 - A_TOP):.0f} trading days. When it happens, the average "
            f"loss is the ES, {eur(P1.loc['Historical', f'ES {lab(A_TOP)}'])}. The normal model gives {eur(pv_)} "
            f"({(pv_ / hv_ - 1) * 100:+.0f}% versus history) because returns have excess kurtosis of {vi['exkurt']:.1f}: more mass in the "
            f"centre and in the tails than the bell curve. Monte Carlo ({eur(mv_)}) "
            + ("uses Student-t with 5 degrees of freedom, so its tail risk can differ from the normal parametric figure."
               if cfg['dist'] == 't' else
               "uses the normal model, so it agrees with the parametric figure up to simulation noise."))

    st.markdown("#### Fat-tail variants of the parametric VaR (1 day)")
    st.dataframe(C["variants"].rename(index=lab).style.format("€{:,.0f}"), width="stretch")
    st.caption(f"Portfolio skewness {vi['skew']:.2f}, excess kurtosis {vi['exkurt']:.1f}, fitted Student-t ν = {vi['nu']:.1f}, "
               f"EWMA volatility {pct(vi['sigma_ewma'], 2)} a day against {pct(vi['sigma_p'], 2)} over the window. "
               f"ν here is estimated from the portfolio returns; it is separate from the fixed ν = 5 in the Monte Carlo option. "
               f"Student-t and Cornish-Fisher adjust the quantile for fat tails; EWMA weights recent days more heavily, so it "
               f"reflects today's volatility regime rather than the window average.")
    st.markdown("#### Monte Carlo convergence (VaR 99%, 25 independent runs per size)")
    conv = convergence(est, cfg["weights"], V, cfg["seed"])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(conv.style.format("€{:,.0f}"), width="stretch")
        st.caption("The standard error falls roughly as 1/√N: 100 times more simulations, 10 times less noise.")
    with c2:
        fig = go.Figure(go.Bar(x=[f"{n:,}" for n in conv.index], y=conv["Std. error"], marker_color=TEAL,
                               text=[eur(v) for v in conv["Std. error"]], textposition="outside", cliponaxis=False))
        fig.update_yaxes(tickprefix="€", tickformat=",.0f"); fig.update_xaxes(title="Simulations")
        show(style_fig(fig, 260, "Standard error of the Monte Carlo VaR", legend=False))

    st.markdown("#### Risk contribution by asset")
    a_c = st.radio("Confidence level", [lab(a) for a in CONF], index=len(CONF) - 1, horizontal=True, key="ctr_conf")
    a_c = [a for a in CONF if lab(a) == a_c][0]
    ctr = C["contrib"][a_c]
    if True:
        tbl = ctr.copy()
        tbl.loc["Total"] = [1, sig_p * np.sqrt(252), tbl["Component VaR (param.)"].sum(), 1, tbl["Component ES (hist.)"].sum(), 1, np.nan]
        st.dataframe(tbl.style.format({"Weight": "{:.1%}", "Ann. volatility": "{:.1%}", "Component VaR (param.)": "€{:,.0f}",
                                       "Share of VaR": "{:.1%}", "Component ES (hist.)": "€{:,.0f}", "Share of ES": "{:.1%}",
                                       "Marginal VaR": lambda v: "" if pd.isna(v) else f"{v:.4f}"}), width="stretch")
    if True:
        fig = go.Figure()
        fig.add_bar(x=TICK, y=ctr["Component VaR (param.)"], name="Component VaR (parametric)", marker_color=RISK)
        fig.add_bar(x=TICK, y=ctr["Component ES (hist.)"], name="Component ES (historical)", marker_color=DEEP)
        fig.update_yaxes(tickprefix="€", tickformat=",.0f")
        show(style_fig(fig, 340, "Euro contribution of each asset to VaR and ES"))
    with st.expander("Methodology: VaR, ES and risk contributions"):
        st.markdown("**Historical simulation.** Today's weights are applied to each past day of the window; VaR is the empirical "
                    "quantile of the resulting P&L and ES the average of the losses beyond it.")
        st.latex(r"\text{VaR}_\alpha=-q_{1-\alpha}(V\,R_p),\qquad \text{ES}_\alpha=-\mathbb E\left[V R_p\mid V R_p\le-\text{VaR}_\alpha\right]")
        st.markdown("**Parametric (normal).**")
        st.latex(r"\sigma_p=\sqrt{w^\top\Sigma w},\quad \text{VaR}_\alpha=(z_\alpha\sigma_p-\mu_p)V,\quad "
                 r"\text{ES}_\alpha=\left(\sigma_p\frac{\varphi(z_\alpha)}{1-\alpha}-\mu_p\right)V")
        st.markdown(f"**Monte Carlo.** Cholesky factorisation Σ = LLᵀ, then r = μ + Lz with z ~ N(0, I) (or Student-t scaled to unit "
                    f"variance); the {H}-day horizon compounds {H} simulated days per path.")
        st.markdown("**Component VaR (Euler).** Sums exactly to the parametric VaR; component ES averages each asset's P&L on the "
                    "historical tail days and sums to the historical ES.")
        st.latex(r"\text{CVaR}_i=V\,w_i\left(z_\alpha\frac{(\Sigma w)_i}{\sigma_p}-\mu_i\right)")

# =============================================================================== 6. backtesting
with tabs[5]:
    st.subheader("Does the VaR model hold out of sample?")
    st.markdown(f"Every model is re-estimated each day on the previous **{cfg['bt_window']} days** and its 1-day VaR is compared with the "
                f"next day's P&L of the current portfolio. A loss larger than the VaR is an **exception**.")
    with st.spinner("Running the rolling backtest (re-estimating every day)…"):
        BT = backtest(R, cfg["weights"], V, CONF, cfg["bt_window"], cfg["bt_mc"], cfg["seed"])
    a_b = st.radio("Confidence level", [lab(a) for a in CONF], index=len(CONF) - 1, horizontal=True, key="bt_conf")
    a_b = [a for a in CONF if lab(a) == a_b][0]
    bt, summ = BT[a_b]
    st.caption(f"Out-of-sample period {bt.index.min():%d %b %Y} to {bt.index.max():%d %b %Y}: {len(bt):,} days.")
    st.dataframe(summ.style.format({"Expected": "{:.1f}", "Rate": "{:.2%}", "Kupiec LR": "{:.2f}", "Kupiec p": "{:.3f}",
                                    "Christoffersen p": "{:.3f}"})
                 .map(zone_style, subset=["Zone (all days)", "Zone last 250d"]), width="stretch")
    models = [c for c in bt.columns if c != "P&L"]
    st.markdown("**Models shown on the chart**")
    mcols = st.columns(len(models))
    shown = [m for m, col in zip(models, mcols) if col.checkbox(m, value=m in ("Historical", "Parametric"), key=f"bt_{m}")]
    exc_model = st.radio("Mark exceptions of", models, horizontal=True, key="exc_model")
    fig = go.Figure()
    fig.add_bar(x=bt.index, y=bt["P&L"], name="Daily P&L", marker_color="#B8C4D3", marker_line_width=0)
    for m in shown:
        fig.add_scatter(x=bt.index, y=-bt[m], name=f"−VaR {m}", line=dict(width=1.3, color=METHOD_COLORS.get(m, SLATE)))
    exc = -bt["P&L"] > bt[exc_model]
    fig.add_scatter(x=bt.index[exc], y=bt["P&L"][exc], mode="markers", name=f"Exceptions ({exc_model})",
                    marker=dict(color=RISK, size=7, line=dict(color="white", width=1)))
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    show(style_fig(fig, 430))
    hist_row = summ.loc["Historical"]
    passing = [m for m in summ.index if summ.loc[m, "Kupiec p"] >= 0.05]
    reading(f"<b>Reading.</b> At {lab(a_b)}, {summ.loc['Historical', 'Days']:,} days should produce about "
            f"{hist_row['Expected']:.1f} exceptions. Historical VaR produced {int(hist_row['Exceptions'])} (Kupiec p "
            f"{hist_row['Kupiec p']:.2f}). Models not rejected by Kupiec at the 5% level: <b>{', '.join(passing) or 'none'}</b>. "
            f"A p-value below 0.05 means the exception frequency is statistically inconsistent with {lab(a_b)} confidence: too many "
            f"exceptions means the model underestimates risk, too few that it is over-conservative. Christoffersen p-values below 0.05 "
            f"would reveal exceptions arriving in clusters, i.e. a VaR that reacts too slowly to volatility changes.")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("The tests"):
            st.latex(r"LR_{POF}=-2\ln\frac{(1-p)^{T-x}p^{x}}{(1-\hat p)^{T-x}\hat p^{x}}\sim\chi^2_1,\qquad \hat p=\frac{x}{T}")
            st.markdown("**Christoffersen** compares the probability of an exception after a calm day with the probability after an "
                        "exception (independence, χ²₁). **Basel traffic light** at 99% on 250 days: green 0–4 exceptions, yellow 5–9, "
                        "red 10 or more. On the full sample and at other levels, zones use the binomial probability of observing fewer "
                        "exceptions (below 95%: green; below 99.99%: yellow).")
    with c2:
        with st.expander(f"List of exceptions ({exc_model})"):
            el = bt.loc[exc, ["P&L", exc_model]].rename(columns={exc_model: "VaR"})
            el["Excess loss"] = -el["P&L"] - el["VaR"]
            st.dataframe(el.sort_index(ascending=False).style.format("€{:,.0f}"), width="stretch", height=280)

# =============================================================================== 7. stress
with tabs[6]:
    st.subheader("Stress testing")
    st.markdown("Statistical measures answer “what is likely?”; stress tests answer “what if?”. Adjust the scenarios below; the "
                "results update immediately and are compared with the 1-day VaR and ES and with the "
                f"{H}-day VaR at {lab(A_TOP)}.")
    betas_mkt = "SPY" if "SPY" in TICK else TICK[0]
    betas = est.cov()[betas_mkt] / est[betas_mkt].var()
    corr_est = est.corr()

    s1, s2, s3 = st.columns([1, 1.2, 1])
    with s1:
        st.markdown("**Historical episodes**")
        ep_on = {}
        for k_, (nm, a0, a1) in rk.EPISODES.items():
            covered = pd.Timestamp(a0) >= prices.index.min()
            ep_on[k_] = st.checkbox(f"{nm} ({a0[:7]} to {a1[:7]})", value=covered, disabled=not covered, key=f"ep_{k_}",
                                    help=None if covered else "Not covered by the fixed study period (2019–2026).")
        use_custom = st.checkbox("Custom episode", value=False, key="ep_custom")
        if use_custom:
            ce1, ce2 = st.columns(2)
            c_start = ce1.date_input("From", value=date(2020, 2, 19), min_value=STUDY_START, max_value=STUDY_END, key="ce_s")
            c_end = ce2.date_input("To", value=date(2020, 3, 23), min_value=STUDY_START, max_value=STUDY_END, key="ce_e")
            c_name = st.text_input("Name", value="Custom episode", key="ce_n")
        proxy = st.text_input("Proxy for assets without history", value=betas_mkt, key="proxy").strip().upper()
        if proxy not in TICK:
            st.error(f"{proxy} is not in the portfolio; {betas_mkt} is used instead.")
            proxy = betas_mkt
    with s2:
        st.markdown("**Hypothetical shocks per asset (%)**")
        defaults = {"AAPL": -25, "NVDA": -35, "MC.PA": -15, "SPY": -18, "TLT": 4, "GLD": 5, "VNQ": -15, "BTC-USD": -45}
        shock_df = st.data_editor(pd.DataFrame({"Ticker": TICK, "Shock (%)": [float(defaults.get(t, -20)) for t in TICK]}),
                                  hide_index=True, width="stretch", disabled=["Ticker"], key="shock_editor",
                                  column_config={"Shock (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=300.0,
                                                                                          step=1.0, format="%.1f")})
        scen_name = st.text_input("Scenario name", value="Tech and risk-asset sell-off", key="scen_name")
        mkt_shock = st.slider(f"Market shock transmitted through betas (vs {betas_mkt}, %)", -60, 0, -20, key="mkt") / 100
    with s3:
        st.markdown("**Model stresses**")
        rho_star = st.slider("Correlation stress: all correlations set to ρ*", 0.80, 1.00, 0.90, 0.01, key="rho")
        vol_k = st.slider("Volatility stress: volatilities multiplied by k", 1.5, 3.0, 2.0, 0.1, key="volk")
        st.markdown("**Reverse stress test**")
        loss_lim = st.slider("Unacceptable loss (% of the portfolio)", 5, 60, 20, key="loss") / 100

    hv_t, he_t = rk.historical_var_es(est.values, w, V, A_TOP, 1)
    hvH_t, _ = rk.historical_var_es(est.values, w, V, A_TOP, H)
    rows, ep_res = [], {}

    def add(fam, name, loss, driver="—", detail=""):
        rows.append({"Family": fam, "Scenario": name, "Loss": loss, "Loss %": loss / V, f"× VaR {lab(A_TOP)} 1d": loss / hv_t,
                     f"× ES {lab(A_TOP)} 1d": loss / he_t, f"× VaR {lab(A_TOP)} {H}d": loss / hvH_t, "Main driver": driver,
                     "Detail": detail})

    eps = [(rk.EPISODES[k][0], rk.EPISODES[k][1], rk.EPISODES[k][2]) for k, on in ep_on.items() if on]
    if use_custom:
        if c_start >= c_end:
            st.error("The custom episode must end after it starts.")
        else:
            eps.append((c_name or "Custom episode", str(c_start), str(c_end)))
    for nm, a0, a1 in eps:
        out = rk.replay_episode(prices, W, V, a0, a1, proxy)
        if out is None:
            st.warning(f"{nm}: the window {a0} to {a1} is outside the data and was skipped.")
            continue
        ep_res[nm] = out
        add("Historical", f"{nm} ({out['start']} to {out['end']})", out["loss"], out["contrib"].idxmin(),
            f"{out['days']} days, worst point {pct(out['max_dd'])}" + (f", proxies: {out['proxies']}" if out["proxies"] else ""))
    shock = pd.Series(shock_df.set_index("Ticker")["Shock (%)"].astype(float) / 100).reindex(TICK).fillna(0)
    add("Hypothetical", scen_name, -float(W @ shock) * V, (W * shock).idxmin(),
        ", ".join(f"{t} {v:+.0%}" for t, v in shock.items()))
    bshock = betas * mkt_shock
    add("Hypothetical", f"Market {mkt_shock:+.0%} through betas", -float(W @ bshock) * V, (W * bshock).idxmin(),
        f"portfolio beta {float(W @ betas):.2f}")

    def risk_under(S, label):
        pv, pe = rk.parametric_var_es(mu.values, S, w, V, A_TOP)
        mv, me = rk.mc_var_es(mu.values, S, w, V, A_TOP, 1, 50_000, cfg["seed"])
        return {"Scenario": label, "Ann. volatility": float(np.sqrt(w @ S @ w * 252)), f"Param. VaR {lab(A_TOP)}": pv,
                f"Param. ES {lab(A_TOP)}": pe, f"MC VaR {lab(A_TOP)}": mv, f"MC ES {lab(A_TOP)}": me}

    sr = pd.DataFrame([risk_under(Sigma.values, "Base case"),
                       risk_under(rk.stressed_cov(corr_est, vols, rho=rho_star), f"Correlation stress ρ* = {rho_star:.2f}"),
                       risk_under(rk.stressed_cov(corr_est, vols, rho=1.0), "Perfect correlation ρ = 1"),
                       risk_under(rk.stressed_cov(corr_est, vols, k=vol_k), f"Volatility stress × {vol_k:.1f}"),
                       risk_under(rk.stressed_cov(corr_est, vols, rho=rho_star, k=vol_k), "Correlation and volatility")]).set_index("Scenario")
    for lbl in sr.index[1:]:
        if lbl != "Perfect correlation ρ = 1":
            add("Model stress", f"{lbl} (stressed 1-day ES)", sr.loc[lbl, f"Param. ES {lab(A_TOP)}"], "—",
                f"volatility {pct(sr.loc['Base case', 'Ann. volatility'])} → {pct(sr.loc[lbl, 'Ann. volatility'])}")
    rev = rk.reverse_stress(W, Sigma.values, betas, loss_lim)
    add("Reverse", f"Loss limit {loss_lim:.0%}: uniform fall of every asset", loss_lim * V, rev["r_star"].idxmin(),
        f"market through betas {pct(rev['market'])}")
    STRESS = pd.DataFrame(rows)

    st.markdown("#### Results")
    st.dataframe(STRESS.drop(columns=["Family"]).set_index("Scenario").style.format(
        {"Loss": "€{:,.0f}", "Loss %": "{:.1%}", f"× VaR {lab(A_TOP)} 1d": "{:.1f}×", f"× ES {lab(A_TOP)} 1d": "{:.1f}×",
         f"× VaR {lab(A_TOP)} {H}d": "{:.1f}×"}), width="stretch")
    plot_df = STRESS[STRESS.Family != "Reverse"]
    fam_col = {"Historical": "#00d4ff", "Hypothetical": RISK, "Model stress": AMBER}
    fig = go.Figure(go.Bar(y=plot_df["Scenario"], x=plot_df["Loss"], orientation="h",
                           marker_color=[fam_col[f] for f in plot_df["Family"]],
                           text=[f"{eur(l)} ({p:.1%})" for l, p in zip(plot_df["Loss"], plot_df["Loss %"])],
                           textposition="outside", cliponaxis=False))
    for v_, nm_, c_, pos in [(hv_t, f"VaR {lab(A_TOP)} 1d", INK, "top left"), (he_t, f"ES {lab(A_TOP)} 1d", SLATE, "top right"),
                             (hvH_t, f"VaR {lab(A_TOP)} {H}d", TEAL, "top right")]:
        fig.add_vline(x=v_, line_dash="dash", line_color=c_, annotation_text=nm_, annotation_font_size=10,
                      annotation_position=pos, annotation_font_color=c_)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="€", tickformat=",.0f", range=[0, plot_df["Loss"].max() * 1.25])
    show(style_fig(fig, 90 + 42 * len(plot_df), "Stress losses against the statistical risk measures", legend=False))
    worst = STRESS[STRESS.Family.isin(["Historical", "Hypothetical"])].sort_values("Loss").iloc[-1]
    reading(f"<b>Reading.</b> The worst scenario is <b>{worst['Scenario']}</b>: a loss of {eur(worst['Loss'])} "
            f"({pct(worst['Loss %'])}), {worst[f'× VaR {lab(A_TOP)} 1d']:.1f} times the 1-day VaR and "
            f"{worst[f'× VaR {lab(A_TOP)} {H}d']:.1f} times the {H}-day VaR; the largest euro loss comes from <b>{worst['Main driver']}</b>. "
            f"Stress losses dwarf VaR and ES because those measures are calibrated on one window at a short horizon and cannot see "
            f"multi-week crises, regime changes or hedges that break. Stress tests fill that gap but carry no probability, so both are needed.")

    if ep_res:
        st.markdown("#### Historical episodes in detail")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            ep_tbl = pd.DataFrame({k: v["asset_returns"] for k, v in ep_res.items()})
            st.dataframe(ep_tbl.style.format("{:+.1%}").background_gradient(cmap="RdYlGn", vmin=-1.0, vmax=1.0),
                         width="stretch")
            st.caption("Buy-and-hold return of each asset in EUR over the episode.")
        with c2:
            fig = go.Figure()
            for i, (k, v) in enumerate(ep_res.items()):
                path = v["path"] / V - 1
                fig.add_scatter(x=np.arange(len(path)), y=path.values, name=k, line=dict(width=1.8, color=ASSET_COLORS[i]))
            fig.update_yaxes(tickformat=".0%")
            show(style_fig(fig, 380, "Portfolio path through each episode (x: trading days)"))

    st.markdown("#### Correlation and volatility stress")
    st.dataframe(sr.style.format({"Ann. volatility": "{:.1%}"} | {c: "€{:,.0f}" for c in sr.columns if c != "Ann. volatility"}),
                 width="stretch")
    if True:
        reading(f"Raising every correlation to {rho_star:.2f} lifts portfolio volatility from "
                f"{pct(sr.loc['Base case', 'Ann. volatility'])} to {pct(sr.iloc[1]['Ann. volatility'])}, the loss of diversification. "
                f"Doubling volatilities scales VaR and ES almost proportionally ({vol_k:.1f}×), as in March 2020. Combined, the stressed "
                f"1-day ES reaches {eur(sr.loc['Correlation and volatility', f'Param. ES {lab(A_TOP)}'])}.")

    st.markdown("#### Reverse stress test")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        m1, m2 = st.columns(2)
        m1.metric("Loss limit", eur(loss_lim * V)); m1.caption(f"{loss_lim:.0%} of the portfolio")
        m2.metric("Market fall that breaches it", pct(rev["market"])); m2.caption(f"through a portfolio beta of {rev['beta_p']:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("Plausibility in one day", f"{rev['d1']:.1f} σ"); m3.caption("distance of the most plausible scenario")
        m4.metric("Plausibility over one year", f"{rev['d_ann']:.2f} σ"); m4.caption(f"≈ {pct(rev['p_ann'])} probability under normality")
        single = rev["single"]
        feas = single[single <= 1]
        st.markdown("**Single asset able to cause the loss alone:** " +
                    (", ".join(f"{t} (fall of {pct(single[t], 0)})" for t in feas.index) if len(feas) else
                     "none; every asset would need to fall by more than 100%."))
    with c2:
        fig = go.Figure(go.Bar(x=TICK, y=rev["r_star"].values, marker_color=[RISK if v < 0 else TEAL for v in rev["r_star"].values],
                               text=[pct(v) for v in rev["r_star"].values], textposition="outside", cliponaxis=False))
        fig.update_yaxes(tickformat=".0%")
        show(style_fig(fig, 320, "Most plausible joint shock r* that produces the loss", legend=False))
    with st.expander("Method of the reverse stress test"):
        st.markdown("Among all shock vectors producing the loss L, the most plausible under the covariance model minimises the "
                    "Mahalanobis distance. The solution is proportional to Σw, and its distance equals L / σp.")
        st.latex(r"r^*=-L\,\frac{\Sigma w}{w^\top\Sigma w},\qquad \sqrt{r^{*\top}\Sigma^{-1}r^*}=\frac{L}{\sigma_p}")

# =============================================================================== 8. concentration
with tabs[7]:
    st.subheader("What happens when the portfolio becomes more concentrated?")
    default_k = ST.drop("Portfolio")["Ann. volatility"].idxmax()
    k_asset = st.segmented_control("Concentrate into", TICK, default=default_k, key="conc_asset") or default_k
    cp = rk.concentration_path(est, w, V, k_asset)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=cp["Weight in asset"], y=cp["Hist. VaR 99%"], name="Historical VaR 99%", line=dict(color=RISK, width=2), mode="lines+markers")
        fig.add_scatter(x=cp["Weight in asset"], y=cp["Hist. ES 99%"], name="Historical ES 99%", line=dict(color=DEEP, width=2, dash="dot"), mode="lines+markers")
        fig.add_scatter(x=cp["Weight in asset"], y=cp["Effective N"], name="Effective number of positions",
                        line=dict(color=SLATE, width=1.5, dash="dash"), secondary_y=True)
        fig.update_xaxes(tickformat=".0%", title=f"Weight in {k_asset}")
        fig.update_yaxes(tickprefix="€", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title="Effective N", secondary_y=True, showgrid=False)
        show(style_fig(fig, 430, f"Moving capital into {k_asset} step by step"))
    with c2:
        w_eq = np.ones(len(w)) / len(w)
        eqv, eqe = rk.historical_var_es(est.values, w_eq, V, 0.99)
        cmp_ = pd.DataFrame({"Current": [1 / np.sum(w**2), sig_p * np.sqrt(252), cp.iloc[0]["Hist. VaR 99%"], cp.iloc[0]["Hist. ES 99%"]],
                             "Equal weights": [len(w), np.sqrt(w_eq @ Sigma.values @ w_eq * 252), eqv, eqe],
                             f"100% {k_asset}": [1, cp.iloc[-1]["Ann. volatility"], cp.iloc[-1]["Hist. VaR 99%"], cp.iloc[-1]["Hist. ES 99%"]]},
                            index=["Effective N", "Ann. volatility", "Hist. VaR 99%", "Hist. ES 99%"])
        st.dataframe(cmp_.style.format("{:.1f}", subset=pd.IndexSlice[["Effective N"], :])
                     .format("{:.1%}", subset=pd.IndexSlice[["Ann. volatility"], :])
                     .format("€{:,.0f}", subset=pd.IndexSlice[["Hist. VaR 99%", "Hist. ES 99%"], :]), width="stretch")
        st.caption("Effective N = 1 / Σ wᵢ², the number of equally weighted positions with the same concentration.")
    st.dataframe(cp.set_index("Shift λ").style.format({"Weight in asset": "{:.0%}", "Effective N": "{:.1f}", "Ann. volatility": "{:.1%}",
                                                     "Diversification ratio": "{:.2f}", "Hist. VaR 99%": "€{:,.0f}",
                                                     "Hist. ES 99%": "€{:,.0f}", "Param. VaR 99%": "€{:,.0f}"}),
                 width="stretch")
    reading(f"<b>Reading.</b> Shifting all the capital into {k_asset} takes the 99% VaR from {eur(cp.iloc[0]['Hist. VaR 99%'])} to "
            f"{eur(cp.iloc[-1]['Hist. VaR 99%'])} and the ES from {eur(cp.iloc[0]['Hist. ES 99%'])} to {eur(cp.iloc[-1]['Hist. ES 99%'])}. "
            f"Risk grows faster than the weight because two effects add up: more exposure to a volatile asset and a diversification "
            f"ratio falling from {cp.iloc[0]['Diversification ratio']:.2f} to 1.")

# =============================================================================== 9. findings & export
with tabs[8]:
    st.subheader("Answers to the project questions")
    b_top = BT[A_TOP][1]
    stats_ = ST.drop("Portfolio")
    stock_like = [t for t in TICK if "equity" in CLASS[t].lower() and "index" not in CLASS[t].lower()]
    most_vol_stock = stats_.loc[stock_like, "Ann. volatility"].idxmax() if stock_like else stats_["Ann. volatility"].idxmax()
    hist, par, mc = P1.loc["Historical"], P1.loc["Parametric"], P1.loc["Monte Carlo"]
    qa = [
        ("1. Which stock is the most volatile?",
         f"Among individual stocks, <b>{most_vol_stock}</b> with {pct(stats_.loc[most_vol_stock, 'Ann. volatility'])} annualised volatility "
         f"in EUR. Across all holdings the most volatile is <b>{stats_['Ann. volatility'].idxmax()}</b> "
         f"({pct(stats_['Ann. volatility'].max())}) and the least volatile {stats_['Ann. volatility'].idxmin()} "
         f"({pct(stats_['Ann. volatility'].min())}). The portfolio itself runs at {pct(ST.loc['Portfolio', 'Ann. volatility'])}."),
        ("2. Does VaR pass the backtest?",
         "; ".join(f"{m}: {int(r['Exceptions'])} exceptions for {r['Expected']:.1f} expected, Kupiec p = {r['Kupiec p']:.3f}, "
                   f"{r['Zone (all days)'].lower()} zone ({'passes' if r['Kupiec p'] >= 0.05 else 'rejected'})"
                   for m, r in b_top.iterrows()) + f" — at {lab(A_TOP)} over {int(b_top.iloc[0]['Days']):,} out-of-sample days."),
        ("3. How does diversification change with crisis correlations?",
         f"The diversification ratio is {div_ratio:.2f} over the estimation window. Correlations rise in crises (see the Correlation tab), "
         f"and forcing every correlation to {rho_star:.2f} raises portfolio volatility from {pct(sr.loc['Base case', 'Ann. volatility'])} to "
         f"{pct(sr.iloc[1]['Ann. volatility'])} and the 1-day ES from {eur(sr.loc['Base case', f'Param. ES {lab(A_TOP)}'])} to "
         f"{eur(sr.iloc[1][f'Param. ES {lab(A_TOP)}'])}: diversification disappears when it is most needed."),
        ("4. Why do Historical and Parametric VaR differ?",
         f"At {lab(A_TOP)}: {eur(hist[f'VaR {lab(A_TOP)}'])} historical against {eur(par[f'VaR {lab(A_TOP)}'])} parametric. The parametric "
         f"model keeps only μ and Σ and assumes normal returns, while portfolio returns show skewness {C['vinfo']['skew']:.2f} and excess "
         f"kurtosis {C['vinfo']['exkurt']:.1f}. Fat tails make the normal model too high near the centre (95%) and too low far in the tail "
         f"(99%). Historical simulation keeps the real tails but depends on the few extreme days inside the window."),
        ("5. Why does Monte Carlo VaR differ from Historical VaR?",
         f"Monte Carlo ({eur(mc[f'VaR {lab(A_TOP)}'])}) draws from a fitted {'Student-t' if cfg['dist'] == 't' else 'normal'} model, so it "
         f"converges to the model quantile, not to the empirical one: the gap with history is a model choice, not simulation noise "
         f"(standard error {eur(conv.loc[100_000, 'Std. error'])} with 100,000 normal draws). Selecting Student-t in the sidebar "
         f"makes extreme simulated returns more frequent; it does not guarantee agreement with history."),
        ("6. Why is Expected Shortfall useful in addition to VaR?",
         f"VaR says where the tail starts, ES how deep it goes: historical ES is {hist[f'ES {lab(A_TOP)}'] / hist[f'VaR {lab(A_TOP)}']:.2f} × VaR "
         f"against {par[f'ES {lab(A_TOP)}'] / par[f'VaR {lab(A_TOP)}']:.2f} under normality, so the real tail is thicker than VaR suggests. "
         f"ES is coherent (subadditive, it never penalises diversification) and is the Basel FRTB measure (97.5%)."),
        ("7. What happens when the portfolio becomes more concentrated?",
         f"Moving everything into {k_asset} raises the 99% VaR from {eur(cp.iloc[0]['Hist. VaR 99%'])} to {eur(cp.iloc[-1]['Hist. VaR 99%'])}; "
         f"the effective number of positions falls from {cp.iloc[0]['Effective N']:.1f} to 1 and diversification vanishes."),
        ("8. Which stress scenario is worst, and which asset drives it?",
         f"<b>{worst['Scenario']}</b>, a loss of {eur(worst['Loss'])} ({pct(worst['Loss %'])}), {worst[f'× VaR {lab(A_TOP)} 1d']:.1f} × the "
         f"1-day VaR; the largest euro loss comes from <b>{worst['Main driver']}</b>."),
    ]
    for q, a in qa:
        st.markdown(f'<div class="answer"><div class="q">{q}</div>{a}</div>', unsafe_allow_html=True)

    st.subheader("Export")
    st.markdown("All tables of this run in one Excel workbook, with the settings used.")

    def build_excel():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            settings = {k: (", ".join(map(str, v)) if isinstance(v, tuple) else v) for k, v in cfg.items() if k != "warns"}
            pd.Series(settings, name="value").to_frame().to_excel(xw, sheet_name="Settings")
            pd.DataFrame({"Weight": w, "Class": [CLASS[t] for t in TICK], "Currency": [CCY[t] for t in TICK]}, index=TICK).to_excel(xw, sheet_name="Portfolio")
            ST.to_excel(xw, sheet_name="Performance")
            P1.to_excel(xw, sheet_name="VaR ES 1d")
            PH.to_excel(xw, sheet_name=f"VaR ES {H}d")
            C["variants"].to_excel(xw, sheet_name="Parametric variants")
            C["contrib"][A_TOP].to_excel(xw, sheet_name="Risk contributions")
            for a, (b_, s_) in BT.items():
                s_.to_excel(xw, sheet_name=f"Backtest {lab(a)}")
            STRESS.to_excel(xw, sheet_name="Stress tests", index=False)
            sr.to_excel(xw, sheet_name="Stressed covariance")
            cp.to_excel(xw, sheet_name="Concentration", index=False)
            reg_tbl.to_excel(xw, sheet_name="Correlation regimes")
            est.corr().to_excel(xw, sheet_name="Correlation matrix")
        return buf.getvalue()

    st.download_button("Download the results (Excel)", build_excel(), f"risk_engine_results_{date.today()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("Portfolio Risk Engine — MSc Finance & Big Data, NEOMA Business School. Data: Yahoo Finance. "
               "For educational use; not investment advice.")


# The notebook stays below every tab so it is always at the bottom of the page.
st.divider()
st.subheader("Python notebook")
st.caption("Step-by-step calculations with short explanations in French. The download uses the portfolio and risk settings "
           "from the last valid analysis, with the predefined stress scenarios. Download both files into the same folder, "
           "then run the notebook from top to bottom in Jupyter or VS Code.")


def build_notebook():
    with open(os.path.join(APP_DIR, "portfolio_risk.ipynb"), encoding="utf-8") as file:
        notebook = json.load(file)
    parameters = {
        "portfolio_value": PORTFOLIO_VALUE, "start_date": str(STUDY_START), "end_date": str(STUDY_END),
        "risk_free_rate": RISK_FREE_RATE, "trading_days": 252,
        "tickers": TICK, "weights": [float(weight) for weight in cfg["weights"]], "currencies": {t: CCY[t] for t in TICK},
        "confidence_levels": list(CONF), "horizon": H, "estimation_window": cfg["window"],
        "n_simulations": cfg["n_sims"], "seed": cfg["seed"], "mc_distribution": cfg["dist"],
        "degrees_of_freedom": 5, "backtest_window": cfg["bt_window"], "include_mc_backtest": cfg["bt_mc"],
    }
    for cell in notebook["cells"]:
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = ["# Paramètres de la dernière analyse validée dans l'application.\n"] + [
                f"{name} = {value!r}\n" for name, value in parameters.items()]
    return json.dumps(notebook, ensure_ascii=False, indent=1).encode("utf-8")


notebook_column, data_column = st.columns(2)
notebook_column.download_button("Download the notebook (.ipynb)", build_notebook(), "portfolio_risk.ipynb",
                                "application/x-ipynb+json", width="stretch", on_click="ignore")
data_column.download_button("Download the notebook data (.csv)", raw.to_csv().encode("utf-8"), "yahoo_adj_close.csv",
                            "text/csv", width="stretch", on_click="ignore")
