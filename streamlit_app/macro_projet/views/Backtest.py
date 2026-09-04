"""
Page 2: Backtest & Performance
Historical strategy performance, heatmaps by quadrant regime.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import plotly.express as px
from data_loader import QUADRANT_NAMES, QUADRANT_COLORS

# Add spark_jobs to path to import optimization_engine
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
spark_jobs_dir = os.path.join(project_root, 'spark_jobs')
if spark_jobs_dir not in sys.path:
    sys.path.append(spark_jobs_dir)
from optimization_engine import get_carry_adjusted_returns_wide, run_efficient_frontier_points


def compute_single_metric(returns, metric):
    if len(returns) < 5: return 0.0
    if metric == 'Sharpe Ratio':
        stdev = np.std(returns)
        return (np.mean(returns) / stdev * np.sqrt(252)) if stdev > 0 else 0.0
    elif metric == 'Sortino Ratio':
        downside = returns[returns < 0]
        stdev_d = np.std(downside) if len(downside) > 0 else 1e-6
        return (np.mean(returns) * 252) / (stdev_d * np.sqrt(252))
    elif metric == 'Win Rate (% de jours positifs)':
        return np.mean(returns > 0) * 100
    return 0.0


def bootstrap_confidence(returns, metric, actual_score, n_sims=500):
    n = len(returns)
    if n < 5: return 0.0
    
    boot_indices = np.random.randint(0, n, (n_sims, n))
    boot_returns = returns.values[boot_indices]
    
    if metric == 'Sharpe Ratio':
        means = np.mean(boot_returns, axis=1)
        stds = np.std(boot_returns, axis=1)
        stds[stds == 0] = 1e-6
        scores = (means / stds) * np.sqrt(252)
        if actual_score > 0:
            return np.mean(scores > 0) * 100
        else:
            return np.mean(scores < 0) * 100
        
    elif metric == 'Sortino Ratio':
        scores = np.zeros(n_sims)
        for i in range(n_sims):
            ret = boot_returns[i]
            d = ret[ret < 0]
            sd = np.std(d) if len(d) > 0 else 1e-6
            scores[i] = (np.mean(ret) * 252) / (sd * np.sqrt(252))
        if actual_score > 0:
            return np.mean(scores > 0) * 100
        else:
            return np.mean(scores < 0) * 100
        
    elif metric == 'Win Rate (% de jours positifs)':
        wr = np.mean(boot_returns > 0, axis=1) * 100
        if actual_score > 50:
            return np.mean(wr > 50) * 100
        else:
            return np.mean(wr < 50) * 100
        
    return 0.0


def get_forex_carry_heatmap_data(forex_df, quadrants_df, quadrant_col, metric, inverse_fx=False):
    """
    Calcule la performance Forex incluant le Carry (différentiel de taux d'intérêt).
    """
    if forex_df is None or quadrants_df is None:
        return None, None

    # On s'assure d'avoir les taux
    rate_cols = ['TAUX_FED', 'TAUX_ECB', 'TAUX_BOJ', 'TAUX_BOC', 'TAUX_RBA', 'TAUX_BCB']
    missing_rates = [c for c in rate_cols if c not in quadrants_df.columns]
    if missing_rates:
        return None, None

    # Merge
    f_df = forex_df.copy()
    f_df['date'] = pd.to_datetime(f_df['date'])
    q_df = quadrants_df[['date', quadrant_col] + rate_cols].copy()
    q_df['date'] = pd.to_datetime(q_df['date'])

    df = pd.merge(f_df, q_df, on='date', how='inner').sort_values('date')
    if df.empty:
        return None, None

    rate_map = {
        'EUR': 'TAUX_ECB',
        'JPY': 'TAUX_BOJ',
        'CAD': 'TAUX_BOC',
        'AUD': 'TAUX_RBA',
        'BRL': 'TAUX_BCB'
    }

    pairs = [c for c in forex_df.columns if c.startswith('USD_')]
    available_pairs = [p for p in pairs if p.split('_')[1] in rate_map]

    if not available_pairs:
        return None, None

    results = []
    
    for pair in available_pairs:
        curr = pair.split('_')[1]
        rate_curr_col = rate_map[curr]
        
        if inverse_fx:
            p = 1.0 / df[pair]
            cap_ret = p.pct_change()
            carry_series = (df[rate_curr_col] - df['TAUX_FED']) / 100.0 / 252.0
            display_name = f"{curr}/USD"
        else:
            p = df[pair]
            cap_ret = p.pct_change()
            carry_series = (df['TAUX_FED'] - df[rate_curr_col]) / 100.0 / 252.0
            display_name = f"USD/{curr}"
            
        total_ret_series = (cap_ret + carry_series).fillna(0)
        
        quad_returns = total_ret_series.groupby(df[quadrant_col])
        row = {'Asset': display_name}
        for q in [1, 2, 3, 4]:
            if q in quad_returns.groups:
                group = quad_returns.get_group(q)
                row[f'Q{q}'] = compute_single_metric(group, metric)
            else:
                row[f'Q{q}'] = 0.0
        results.append(row)

    scores_df = pd.DataFrame(results).set_index('Asset')
    conf_df = pd.DataFrame(100.0, index=scores_df.index, columns=scores_df.columns)
    
    return scores_df, conf_df


def get_dynamic_heatmap_data(daily_df, date_quadrant_df, quadrant_col, metric, inverse_fx=False):
    if daily_df is None or date_quadrant_df is None or quadrant_col not in date_quadrant_df.columns:
        return None, None
        
    # Ensure date columns are of the same type (datetime64[ns]) before merging
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    date_quad_sub = date_quadrant_df[['date', quadrant_col]].copy()
    date_quad_sub['date'] = pd.to_datetime(date_quad_sub['date'])
        
    merged = pd.merge(daily_df, date_quad_sub, on='date', how='inner')
    # Sort by date to ensure pct_change is sequential
    merged = merged.sort_values('date')
    
    if merged.empty: return None, None
    
    # drop date from assets
    assets = [c for c in daily_df.columns if c != 'date']
    
    scores_matrix = pd.DataFrame(index=assets, columns=[1, 2, 3, 4], dtype=float).fillna(0)
    conf_matrix = pd.DataFrame(index=assets, columns=[1, 2, 3, 4], dtype=float).fillna(0)
    
    for q in [1, 2, 3, 4]:
        df_q = merged[merged[quadrant_col] == q]
        for a in assets:
            # Inversion logic for Forex
            if inverse_fx:
                # If price is USD/EUR, return is (Old/New - 1)
                ret_series = (merged[a].shift(1) / merged[a] - 1).loc[df_q.index].dropna()
            else:
                ret_series = merged[a].pct_change().loc[df_q.index].dropna()
            
            # Remove infinity values that might result from division by very small numbers
            ret_series = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            score = compute_single_metric(ret_series, metric)
            conf = bootstrap_confidence(ret_series, metric, score)
            
            scores_matrix.at[a, q] = score
            conf_matrix.at[a, q] = conf
            
    # If inverted, we rebuild the names (USD_EUR -> EUR/USD)
    if inverse_fx:
        new_index = [f"{c.split('_')[1]}/{c.split('_')[0]}" if '_' in c else f"Inv_{c}" for c in assets]
        scores_matrix.index = new_index
        conf_matrix.index = new_index

    return scores_matrix, conf_matrix


def _render_heatmap(scores_matrix, conf_matrix, fallback_perf_data, title, caption, metric_name, height=350):
    if scores_matrix is not None:
        pivot = scores_matrix
        text_vals = []
        for a in pivot.index:
            row_text = []
            for q in pivot.columns:
                val = pivot.at[a, q]
                conf = conf_matrix.at[a, q]
                
                if metric_name == 'Win Rate (% de jours positifs)':
                    row_text.append(f"{val:.1f}%<br>(Conf: {conf:.0f}%)")
                else:
                    row_text.append(f"{val:.2f}<br>(Conf: {conf:.0f}%)")
            text_vals.append(row_text)
            
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f'Q{int(c)}' for c in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn',
            text=text_vals,
            texttemplate='%{text}',
            textfont={"size": 10},
            zmid=50 if metric_name == 'Win Rate (% de jours positifs)' else 0
        ))
        
        # Adjust layout for smaller width in columns
        fig.update_layout(
            title={"text": title, "font": {"size": 14}},
            height=height,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(caption)
        return True
        
    elif fallback_perf_data is not None and 'sharpe' in fallback_perf_data.columns:
        df = fallback_perf_data.copy()
        pivot = df.pivot(index='asset', columns='quadrant', values='sharpe').fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f'Q{int(c)}' for c in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn',
            text=[[f'{v:.2f}' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            zmid=0
        ))
        fig.update_layout(
            title={"text": f"{title} (Fallback Sharpe)", "font": {"size": 14}},
            height=height,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Alerte: Calcul dynamique indisponible, affichage du Sharpe par défaut.")
        return True
        
    return False


def render(data):
    st.header("Performance Historique")

    # =========================================================
    # DATE RANGE FILTER (Rolling Window)
    # =========================================================
    df_bt_raw = data.get('backtest')
    df_oos_raw = data.get('backtest_oos')
    df_ibkr_live_raw = data.get('backtest_ibkr_live')

    # Determine available date range
    if df_bt_raw is not None and 'date' in df_bt_raw.columns:
        _min_date = df_bt_raw['date'].min().date()
        _max_date = df_bt_raw['date'].max().date()
    else:
        _min_date = pd.Timestamp('2009-01-01').date()
        _max_date = pd.Timestamp.today().date()

    with st.container():
        rc1, rc2, rc3 = st.columns([2, 2, 1])
        with rc1:
            filter_start = st.date_input(
                "Date début Backtest",
                value=_min_date,
                min_value=_min_date,
                max_value=_max_date,
                key="bt_filter_start"
            )
        with rc2:
            filter_end = st.date_input(
                "Date fin Backtest",
                value=_max_date,
                min_value=_min_date,
                max_value=_max_date,
                key="bt_filter_end"
            )
        with rc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺ Réinitialiser", key="bt_reset", use_container_width=True):
                filter_start = _min_date
                filter_end = _max_date

    # Apply filter
    _ts = pd.Timestamp
    if df_bt_raw is not None:
        df_bt = df_bt_raw[
            (df_bt_raw['date'] >= _ts(filter_start)) &
            (df_bt_raw['date'] <= _ts(filter_end))
        ].copy()
    else:
        df_bt = None

    if df_oos_raw is not None and 'date' in df_oos_raw.columns:
        df_oos_bt = df_oos_raw[
            (df_oos_raw['date'] >= _ts(filter_start)) &
            (df_oos_raw['date'] <= _ts(filter_end))
        ].copy()
    else:
        df_oos_bt = None

    if df_ibkr_live_raw is not None and 'date' in df_ibkr_live_raw.columns:
        df_ibkr_live = df_ibkr_live_raw[
            (df_ibkr_live_raw['date'] >= _ts(filter_start)) &
            (df_ibkr_live_raw['date'] <= _ts(filter_end))
        ].copy()
    else:
        df_ibkr_live = None

    _date_filtered = (filter_start != _min_date or filter_end != _max_date)
    if _date_filtered:
        st.caption(f"Fenêtre Backtest : **{filter_start}** → **{filter_end}** — toutes les métriques sont recalculées sur cette période.")

    # Calculate years for CAGR from the FILTERED window
    days = (df_bt['date'].max() - df_bt['date'].min()).days if df_bt is not None else 1
    years = days / 365.25 if days > 0 else 1.0

    # Helper: recalculate all metrics from a wealth series (always from filtered data)
    def _calc_metrics(w_series):
        """Returns (tot_ret, cagr, sharpe, vol, max_dd) from a wealth series."""
        if w_series is None or len(w_series) < 5:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if isinstance(w_series, np.ndarray):
            w_series = pd.Series(w_series)
        w = w_series.reset_index(drop=True)
        tot_ret = (w.iloc[-1] / w.iloc[0]) - 1
        cagr = (1 + tot_ret) ** (1 / years) - 1 if years > 0 else 0.0
        rets = w.pct_change().dropna()
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / vol if vol > 0 else 0.0
        peak = w.expanding(min_periods=1).max()
        max_dd = ((w - peak) / peak).min()
        return tot_ret, cagr, sharpe, vol, max_dd

    st.divider()

    # === Smooth Quadrant Distribution ===
    st.subheader("Répartition des Quadrants sur la Période Sélectionnée")
    if df_bt is not None and 'smooth_quadrant' in df_bt.columns:
        smooth_q_counts = df_bt['smooth_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        total_days = smooth_q_counts.sum()

        fig_smooth = go.Figure(data=[go.Bar(
            x=[f"Q{i} — {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=smooth_q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=[f"{v} jours ({v / total_days * 100:.1f}%)" for v in smooth_q_counts.values],
            textposition='auto',
        )])

        fig_smooth.update_layout(
            title=f"Répartition des Quadrants — {total_days} jours de trading",
            yaxis_title="Nombre de Jours",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_smooth, use_container_width=True)

        dominant_smooth_q = smooth_q_counts.idxmax()
        start_label = df_bt['date'].min().strftime('%Y-%m-%d') if 'date' in df_bt.columns else 'N/A'
        end_label   = df_bt['date'].max().strftime('%Y-%m-%d') if 'date' in df_bt.columns else 'N/A'
        st.info(
            f"Période: **{start_label}** → **{end_label}** | "
            f"Régime dominant: **Q{dominant_smooth_q} — {QUADRANT_NAMES.get(dominant_smooth_q)}** "
            f"({smooth_q_counts.max() / total_days * 100:.1f}%)"
        )
    else:
        st.warning("Données smooth_quadrant non disponibles. Lancez le backtest pour générer ces données.")

    st.divider()

    # === Strategy vs Benchmark ===
    st.subheader("Stratégie vs Benchmark ")
    if df_bt is not None:

        # Normalize all series to $1000 at the start of the filtered window
        def _norm(series, base=1000.0):
            """Rebases a wealth series so the first value = base."""
            s = series.dropna()
            if len(s) == 0: return series
            return s / s.iloc[0] * base

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bt['date'], y=_norm(df_bt['wealth']),
            name='Stratégie Modèle Complet', line=dict(color='cyan')
        ))

        if df_ibkr_live is not None and not df_ibkr_live.empty and 'ibkr_live_wealth' in df_ibkr_live.columns:
            fig.add_trace(go.Scatter(
                x=df_ibkr_live['date'], y=_norm(df_ibkr_live['ibkr_live_wealth']),
                name='Stratégie IBKR Live-Compatible', line=dict(color='#ff4f8b', dash='dot')
            ))

        # EUR conversion
        df_f = data.get('daily_forex')
        wealth_eur = None
        if df_f is not None and not df_f.empty and 'USD_EUR' in df_f.columns:
            df_ret_f = df_f.copy()
            df_ret_f['date'] = pd.to_datetime(df_ret_f['date'])
            df_ret_f = df_ret_f.set_index('date').sort_index()
            eur_usd_ret = df_ret_f['USD_EUR'].pct_change().fillna(0)
            idx = df_bt['date'].dt.tz_localize(None)
            eur_usd_ret = eur_usd_ret.reindex(idx).fillna(0)
            strat_ret = df_bt['portfolio_return'].values
            eur_ret = (1 + strat_ret) * (1 + eur_usd_ret.values) - 1
            wealth_eur_raw = pd.Series(1000 * (1 + eur_ret).cumprod(), index=df_bt.index)
            wealth_eur = _norm(wealth_eur_raw)
            fig.add_trace(go.Scatter(
                x=df_bt['date'], y=wealth_eur,
                name='Stratégie (EUR)', line=dict(color='#00ff88', dash='dash')
            ))

        fig.add_trace(go.Scatter(
            x=df_bt['date'], y=_norm(df_bt['SP500_wealth']),
            name='SP500', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=df_bt['date'], y=_norm(df_bt['GOLD_wealth']),
            name='Gold', line=dict(color='gold')
        ))

        fig.update_layout(
            height=490, yaxis_title="Wealth indicée ($, base 1 000)", xaxis_title="Date",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)
        if df_ibkr_live is not None and not df_ibkr_live.empty:
            st.caption(
                "Lecture: la courbe Modele Complet utilise l'univers de backtest/proxies. "
                "La courbe IBKR live-compatible remplace ces proxies par les instruments tradables "
                "du paper account et sert de reference pour comparer le live reel."
            )

        live_mapping = data.get('backtest_ibkr_live_mapping')
        live_stats = data.get('backtest_ibkr_live_stats')
        if live_mapping is not None and not live_mapping.empty:
            with st.expander("Mapping IBKR live-compatible"):
                cols = [c for c in [
                    'asset', 'model_proxy', 'ibkr_symbol', 'yahoo_ticker',
                    'isin', 'currency', 'ter', 'status', 'note'
                ] if c in live_mapping.columns]
                st.dataframe(live_mapping[cols], use_container_width=True, hide_index=True)
                if live_stats is not None and not live_stats.empty:
                    row = live_stats.iloc[0]
                    st.caption(
                        f"Base {row.get('base_currency', 'EUR')} | "
                        f"Missing weight max: {float(row.get('max_missing_weight', 0.0)) * 100:.2f}% | "
                        f"Missing assets: {row.get('missing_assets') or 'none'}"
                    )
    else:
        st.warning("Données backtest non disponibles")


    # =========================================================
    # SECTION 1.B: COMPARAISON DE PERFORMANCE
    # =========================================================
    st.divider()
    if data['stats'] is not None and df_bt is not None:

        # Build wealth map from FILTERED df_bt
        wealth_map = {
            "Stratégie Modèle Complet (USD)": df_bt['wealth'],
            "S&P 500 (Benchmark)": df_bt['SP500_wealth'],
            "Or (Gold)": df_bt['GOLD_wealth'],
        }



        if df_ibkr_live is not None and not df_ibkr_live.empty and 'ibkr_live_wealth' in df_ibkr_live.columns:
            wealth_map["Stratégie IBKR Live-Compatible"] = df_ibkr_live['ibkr_live_wealth']

        if wealth_eur is not None:
            wealth_map["Stratégie (EUR)"] = wealth_eur

        def display_compare_panel(key_id, default_selection_idx):
            choice = st.selectbox(
                f"Sélecteur {key_id} :",
                options=list(wealth_map.keys()),
                index=min(default_selection_idx, len(wealth_map) - 1),
                key=f"sel_{key_id}"
            )
            w_series = wealth_map[choice]
            tot_ret, cagr, sharpe, vol, max_dd = _calc_metrics(w_series)

            st.write(f"### {choice}")
            m_c1, m_c2 = st.columns(2)
            m_c1.metric("Total Return", f"{tot_ret * 100:.1f}%")
            m_c1.metric("Annual Return", f"{cagr * 100:.1f}%")
            m_c1.metric("Max Drawdown", f"{abs(max_dd) * 100:.1f}%")
            m_c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            m_c2.metric("Annual Vol", f"{vol * 100:.1f}%")

        st.write("### Comparaison de Performance")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            display_compare_panel("A", 0)
        with comp_col2:
            display_compare_panel("B", 1)


    st.divider()


    # =========================================================
    # SECTION 2.B: FRONTIERE EFFICIENTE PAR QUADRANT
    # =========================================================
    st.subheader("Optimisation et Frontiere Efficiente (Par Quadrant)")
    st.markdown("""L'algorithme génère un échantillonnage aléatoire de l'espace des poids de Markowitz (*via Numpy*) pour tester **8 000 portefeuilles virtuels**, en testant différentes combinaisons de poids (*weights*) sur notre univers d'actifs. Cette méthode stochastique permet d'explorer la "frontière efficiente" afin de trouver l'allocation statistiquement optimale face au risque.
    Le **Custom Z-Score Average** est la moyenne pondérée du Z-score de 3 ratios clés (Sharpe, Sortino et Calmar).
    """)
    
    col_q, col_m = st.columns(2)
    with col_q:
        selected_q = st.selectbox("Sélectionnez le Régime (Quadrant) :", options=[1, 2, 3, 4], 
            format_func=lambda x: f"Q{x} - {QUADRANT_NAMES.get(x)}")
    with col_m:
        selected_opt_metric = st.selectbox("Sélectionnez la Métrique d'Optimisation :",
            options=["custom", "sharpe", "sortino", "calmar"],
            format_func=lambda x: "Custom Z-Score Average" if x == "custom" else x.capitalize() + " Ratio")

    if data.get('daily_assets') is not None and data.get('quadrants') is not None:
        try:
            df_returns_all = get_carry_adjusted_returns_wide(
                data['daily_assets'], 
                data.get('daily_forex'), 
                data.get('indicators')
            )
            df_q = data['quadrants'].copy()
            df_q['date'] = pd.to_datetime(df_q['date'])
            df_q = df_q.set_index('date')
            
            # Shift quadrant by 1 day as in backtest to align predicting close with trading return
            df_q['assigned_quadrant'] = df_q['assigned_quadrant'].shift(1)
            q_dates = df_q[df_q['assigned_quadrant'] == selected_q].index.intersection(df_returns_all.index)
            
            if len(q_dates) < 20:
                st.warning("⚠️ Pas assez de données historiques pour ce quadrant.")
            else:
                TARGET_ASSETS = {
                    1: ['NASDAQ_100', 'SmallCAP', 'SP500', 'US_REIT_VNQ'],
                    2: ['NASDAQ_100', 'SmallCAP', 'SP500', 'GOLD_OZ_USD', 'COMMODITIES'],
                    3: ['SHORT_SP500', 'COMMODITIES', 'USD_JPY', 'USD_EUR'],
                    4: ['TREASURY_10Y', 'OBLIGATION', 'GOLD_OZ_USD']
                }
                
                allowed_assets = [c for c in TARGET_ASSETS.get(selected_q, []) if c in df_returns_all.columns]
                ret_q_opt = df_returns_all.loc[q_dates, allowed_assets].fillna(0)

                # Drop cols with absolutely zero return inside this quadrant
                ret_q_opt = ret_q_opt.loc[:, (ret_q_opt != 0).any(axis=0)]
                
                rf = 0.02
                if data.get('indicators') is not None and 'TAUX_FED' in data['indicators'].columns:
                    # Align dates with indicators
                    shared_idx = q_dates.intersection(data['indicators']['date'])
                    if len(shared_idx) > 0:
                        rf = data['indicators'].set_index('date').loc[shared_idx, 'TAUX_FED'].mean() / 100.0
                
                # Execute PyPortfolioOpt Efficient Frontier Simulation
                # Use cache for performance to prevent 8000 simulations on every ui click unless parameters change
                @st.cache_data(ttl=3600, show_spinner="Simulation de 8000 portefeuilles Markowitz...")
                def compute_ef_v6(returns_df_bytes, rf_rate):
                    import io
                    # Streamlit cache bug bypass by passing parquet bytes
                    r = pd.read_parquet(io.BytesIO(returns_df_bytes))
                    return run_efficient_frontier_points(r, rf_rate, n_sims=8000)

                # Convert dataframe to bytes for caching properly
                import io
                parquet_bytes = io.BytesIO()
                ret_q_opt.to_parquet(parquet_bytes)
                ef_res = compute_ef_v6(parquet_bytes.getvalue(), rf)
                
                # Render Plot
                frontier = ef_res['frontier_data']
                opt_key = f"opt_{selected_opt_metric}"
                opt_w = ef_res[opt_key]
                
                # Compute return and vol of the chosen optimal portfolio
                opt_w_array = np.array([opt_w.get(c, 0) for c in ret_q_opt.columns])
                opt_rp = ret_q_opt.dot(opt_w_array)
                opt_mean = opt_rp.mean() * 252
                opt_vol = opt_rp.std() * np.sqrt(252)
                
                # Min vol portfolio
                min_w_array = np.array([ef_res['min_vol_weights'].get(c, 0) for c in ret_q_opt.columns])
                min_rp = ret_q_opt.dot(min_w_array)
                min_mean = min_rp.mean() * 252
                min_vol = min_rp.std() * np.sqrt(252)

                # Extract EXACT weights used in the BACKTEST (Source of Truth)
                backtest_w_array = np.zeros(len(ret_q_opt.columns))
                if data.get('backtest') is not None:
                    df_b = data['backtest']
                    df_q = df_b[df_b['smooth_quadrant'] == selected_q]
                    if not df_q.empty:
                        for i, col in enumerate(ret_q_opt.columns):
                            # The column in CSV is 'ASSET_base_weight'
                            bw_col = f"{col}_base_weight"
                            if bw_col in df_q.columns:
                                backtest_w_array[i] = df_q[bw_col].median()
                
                # If no backtest data, just ignore
                bt_mean, bt_vol = None, None
                if backtest_w_array.sum() > 0.95: # Ensure we found the weights
                    backtest_w_array = backtest_w_array / backtest_w_array.sum()
                    bt_rp = ret_q_opt.dot(backtest_w_array)
                    bt_mean = bt_rp.mean() * 252
                    bt_vol = bt_rp.std() * np.sqrt(252)

                old_rp = ret_q_opt.sum(axis=1) * 0 # Removed IBKR old alloc logic

                fig_ef = go.Figure()
                
                # Add scatter of simulated portfolios
                metric_col_map = {
                    'custom': 'customs',
                    'sharpe': 'sharpes',
                    'sortino': 'sortinos',
                    'calmar': 'calmars'
                }
                metric_name_map = {
                    'custom': 'Custom Z-Score Average',
                    'sharpe': 'Sharpe Ratio',
                    'sortino': 'Sortino Ratio',
                    'calmar': 'Calmar Ratio'
                }
                c_key = metric_col_map.get(selected_opt_metric, 'sharpes')
                c_title = metric_name_map.get(selected_opt_metric, 'Sharpe Ratio')
                metric_color = frontier[c_key]
                
                # Format weights for hover template
                weights_hover = []
                for w in frontier['weights']:
                    hover_text = "<br>".join([f"{a}: {w[i]*100:.1f}%" for i, a in enumerate(ret_q_opt.columns) if w[i] > 0.005])
                    weights_hover.append(hover_text)
                    
                fig_ef.add_trace(go.Scatter(
                    x=frontier['volatilities']*100, y=frontier['returns']*100,
                    mode='markers',
                    marker=dict(size=4, color=metric_color, colorscale='Viridis', showscale=True, 
                                colorbar=dict(title=c_title)),
                    name='Espace des Poids (Markowitz)',
                    customdata=weights_hover,
                    hovertemplate="<b>Volatilité:</b> %{x:.2f}%<br><b>Rendement:</b> %{y:.2f}%<br><b>" + c_title + ":</b> %{marker.color:.2f}<br><br><b>Allocation:</b><br>%{customdata}<extra></extra>"
                ))
                
                # Prepare hover texts
                def format_hover(w_arr):
                    return "<br>".join([f"{a}: {w_arr[i]*100:.1f}%" for i, a in enumerate(ret_q_opt.columns) if w_arr[i] > 0.005])
                    
                # Min Volatility
                fig_ef.add_trace(go.Scatter(
                    x=[min_vol*100], y=[min_mean*100],
                    mode='markers', marker=dict(size=14, color='#00e5ff', symbol='star'),
                    name='Minimum Volatility',
                    customdata=[format_hover(min_w_array)],
                    hovertemplate="<b>%{y:.2f}%</b> Rendement, %{x:.2f}% Volatilité<br><br><b>Allocation:</b><br>%{customdata}<extra></extra>"
                ))
                
                # Optimal
                fig_ef.add_trace(go.Scatter(
                    x=[opt_vol*100], y=[opt_mean*100],
                    mode='markers', marker=dict(size=16, color='#ff00ff', symbol='diamond'),
                    name=f'Optimal ({selected_opt_metric.capitalize()})',
                    customdata=[format_hover(opt_w_array)],
                    hovertemplate="<b>%{y:.2f}%</b> Rendement, %{x:.2f}% Volatilité<br><br><b>Allocation:</b><br>%{customdata}<extra></extra>"
                ))

                # Red Star: Official Backtest Allocation
                if bt_mean is not None and bt_vol is not None:
                    fig_ef.add_trace(go.Scatter(
                        x=[bt_vol*100], y=[bt_mean*100],
                        mode='markers', marker=dict(size=18, color='red', symbol='star-diamond', line=dict(width=2, color='white')),
                        name='Configuration Officielle (Backtest)',
                        customdata=[format_hover(backtest_w_array)],
                        hovertemplate="<b>%{y:.2f}%</b> Rendement, %{x:.2f}% Volatilité<br><br><b>ALLOCATION OFFICIELLE:</b><br>%{customdata}<extra></extra>"
                    ))
                
                fig_ef.update_layout(
                    title=f"Frontière Efficiente — {QUADRANT_NAMES[selected_q]}",
                    xaxis_title="Volatilité Annualisée (%)",
                    yaxis_title="Rendement Annualisé (%)",
                    legend=dict(yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation="h")
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)
                
                st.markdown("*Note : Dans un souci de simplicité et afin d'éviter un biais de sur-optimisation (overfitting), l'allocation du modèle officiel est arrondie à des chiffres ronds, ce qui explique son léger décalage assumé avec le point optimal théorique de Markowitz.*")
                
                # Composition du Portefeuille Optimal
                st.markdown(f"**Composition du Portefeuille Optimal ({selected_opt_metric.capitalize()}) :**")
                weights_s = pd.Series(opt_w).sort_values(ascending=False)
                # Strict 5% display filter
                weights_s = weights_s[weights_s >= 0.049].round(4) * 100
                st.dataframe(pd.DataFrame({'Allocation (%)': weights_s.round(2)}).T)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Stratégie d'Allocation", expanded=False):
                    st.markdown(
                        "L'allocation de ce portefeuille pivote dynamiquement selon les quatres régimes identifiés. "
                        "Les heatmaps ci-dessus comparent les Performances obtenues selon les prédictions du modèle face aux données réelles de marché.\n\n"
                        "Les quadrants sont définis par deux indicateurs clés (proxies) :\n\n"
                        "- **Axe Croissance :** High Yield Bond Spread (le risque de crédit comme proxy de la croissance).\n"
                        "- **Axe Inflation :** 10Y Breakeven Inflation Rate (les anticipations d'inflation du marché obligataire).\n\n"
                        "**Détail de la Répartition par Régime (Issue du Modèle d'Optimisation Dynamique Z-Score) :**"
                    )
                    
                    # Extract base weights for each quadrant from backtest output
                    q_weights = {}
                    if data.get('backtest') is not None:
                        df_b = data['backtest']
                        weight_cols = [c for c in df_b.columns if c.endswith('_base_weight') and '_hc_' not in c]
                        for q in [1, 2, 3, 4]:
                            df_q = df_b[df_b['smooth_quadrant'] == q]
                            if not df_q.empty:
                                # Take the mean of base_weights (which are constant per quadrant) to get the exact optimizer output
                                w_q = df_q[weight_cols].median()
                                w_q = w_q[w_q > 0.005].sort_values(ascending=False) * 100
                                q_weights[q] = w_q
            
                    c1_alloc, c2_alloc, c3_alloc, c4_alloc = st.columns(4)
            
                    with c1_alloc:
                        st.markdown("**Q1 | Croissance Saine**\n*Expansion, risque récompensé.*")
                        if 1 in q_weights:
                            for idx, val in q_weights[1].items():
                                asset_name = idx.replace('_base_weight', '').replace('_weight', '')
                                st.markdown(f"- {val:.1f}% {asset_name}")
            
                    with c2_alloc:
                        st.markdown("**Q2 | Inflation**\n*Pricing power et matières premières.*")
                        if 2 in q_weights:
                            for idx, val in q_weights[2].items():
                                asset_name = idx.replace('_base_weight', '').replace('_weight', '')
                                st.markdown(f"- {val:.1f}% {asset_name}")
            
                    with c3_alloc:
                        st.markdown("**Q3 | Stagflation**\n*Protection contre baisse et hausse des prix.*")
                        if 3 in q_weights:
                            for idx, val in q_weights[3].items():
                                asset_name = idx.replace('_base_weight', '').replace('_weight', '')
                                st.markdown(f"- {val:.1f}% {asset_name}")
            
                    with c4_alloc:
                        st.markdown("**Q4 | Crash Déflationniste**\n*Priorité à la sécurité et décorrélation.*")
                        if 4 in q_weights:
                            for idx, val in q_weights[4].items():
                                asset_name = idx.replace('_base_weight', '').replace('_weight', '')
                                st.markdown(f"- {val:.1f}% {asset_name}")
            
                    st.info(
                        " **Overlay Risk-Off (Filtre de Tendance) :** "
                        "En complément de cette allocation socle par régime macro, une protection systématique de suivi de tendance (**MA 200 jours**) est active. "
                        "Si le S&P 500, le NASDAQ 100 ou l'Or clôturent sous leur moyenne mobile à 200 jours pendant 5 jours consécutifs, leur pondération est instantanément coupée à 0% et réallouée en bons du Trésor à 10 ans (Treasuries) jusqu'à ce que la tendance soit reprise."
                    )
        except Exception as e:
            st.error(f"Erreur lors du calcul de la Frontière Efficiente : {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
    st.divider()

    # =========================================================
    # SECTION 3: SIGNAUX DE CONVICTION STRUCTURELLE (Intensité du Régime)
    # =========================================================
    st.subheader("Signaux de Conviction Structurelle")

    col_ui1, col_ui2 = st.columns([2, 1])
    with col_ui1:
        selected_metric = st.selectbox(
            "Choisissez la Métrique d'Évaluation :",
            ["Sharpe Ratio", "Sortino Ratio", "Win Rate (% de jours positifs)"],
            index=0,
            help="Sharpe: Rendement vs Volatilité Globale | Sortino: Rendement vs Volatilité à la Baisse | Win Rate: % de Jours Positifs"
        )
    with col_ui2:
        inverse_fx = st.toggle("Inverser paires Forex (ex: EUR/USD)", value=False)

    st.markdown(
        """
        L'idée de cette section est d'analyser la performance du modèle dans la gestion de portefeuille uniquement lors des jours où le modèle a une **forte conviction** sur le régime macro-économique. 
        Ce niveau de conviction traduit un **alignement fort des différents indicateurs** économiques en faveur d'un quadrant spécifique. 
        Le modèle sort ainsi de sa zone centrale ( a partir de 65% de score de confiance sur les 2 axes ) pour valider un régime clair et réduire le risque de faux signaux. 
        la strategie HC *2 applique donc un effet de levier 2x dans les entrées en positions. On sort de position lorsque le modèle sort de la zone de certitude.
        """
    )

    if data.get('quadrants') is not None and 'PROB_GROWTH_EMA' in data['quadrants'].columns and 'PROB_INFLATION_EMA' in data['quadrants'].columns:
        # High Conviction = far from the 0.5 center (uncertainty)
        high_conviction_mask = (abs(data['quadrants']['PROB_GROWTH_EMA'] - 0.5) >= 0.15) & \
                               (abs(data['quadrants']['PROB_INFLATION_EMA'] - 0.5) >= 0.15)
        
        df_high_conviction = data['quadrants'][high_conviction_mask]
        
        if not df_high_conviction.empty:
            st.write(f"Nombre de jours de Haute Conviction : **{len(df_high_conviction)}** jours sur {len(data['quadrants'])}.")
            
            if data.get('backtest') is not None:
                bt_filtered = data['backtest'][data['backtest']['date'].isin(df_high_conviction['date'])]
                
                hc_col1, hc_col2 = st.columns(2)
                
                with hc_col1:
                    st.markdown("**Actions/ETF — HAUTE CONVICTION**")
                    # Remove BTC for High Conviction Heatmap
                    assets_no_btc = data.get('daily_assets').drop(columns=['BTC_USD'], errors='ignore') if data.get('daily_assets') is not None else None
                    scores_hc, conf_hc = get_dynamic_heatmap_data(assets_no_btc, bt_filtered, 'smooth_quadrant', selected_metric)
                    if not _render_heatmap(scores_hc, conf_hc, None, "Actions/ETF — HAUTE CONVICTION", f"Signaux profonds (éloignement des axes)", selected_metric, height=350):
                        st.info("Données insuffisantes pour le heatmap HC Actions.")
              
                
                with hc_col2:
                    st.markdown("**Forex — HAUTE CONVICTION**")
                    scores_hcfx, conf_hcfx = get_dynamic_heatmap_data(data.get('daily_forex'), bt_filtered, 'smooth_quadrant', selected_metric, inverse_fx=inverse_fx)
                    if not _render_heatmap(scores_hcfx, conf_hcfx, None, "Forex — HAUTE CONVICTION", f"Signaux nets (>65% proba)", selected_metric, height=300):
                        st.info("Données insuffisantes pour le heatmap HC Forex.")
                    
        else:
            st.warning("Aucun jour de haute conviction détecté.")
    else:
        st.warning("Probabilités non disponibles.")


    st.markdown(
        "On observe qu'en Q1 profond renvoi effectivement vers un marché fortement Risk-ON."
    )

    with st.expander("Stratégie sur Signaux à Haute Conviction", expanded=True):
        if df_bt is not None and 'hc_wealth' in df_bt.columns:
            fig_hc = go.Figure()
            
            def _norm_hc(series, base=1000.0):
                s = series.dropna()
                if len(s) == 0: return series
                return s / s.iloc[0] * base

            fig_hc.add_trace(go.Scatter(
                x=df_bt['date'], y=_norm_hc(df_bt['SP500_wealth']),
                name='SP500 (Benchmark)', line=dict(color='orange')
            ))
            
            fig_hc.add_trace(go.Scatter(
                x=df_bt['date'], y=_norm_hc(df_bt['wealth']),
                name='Stratégie Modèle Complet', line=dict(color='cyan', dash='dash')
            ))
            
            fig_hc.add_trace(go.Scatter(
                x=df_bt['date'], y=_norm_hc(df_bt['hc_wealth']),
                name='HC Haute Conviction', line=dict(color='#39FF14')
            ))

            if 'hc2x_wealth' in df_bt.columns:
                fig_hc.add_trace(go.Scatter(
                    x=df_bt['date'], y=_norm_hc(df_bt['hc2x_wealth']),
                    name='HC 2x Levier',
                    line=dict(color='#ff4444', width=2)
                ))
            
            fig_hc.update_layout(
                height=400, yaxis_title="Wealth indicée ($, base 1 000)", xaxis_title="Date",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_hc, use_container_width=True)
            
            hc_wealth_map = {
                "HC Haute Conviction": df_bt['hc_wealth'],
                "Stratégie Modèle Complet": df_bt['wealth'],
                "S&P 500 (Benchmark)": df_bt['SP500_wealth']
            }
            if 'hc2x_wealth' in df_bt.columns:
                hc_wealth_map["HC 2x Levier"] = df_bt['hc2x_wealth']
                
            def display_hc_compare_panel(key_id, default_selection_idx):
                choice = st.selectbox(
                    f"Sélecteur {key_id} :",
                    options=list(hc_wealth_map.keys()),
                    index=min(default_selection_idx, len(hc_wealth_map) - 1),
                    key=f"sel_hc_{key_id}"
                )
                w_series = hc_wealth_map[choice]
                tot_ret, cagr, sharpe, vol, max_dd = _calc_metrics(w_series)

                st.write(f"### {choice}")
                m_c1, m_c2 = st.columns(2)
                m_c1.metric("Total Return", f"{tot_ret * 100:.1f}%")
                m_c1.metric("Annual Return", f"{cagr * 100:.1f}%")
                m_c1.metric("Max Drawdown", f"{abs(max_dd) * 100:.1f}%")
                m_c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m_c2.metric("Annual Vol", f"{vol * 100:.1f}%")

            st.write("### Comparaison Stratégie HC")
            comp_hc1, comp_hc2 = st.columns(2)
            with comp_hc1:
                display_hc_compare_panel("A_HC", 0)
            with comp_hc2:
                display_hc_compare_panel("B_HC", 2)



