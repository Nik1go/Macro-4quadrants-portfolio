"""
Page 2: Backtest & Performance
Historical strategy performance, heatmaps by quadrant regime.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_loader import QUADRANT_NAMES, QUADRANT_COLORS


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

    # === Smooth Quadrant Distribution (Historique Complet) ===
    st.subheader("Repartition Totale des Quadrants (Historique Complet)")
    if data['backtest'] is not None and 'smooth_quadrant' in data['backtest'].columns:
        smooth_q_counts = data['backtest']['smooth_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        total_days = smooth_q_counts.sum()

        fig_smooth = go.Figure(data=[go.Bar(
            x=[f"Q{i} — {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=smooth_q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=[f"{v} jours ({v / total_days * 100:.1f}%)" for v in smooth_q_counts.values],
            textposition='auto',
        )])

        fig_smooth.update_layout(
            title=f"Repartition des Quadrants Lisses — {total_days} jours de trading",
            yaxis_title="Nombre de Jours",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_smooth, use_container_width=True)

        dominant_smooth_q = smooth_q_counts.idxmax()
        start_date = data['backtest']['date'].min().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        end_date = data['backtest']['date'].max().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        st.info(
            f"Periode: **{start_date}** → **{end_date}** | "
            f"Regime dominant (lisse): **Q{dominant_smooth_q} — {QUADRANT_NAMES.get(dominant_smooth_q)}** "
            f"({smooth_q_counts.max() / total_days * 100:.1f}%)"
        )
    else:
        st.warning("Donnees smooth_quadrant non disponibles. Lancez le backtest pour generer ces donnees.")

    st.divider()

    # === Strategy vs Benchmark ===
    st.subheader("Strategie vs Benchmark")
    if data['backtest'] is not None:
        df_bt = data['backtest']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['wealth'], name='Strategy', line=dict(color='cyan')))
        if 'hc_wealth' in df_bt.columns:
            fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['hc_wealth'], name='Strategy Haute Conviction', line=dict(color='#39FF14', dash='dot')))
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['SP500_wealth'], name='SP500', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['GOLD_wealth'], name='Gold', line=dict(color='gold')))
        fig.update_layout(height=400, yaxis_title="Wealth ($)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Donnees backtest non disponibles")

    # Key Metrics
    if data['stats'] is not None:
        stats_dict = data['stats'].iloc[0].to_dict() if len(data['stats']) > 0 else {}
        
        st.markdown("#####  Stratégie d'Allocation (Tous régimes)")
        st.caption("On définit une allocation systématique pour chaque quadrant, le portefeuille est toujours investi.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{stats_dict.get('total_return', 0) * 100:.1f}%")
        m2.metric("Max Drawdown", f"{stats_dict.get('strategy_max_drawdown', 0) * 100:.1f}%")
        m3.metric("Sharpe Ratio", f"{stats_dict.get('strategy_sharpe_annual', 0):.2f}")
        m4.metric("Annual Vol", f"{stats_dict.get('strategy_vol_annual', 0) * 100:.1f}%")

        if 'hc_total_return' in stats_dict:
            st.markdown("##### Stratégie Trading (Haute Conviction)")
            st.caption("On trade uniquement les jours de forte conviction (>65%). Nous ne tradons pas le Q3 car le signal n'est pas détecté avec suffisamment de probabilité/stabilité.")
            m1b, m2b, m3b, m4b = st.columns(4)
            m1b.metric("Total Return", f"{stats_dict.get('hc_total_return', 0) * 100:.1f}%")
            m2b.metric("Max Drawdown", f"{stats_dict.get('strategy_hc_max_drawdown', 0) * 100:.1f}%")
            m3b.metric("Sharpe Ratio", f"{stats_dict.get('strategy_hc_sharpe_annual', 0):.2f}")
            m4b.metric("Annual Vol", f"{stats_dict.get('strategy_hc_vol_annual', 0) * 100:.1f}%")

        if data['backtest'] is not None and 'SP500_wealth' in data['backtest'].columns:
            sp500_wealth = data['backtest']['SP500_wealth']
            sp500_tot_ret = (sp500_wealth.iloc[-1] / sp500_wealth.iloc[0]) - 1 if len(sp500_wealth) > 0 else 0
            
            st.markdown("##### Benchmark (S&P 500 B&H)")
            st.caption("Acheter et conserver le marché américain (Buy & Hold).")
            m1c, m2c, m3c, m4c = st.columns(4)
            m1c.metric("Total Return", f"{sp500_tot_ret * 100:.1f}%")
            m2c.metric("Max Drawdown", f"{stats_dict.get('SP500_max_drawdown', 0) * 100:.1f}%")
            m3c.metric("Sharpe Ratio", f"{stats_dict.get('SP500_sharpe_annual', 0):.2f}")
            m4c.metric("Annual Vol", f"{stats_dict.get('SP500_vol_annual', 0) * 100:.1f}%")

    st.divider()

    # =========================================================
    # SECTION 1: ANALYSE DE PERFORMANCE DES ACTIFS (Actions/ETF)
    # =========================================================
    st.subheader("Analyse de Performance des Actifs (Actions/ETF)")
    
    selected_metric = st.selectbox(
        "Choisissez la Métrique d'Évaluation :",
        ["Sharpe Ratio", "Sortino Ratio", "Win Rate (% de jours positifs)"],
        index=0,
        help="Sharpe: Rendement vs Volatilité Globale | Sortino: Rendement vs Volatilité à la Baisse | Win Rate: % de Jours Positifs"
    )

    st.info("ℹ **Score de Confiance (Conf: X%) :** Ce score indique le pourcentage des échantillons qui ont la meme polarité (mesure l'homogénéité de la distribution). ")

    # Prepare daily_assets with BTC_USD only from 2024-03-01
    daily_assets_for_heatmap = None
    if data.get('daily_assets') is not None:
        daily_assets_for_heatmap = data['daily_assets'].copy()
        if 'BTC_USD' in daily_assets_for_heatmap.columns:
            daily_assets_for_heatmap['date'] = pd.to_datetime(daily_assets_for_heatmap['date'])
            daily_assets_for_heatmap.loc[daily_assets_for_heatmap['date'] < pd.Timestamp('2024-03-01'), 'BTC_USD'] = np.nan

    col1_assets, col2_assets = st.columns(2)

    with col1_assets:
        st.markdown("**Quadrants PREDITS (Modèle ML)**")
        scores_acc, conf_acc = get_dynamic_heatmap_data(daily_assets_for_heatmap, data.get('backtest'), 'smooth_quadrant', selected_metric)
        perf_source = data.get('perf_smooth') if data.get('perf_smooth') is not None else data.get('perf')
        if not _render_heatmap(scores_acc, conf_acc, perf_source, "Actions/ETF — PREDIT", "ML predict_proba + EMA", selected_metric, height=400):
            st.info("Données de performance Actions ML non disponibles.")

    with col2_assets:
        st.markdown("**Quadrants TARGET (Ground Truth)**")
        scores_tgt, conf_tgt = get_dynamic_heatmap_data(daily_assets_for_heatmap, data.get('quadrants'), 'target_quadrant', selected_metric)
        if not _render_heatmap(scores_tgt, conf_tgt, data.get('perf_target'), "Actions/ETF — TARGET", "Quadrants parfaits selon les targets", selected_metric, height=400):
            st.info("Données de performance Actions Target non disponibles.")

    st.caption("ℹ BTC (Bitcoin) est inclus uniquement depuis 03/2024, date de son institutionnalisation via les ETF spot.")

    st.divider()

    # =========================================================
    # SECTION 2: ANALYSE DE PERFORMANCE FOREX
    # =========================================================
    st.subheader("Analyse de Performance Forex")
    
    col_fx_ui1, col_fx_ui2 = st.columns([2, 1])
    with col_fx_ui2:
        inverse_fx = st.toggle("Inverser paires Forex (ex: EUR/USD)", value=False)
    
    col1_fx, col2_fx = st.columns(2)

    with col1_fx:
        st.markdown("**Quadrants PREDITS (Modèle ML)**")
        scores_fx, conf_fx = get_dynamic_heatmap_data(data.get('daily_forex'), data.get('backtest'), 'smooth_quadrant', selected_metric, inverse_fx=inverse_fx)
        if not _render_heatmap(scores_fx, conf_fx, data.get('forex_perf'), "Forex — PREDIT", "Paires Forex", selected_metric, height=330):
            st.info("Données de performance Forex ML non disponibles.")

    with col2_fx:
        st.markdown("**Quadrants TARGET (Ground Truth)**")
        scores_fxtgt, conf_fxtgt = get_dynamic_heatmap_data(data.get('daily_forex'), data.get('quadrants'), 'target_quadrant', selected_metric, inverse_fx=inverse_fx)
        if not _render_heatmap(scores_fxtgt, conf_fxtgt, data.get('forex_perf_target'), "Forex — TARGET", "Paires Forex cibles", selected_metric, height=330):
            st.info("Données de performance Forex Target non disponibles.")

    with st.expander("Focus sur le Marché des Changes (Forex)", expanded=False):
        st.markdown(
            "J'ai intégré une analyse spécifique au Forex pour explorer des opportunités de Carry Trading. "
            "Les Ratios de Sharpe affichés incluent les swaps d'intérêt journaliers.\n\n"
            "**Observations :**\n\n"
            "- Bien que le modèle identifie des disparités (ex: force du JPY ou de l'USD selon les quadrants), aucune stratégie systématique n'a été retenue pour le moment.\n"
            "- Le modèle évoluant principalement en Q2 et Q4 (avec Q1/Q3 comme phases de transition rapides), l'extraction d'un \"edge\" persistant sur le Forex reste un défi. Pour l'instant, je juge la fiabilité des classes d'actifs (Actions/Obligations) supérieure."
        )

    st.divider()

    # =========================================================
    # SECTION 3: SIGNAUX DE CONVICTION STRUCTURELLE (Intensité du Régime)
    # =========================================================
    st.subheader("Signaux de Conviction Structurelle")
    st.markdown(
        "Cette section affiche la performance uniquement lors des jours où le modèle a une **forte conviction** sur le régime macro-économique. "
        "Ce niveau de conviction traduit un **alignement fort des différents indicateurs** économiques en faveur d'un quadrant spécifique. "
        "Le modèle sort ainsi de sa zone d'incertitude centrale (proche de 50%) pour valider un régime clair et réduire le risque de faux signaux."
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

    with st.expander("Focus sur les Signaux à forte Conviction", expanded=False):
        st.markdown(
            "On observe qu'en Q1 profond renvoi effectivement vers un marché fortement Risk-ON. Il est intéressant de tenter une strategie autour de ces signaux."
        )

    st.divider()

    with st.expander("Stratégie d'Allocation", expanded=True):
        st.markdown(
            "L'allocation de ce portefeuille pivote dynamiquement selon les quatres régimes identifiés. "
            "Les heatmaps ci-dessus comparent les Performances obtenues selon les prédictions du modèle face aux données réelles de marché.\n\n"
            "Les quadrants sont définis par deux indicateurs clés (proxies) :\n\n"
            "- **Axe Croissance :** High Yield Bond Spread (le risque de crédit comme proxy de la croissance).\n"
            "- **Axe Inflation :** 10Y Breakeven Inflation Rate (les anticipations d'inflation du marché obligataire).\n\n"
            "**Détail de la Répartition par Régime :**"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                "**Q1 | Croissance Saine (Goldilocks)**\n\n"
                "*C'est la phase d'expansion où le risque est récompensé.*\n"
                "- 40% NASDAQ_100 (Moteur de performance technologique)\n"
                "- 30% SmallCAP (Bêta élevé pour maximiser la hausse)\n"
                "- 30% S&P 500 (Large caps pour la stabilité relative)"
            )

        with c2:
            st.markdown(
                "**Q2 | Inflation**\n\n"
                "*Le régime où le pricing power et la cyclicité jouent à plein.*\n"
                "- 40% S&P 500 (Dominance des grandes capitalisations)\n"
                "- 30% NASDAQ_100 (Exposition Growth réagissant à l'inflation)\n"
                "- 30% SmallCAP (Effet cyclique de rattrapage)"
            )

        with c3:
            st.markdown(
                "**Q3 | Stagflation (Défense Totale)**\n\n"
                "*Protection du capital contre la baisse de croissance et la hausse des prix.*\n"
                "- 40% Or (GOLD) (Valeur refuge ultime)\n"
                "- 30% Matières Premières (COMMODITIES) (Hedge direct contre l'inflation)\n"
                "- 30% Treasuries 10Y (Sécurité obligataire face aux actions)"
            )

        with c4:
            st.markdown(
                "**Q4 | Crash Déflationniste (Le Bunker)**\n\n"
                "*Priorité absolue à la sécurité et à la décorrélation des actions.*\n"
                "- 50% Treasuries 10Y (Profite de la baisse des taux directeurs)\n"
                "- 30% Or (GOLD) (Refuge ultime en cas de crise majeure)\n"
                "- 20% Obligations (IG) (Rendement sécurisé sur le crédit solide)"
            )

        st.info(
            " **Overlay Risk-Off (Filtre de Tendance) :** "
            "En complément de cette allocation socle par régime macro, une protection systématique de suivi de tendance (**MA 200 jours**) est active. "
            "Si le S&P 500, le NASDAQ 100 ou l'Or clôturent sous leur moyenne mobile à 200 jours pendant 5 jours consécutifs, leur pondération est instantanément coupée à 0% et réallouée en bons du Trésor à 10 ans (Treasuries) jusqu'à ce que la tendance soit reprise."
        )


    st.divider()
