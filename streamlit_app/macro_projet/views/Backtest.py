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


def get_dynamic_heatmap_data(daily_df, date_quadrant_df, quadrant_col, metric):
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
            # We must compute daily returns (pct_change) from the absolute prices
            # We do this on the whole merged df to maintain day-to-day sequencing, then filter by quadrant
            ret_series = merged[a].pct_change().loc[df_q.index].dropna()
            
            # Remove infinity values that might result from division by very small numbers
            ret_series = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            score = compute_single_metric(ret_series, metric)
            conf = bootstrap_confidence(ret_series, metric, score)
            scores_matrix.at[a, q] = score
            conf_matrix.at[a, q] = conf
            
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
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['SP500_wealth'], name='SP500', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['GOLD_wealth'], name='Gold', line=dict(color='gold')))
        fig.update_layout(height=400, yaxis_title="Wealth ($)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Donnees backtest non disponibles")

    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    if data['stats'] is not None:
        stats_dict = data['stats'].iloc[0].to_dict() if len(data['stats']) > 0 else {}
        m1.metric("Total Return", f"{stats_dict.get('total_return', 0) * 100:.1f}%")
        m2.metric("Max Drawdown", f"{stats_dict.get('strategy_max_drawdown', 0) * 100:.1f}%")
        m3.metric("Sharpe Ratio", f"{stats_dict.get('strategy_sharpe_annual', 0):.2f}")
        m4.metric("Annual Vol", f"{stats_dict.get('strategy_vol_annual', 0) * 100:.1f}%")

    st.divider()

    # =========================================================
    # SECTION DYNAMIQUE (ROBUSTESSE)
    # =========================================================
    st.subheader("Performance par Quadrant et Test de Robustesse (Bootstrap)")
    
    selected_metric = st.selectbox(
        "Choisissez la Métrique d'Évaluation :",
        ["Sharpe Ratio", "Sortino Ratio", "Win Rate (% de jours positifs)"],
        index=0,
        help="Sharpe: Rendement vs Volatilité Globale | Sortino: Rendement vs Volatilité à la Baisse | Win Rate: % de Jours Positifs"
    )

    # 2 colonnes pour un affichage plus dense
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Quadrants PREDITS (Modèle ML)**")
        # Actions
        scores_acc, conf_acc = get_dynamic_heatmap_data(data.get('daily_assets'), data.get('backtest'), 'smooth_quadrant', selected_metric)
        perf_source = data.get('perf_smooth') if data.get('perf_smooth') is not None else data.get('perf')
        if not _render_heatmap(scores_acc, conf_acc, perf_source, "Actions/ETF — PREDIT", "ML predict_proba + EMA", selected_metric, height=350):
            st.info("Donnees de perf Actions ML non dispo.")

        # Forex
        scores_fx, conf_fx = get_dynamic_heatmap_data(data.get('daily_forex'), data.get('backtest'), 'smooth_quadrant', selected_metric)
        if not _render_heatmap(scores_fx, conf_fx, data.get('forex_perf'), "Forex — PREDIT", "Paires Forex", selected_metric, height=300):
            st.info("Donnees de perf Forex ML non dispo.")

    with col2:
        st.markdown("**Quadrants TARGET (Ground Truth)**")
        # Actions
        scores_tgt, conf_tgt = get_dynamic_heatmap_data(data.get('daily_assets'), data.get('quadrants'), 'target_quadrant', selected_metric)
        if not _render_heatmap(scores_tgt, conf_tgt, data.get('perf_target'), "Actions/ETF — TARGET", "Quadrants parfaits selon les targets", selected_metric, height=350):
            st.info("Donnees de perf Actions Target non dispo.")

        # Forex
        scores_fxtgt, conf_fxtgt = get_dynamic_heatmap_data(data.get('daily_forex'), data.get('quadrants'), 'target_quadrant', selected_metric)
        if not _render_heatmap(scores_fxtgt, conf_fxtgt, data.get('forex_perf_target'), "Forex — TARGET", "Paires Forex cibles", selected_metric, height=300):
            st.info("Donnees de perf Forex Target non dispo.")

    st.info("ℹ **Score de Confiance (Conf: X%) :** Ce score indique le pourcentage des échantillons qui ont la meme polarité (mesure l'homogénéité de la distribution). ")

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

    with st.expander("Focus sur le Marché des Changes (Forex)", expanded=False):
        st.markdown(
            "J'ai intégré une analyse spécifique au Forex pour explorer des opportunités de Carry Trading. "
            "Les Ratios de Sharpe affichés incluent les swaps d'intérêt journaliers.\n\n"
            "**Observations :**\n\n"
            "- Bien que le modèle identifie des disparités (ex: force du JPY ou de l'USD selon les quadrants), aucune stratégie systématique n'a été retenue pour le moment.\n"
            "- Le modèle évoluant principalement en Q2 et Q4 (avec Q1/Q3 comme phases de transition rapides), l'extraction d'un \"edge\" persistant sur le Forex reste un défi. Pour l'instant, je juge la fiabilité des classes d'actifs (Actions/Obligations) supérieure."
        )

    st.divider()
