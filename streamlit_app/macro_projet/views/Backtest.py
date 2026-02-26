"""
Page 2: Backtest & Performance
Historical strategy performance, heatmaps by quadrant regime.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render(data):
    st.header("Performance Historique")

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

    # === Helper for heatmap rendering ===
    def _render_heatmap(perf_data, title, caption, height=350):
        if perf_data is not None and 'sharpe' in perf_data.columns:
            df = perf_data.copy()
            # Pivot using 'sharpe' instead of 'annual_return'
            pivot = df.pivot(index='asset', columns='quadrant', values='sharpe').fillna(0)
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,  # Sharpe is already a ratio, no need to multiply by 100
                x=[f'Q{int(c)}' for c in pivot.columns],
                y=pivot.index,
                colorscale='RdYlGn',
                text=[[f'{v:.2f}' for v in row] for row in pivot.values],  # Format as 2 decimals
                texttemplate='%{text}',
                textfont={"size": 10},
                zmid=0
            ))
            fig.update_layout(height=height, title=title)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)
            return True
        return False

    # =========================================================
    # SECTION 1: Performance avec Quadrants PREDITS (ML)
    # =========================================================
    st.subheader("Performance par Quadrant PREDIT (Modele ML)")
    st.markdown("*Ratio de Sharpe par actif en utilisant les quadrants assignes par le modele ML (predict_proba + EMA).*")

    # Heatmap 1: Actions ETF — Predicted
    perf_source = data.get('perf_smooth') if data.get('perf_smooth') is not None else data.get('perf')
    if not _render_heatmap(
        perf_source,
        "Actions/ETF — Sharpe Ratio (Quadrants Predits ML)",
        "Quadrants ML : Random Forest predict_proba + EMA span=5."
    ):
        st.info("Donnees de performance Actions non disponibles.")

    # Heatmap 2: Forex — Predicted
    if not _render_heatmap(
        data.get('forex_perf'),
        "Forex — Sharpe Ratio (Quadrants Predits ML)",
        "Paires Forex (USD/XXX et Inverses XXX/USD avec Carry) — quadrants ML."
    ):
        st.info("Donnees de performance Forex non disponibles.")

    st.divider()

    # =========================================================
    # SECTION 2: Performance avec Quadrants TARGET (Ground Truth)
    # =========================================================
    st.subheader("Performance par Quadrant TARGET (Ground Truth)")
    st.markdown("*Ratio de Sharpe si le modele etait 100% precis — base sur les quadrants définis par les targets (spread HY et breakeven inflation).*")

    # Heatmap 3: Actions ETF — Target
    if not _render_heatmap(
        data.get('perf_target'),
        "Actions/ETF — Sharpe Ratio (Quadrants Reels Target)",
        "Ground truth : quadrants définis par les targets fixés au modèle (spread HY et breakeven inflation)"
    ):
        st.info("Donnees de performance Target Actions non disponibles. Relancez le DAG.")

    # Heatmap 4: Forex — Target
    if not _render_heatmap(
        data.get('forex_perf_target'),
        "Forex — Sharpe Ratio (Quadrants Reels Target)",
        "Paires Forex — si le modele etait parfait (quadrants ground truth)."
    ):
        st.info("Donnees de performance Target Forex non disponibles. Relancez le DAG.")

    with st.expander("Stratégie d'Allocation ", expanded=False):
        st.markdown(
            "L'allocation de ce portefeuille pivote dynamiquement selon les quatres régimes identifiés. "
            "Les heatmaps ci-dessus comparent les Ratios de Sharpe obtenus selon les prédictions du modèle face aux données réelles de marché.\n\n"
            "Les quadrants sont définis par deux indicateurs clés (proxies) :\n\n"
            "- **Axe Croissance :** High Yield Bond Spread (le risque de crédit comme proxy de la croissance).\n"
            "- **Axe Inflation :** 10Y Breakeven Inflation Rate (les anticipations d'inflation du marché obligataire).\n\n"
            "📊 **Détail de la Répartition par Régime :**"
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
                "*Le régime où le pricing power des entreprises est crucial.*\n"
                "- 40% S&P 500 (Dominance des entreprises de qualité)\n"
                "- 30% NASDAQ_100 (Maintien d'une exposition Growth)\n"
                "- 30% SmallCAP (Sélection d'opportunités cycliques)"
            )

        with c3:
            st.markdown(
                "**Q3 | Stagflation (Défense Totale)**\n\n"
                "*Protection du capital contre la baisse de croissance et la hausse des prix.*\n"
                "- 40% Or (GOLD) (Valeur refuge ultime)\n"
                "- 30% Matières Premières (COMMODITIES) (Hedge direct contre l'inflation)\n"
                "- 30% Treasuries 10Y (Sécurité obligataire)"
            )

        with c4:
            st.markdown(
                "**Q4 | Crash Déflationniste (Le Bunker)**\n\n"
                "*Priorité absolue à la sécurité et à la décorrélation des stock market.*\n"
                "- 50% Treasuries 10Y (Profite de la baisse des taux directeurs)\n"
                "- 30% Or (GOLD) (Refuge contre la panique de marché)\n"
                "- 20% Obligations (Investment Grade) (Rendement sécurisé hors corporate risqué)"
            )
    st.divider()

    with st.expander("Focus sur le Marché des Changes (Forex)", expanded=False):
        st.markdown(
            "J'ai intégré une analyse spécifique au Forex pour explorer des opportunités de Carry Trading. "
            "Les Ratios de Sharpe affichés incluent les swaps d'intérêt journaliers.\n\n"
            "**Observations :**\n\n"
            "- Bien que le modèle identifie des disparités (ex: force du JPY ou de l'USD selon les quadrants), aucune stratégie systématique n'a été retenue pour le moment.\n"
            "- Le modèle évoluant principalement en Q2 et Q4 (avec Q1/Q3 comme phases de transition rapides), l'extraction d'un \"edge\" persistant sur le Forex reste un défi. Pour l'instant, je juge la fiabilité des classes d'actifs (Actions/Obligations) supérieure."
        )

    st.divider()
