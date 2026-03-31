import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render():
    import dca_strat.dca_utils as du
    st.title("DCA Investment Strategy")
    st.markdown("illustration et optimisation d'une stratégie de DCA (Dollar Cost Averaging). "
                "Ayant moi-même un petit capital, j'utilise le DCA pour une partie de mon portefeuille (40%) "
                "et je cherchais un moyen de l'optimiser pour améliorer mes rendements "
            )
    
    st.markdown("---")
    
    st.markdown(
        """
        <style>
        div[data-baseweb="popover"] ul {
            background-color: #0a0e27 !important;
        }
        div[data-baseweb="popover"] ul li {
            color: white !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #0a0e27 !important;
            color: white !important;
        }
        span[data-baseweb="tag"] {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col_params, col_results = st.columns([1, 3])
    
    with col_params:
        st.markdown("### Paramètres")
        ticker_options = {
            "S&P 500 (^GSPC)": "^GSPC",
            "Or (GC=F)": "GC=F",
            "Bitcoin (BTC-USD)": "BTC-USD"
        }
        asset_choice = st.selectbox("Actif à tester", list(ticker_options.keys()), key="dca_asset")
        selected_ticker = ticker_options[asset_choice]
        
        st.markdown("**Investissement fixe :** 100 € par 2 semaines")
        invest_per_trade = 100.0
        rolling_window = st.slider("Fenêtre glissante (jours, pour médiane/écart-type)", min_value=90, max_value=730, value=365, step=5, key="dca_window")
        
        # Hardcode dates for now, could be dynamic
        start_date = "2015-01-01"
        end_date = "2025-01-01"
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("Lancer le Backtest", use_container_width=True, type="primary", key="dca_run_sim")
        
    with col_results:
        with st.spinner("Téléchargement des données et backtesting..."):
            res_classic, res_smart, res_cost = du.fetch_and_run_dca(
                ticker=selected_ticker,
                start_date=start_date,
                end_date=end_date,
                invest_per_trade=invest_per_trade,
                rolling_window=rolling_window
            )
            
        if res_classic is None:
            st.error(f"Impossible de télécharger les données pour {selected_ticker}.")
        else:
            dca_classic, sh_classic = res_classic
            dca_smart, sh_smart = res_smart
            dca_cost, sh_cost = res_cost
            
            st.markdown("### Résultats du Backtest (2015 - 2025)")
            st.markdown(f"**Comparaison des stratégies sur : {asset_choice}**")
            
            # Extract final metrics
            invested_c = dca_classic["Invested"].iloc[-1]
            invested_s = dca_smart["BuyBudget"].sum()
            invested_cb = dca_cost["BuyBudget"].sum()

            end_eq_c  = dca_classic["Equity"].iloc[-1]
            end_eq_s  = dca_smart["Equity"].iloc[-1]
            end_eq_cb = dca_cost["Equity"].iloc[-1]
            
            # Display metrics in columns
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### DCA Classique")
                st.metric(" Capital Final", f"{end_eq_c:,.2f} €", f"{(end_eq_c/invested_c - 1)*100:,.2f}%" if invested_c>0 else "0%")
                st.caption(f"Investi total: {invested_c:,.2f} €")

            with c2:
                st.markdown("##### Smart DCA (Z-Score + Cash)")
                st.metric(" Capital Final", f"{end_eq_s:,.2f} €", f"{(end_eq_s/invested_s - 1)*100:,.2f}%" if invested_s>0 else "0%")
                st.caption(f"Budget total: {invested_s:,.2f} €")

            with c3:
                st.markdown("##### Cost Basis DCA + Z-socre")
                st.metric(" Capital Final", f"{end_eq_cb:,.2f} €", f"{(end_eq_cb/invested_cb - 1)*100:,.2f}%" if invested_cb>0 else "0%")
                st.caption(f"Budget total: {invested_cb:,.2f} €")
            
            # Slice data for plotting to hide the rolling-window warm-up period
            plot_c = dca_classic.loc[start_date:]
            plot_s = dca_smart.loc[start_date:]
            plot_cb = dca_cost.loc[start_date:]

            # sPlot the results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(plot_c.index, plot_c["Equity"], label=f"Classique", color="#00d4ff", linewidth=1.5)
            ax.plot(plot_s.index, plot_s["Equity"], label=f"Smart Z+Cash", color="#00ff00", linewidth=1.5)
            ax.plot(plot_cb.index, plot_cb["Equity"], label=f"CostBasis+Z", color="#ff00ff", linewidth=1.5)
            ax.plot(plot_c.index, plot_c["Invested"], "--", label="Investissement", color="#ff4b4b", linewidth=1.5)
            
            # Theme adjustments for Dark Mode
            fig.patch.set_facecolor('#0a0e27')
            ax.set_facecolor('#0a0e27')
            for spine in ['bottom', 'top', 'left', 'right']:
                ax.spines[spine].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Équité / Valeur (€)')
            ax.set_title(f"Évolution de l'équité pour {asset_choice}", color='white')
            ax.legend(facecolor='#0a0e27', edgecolor='white', labelcolor='white')
            
            st.pyplot(fig)
            
            st.info("💡 **Explication des méthodes :**\n\n"
                    "- **DCA Classique** : Achat d'un montant fixe à intervalles réguliers, peu importe les conditions.\n\n"
                    "- **Smart DCA (Z-Score + Cash)** : Modulation du montant de l'achat en fonction de l'écart au prix médian "
                    "(Z-Score). On achète moins quand le marché est surévalué et on met en réserve le cash non utilisé "
                    "pour acheter davantage lors de baisses (soldes).\n\n"
                    "- **Cost Basis DCA + Z** : Variante qui module l'allocation non seulement avec le Z-Score, mais "
                    "aussi en comparant le prix actuel au prix de revient unitaire (Cost Basis). L'objectif est d'accélérer "
                    "l'investissement quand on est en perte par rapport à sa propre moyenne d'achat. \n\n"
                    "- On peu voir que ces methodes ne sont finalement pas plus profitable qu'un DCA classique sur le long terme."
                    "Il est toute fois évident que cela reste moins efficace qu'un investissement en une seul fois.")
