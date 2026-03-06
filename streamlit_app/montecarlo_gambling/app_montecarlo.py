import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the roulette strategy
def roulette(mise_initial, portfolio_depart, nb_tirage, objectif_gain_pct=0.1):        
    portfolio = portfolio_depart
    mise = mise_initial  
    historique = [portfolio]
    partie_win = 0  

    for i in range(nb_tirage): 
        tirage = random.randint(0, 37)
        if tirage % 2 == 0 and tirage not in [0, 37]:
            win = mise 
            portfolio += win
            mise = mise_initial
        elif (tirage + 1) % 2 == 0 and tirage not in [0, 37]: 
            loose = mise
            portfolio -= loose
            mise = mise * 2 
        elif tirage == 0 or tirage == 37:
            loose = mise
            portfolio -= loose
            mise = mise * 2      

        historique.append(portfolio) 
        if portfolio <= 0 or portfolio < mise: 
            break 
        if portfolio >= portfolio_depart * (1 + objectif_gain_pct): 
            partie_win = 1  
            break 

    return portfolio, historique, partie_win

@st.cache_data(show_spinner=False)
def run_monte_carlo(nb_simul, mise_initial, portfolio_depart, nb_tirage, objectif_gain_pct):
    resultats = []
    simul = []
    victoires = []
    for i in range(nb_simul):
        portfolio, historique, partie_win = roulette(mise_initial=mise_initial, portfolio_depart=portfolio_depart, nb_tirage=nb_tirage, objectif_gain_pct=objectif_gain_pct)
        resultats.append(portfolio)
        simul.append(historique)
        victoires.append(partie_win)    
    return np.array(resultats), simul, np.array(victoires)

@st.cache_data(show_spinner=False)
def run_optimization():
    # Paramètres à tester (les mêmes que dans le notebook)
    portfolios_test = [1000, 50000, 100000, 1000000]
    mises_test = [1, 10, 20]
    nb_tirage_test = [10000]
    resultats_vbt = []

    for portfolio_init in portfolios_test:
        for mise_init in mises_test:
            for nb_tirage_init in nb_tirage_test: 
                simulations = []
                portfolio_finaux = []
                for i in range(1000):
                    portfolio_final, historique, partie_win = roulette(mise_initial=mise_init, portfolio_depart=portfolio_init, nb_tirage=nb_tirage_init)
                    simulations.append(historique)
                    portfolio_finaux.append(portfolio_final)

                capital_finaux = np.array(portfolio_finaux) 
                proba_gain = (capital_finaux > portfolio_init).sum() / len(capital_finaux) * 100
                return_moyen = ((capital_finaux.mean() - portfolio_init) / portfolio_init) * 100
                max_capital = capital_finaux.max()
                min_capital = capital_finaux.min()
                all_returns = []

                for hist in simulations:   
                    hist_array = np.array(hist)
                    returns = (hist_array[1:] / hist_array[:-1]) - 1
                    all_returns.extend(returns)

                ecart_type = np.std(all_returns) * 100
                rendement_moyen_periodique = np.mean(all_returns) * 100
                
                if ecart_type > 0:
                    sharpe_ratio = rendement_moyen_periodique / ecart_type
                else:
                    sharpe_ratio = 0

                resultats_vbt.append({
                    'Portfolio Initial': portfolio_init,
                    'Mise Initiale': mise_init,
                    'nb_tirage': nb_tirage_init,
                    'Capital Final Moyen': capital_finaux.mean(),
                    'Return Moyen (%)': return_moyen,
                    'Écart-Type (%)': ecart_type,
                    'Sharpe Ratio': sharpe_ratio,
                    'Capital Max': max_capital,
                    'capital Min': min_capital,
                    '% Gain': proba_gain,
                    'Return Max (%)': ((max_capital - portfolio_init) / portfolio_init) * 100, 
                    'return Min (%)': ((min_capital - portfolio_init) / portfolio_init) * 100
                })
    return pd.DataFrame(resultats_vbt)

def render():
    st.title("🎲 Monte Carlo Gambling Simulator")
    st.markdown("Simulation d'une ** Martingale** à la roulette via des simulations de Monte Carlo, afin de démontrer l'impossibilité mathématique de battre le casino de manière consistante.")
    
    st.markdown("---")
    
    col_params, col_results = st.columns([1, 3])
    
    with col_params:
        st.markdown("### Paramètres")
        portfolio_depart = st.number_input("Capital de départ (Stop Loss)", min_value=100, max_value=1000000, value=10000, step=100, key="mc_port")
        mise_initial = st.number_input("Mise initiale", min_value=1, max_value=1000, value=5, step=1, key="mc_mise")
        nb_tirage = st.number_input("Nombre de tirages max", min_value=10, max_value=10000, value=1000, step=10, key="mc_tirage")
        nb_simul = st.slider("Simulations (n)", min_value=100, max_value=2000, value=1000, step=100, key="mc_simul")
        objectif_gain_pct = st.number_input("Objectif de gain (%)", min_value=1, max_value=1000, value=10, step=1, key="mc_obj")
        
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("Lancer la simulation ", use_container_width=True, type="primary", key="mc_run_sim")
        
    with col_results:
        with st.spinner("Calcul des trajectoires de Monte Carlo en cours..."):
            resultats, simul, victoires = run_monte_carlo(nb_simul, mise_initial, portfolio_depart, nb_tirage, objectif_gain_pct / 100.0)
            
        st.markdown("### Résultats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moyenne Finale", f"{resultats.mean():.2f} €", f"{(resultats.mean() - portfolio_depart)/portfolio_depart * 100:.2f} %")
        c2.metric("Médiane", f"{np.median(resultats):.2f} €")
        c3.metric("% de Gain Global", f"{(resultats > portfolio_depart).sum() / len(resultats) * 100:.1f} %")
        c4.metric(f"% Atteint l'Objectif (+{objectif_gain_pct}%)", f"{victoires.sum() / len(victoires) * 100:.1f} %")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        simul_to_plot = simul[:min(1000, len(simul))]
        for hist in simul_to_plot:
            ax.plot(hist, linewidth=1, color='#00d4ff', alpha=0.1)
            
        ax.axhline(y=portfolio_depart, color='#ff4b4b', linestyle='--', linewidth=1.5, label='Capital initial')
        ax.axhline(y=(1 + objectif_gain_pct / 100.0) * portfolio_depart, color='#00ff00', linestyle='-', linewidth=1.5, label=f'Limite de gain (+{objectif_gain_pct}%)')
        
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#0a0e27')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        ax.set_xlabel('Nombre de tirages')
        ax.set_ylabel('Portfolio (€)')
        ax.set_title(f"Évolution de {len(simul_to_plot)} trajectoires de joueurs", color='white')
        ax.legend(facecolor='#0a0e27', edgecolor='white', labelcolor='white')
        
        st.pyplot(fig)
        
        st.info("💡Même avec un taux de réussite élevé, les rares séquences de pertes consécutives viennent détruire le capital. D'où un retour moyen négatif.")

    st.markdown("---")
    st.markdown("A des fins d'entrainement, j'ai voulu apprendre a utiliser VectorBT pour optimiser les paramètres de la Martingale.")
    st.subheader(" Optimisation des paramètres (VectorBT / Heatmap)")
    st.markdown("Existe-t-il statistiquement un ratio `Portefeuille Initial / Mise Initiale` qui permettrait d'être rentable sur la durée ? Nous calculons ici **12 000 backtests de trajectoires** sur 10 000 tirages.")
    
    if st.button("Lancer la recherche VectorBT (Prendra quelques secondes)", key="mc_run_vbt"):  
        with st.spinner("Exécution de l'optimisation en arrière-plan..."):
            df_resultats = run_optimization()
        
        nb_tirage_test = [10000]
        
        fig_heatmap = make_subplots(
            rows=1, cols=len(nb_tirage_test),
            subplot_titles=[f'Return Moyen (%) - {nb} tirages' for nb in nb_tirage_test],
            horizontal_spacing=0.1
        )
        
        for idx, nb_tirage in enumerate(nb_tirage_test):
            df_subset = df_resultats[df_resultats['nb_tirage'] == nb_tirage]

            pivot_data = df_subset.pivot_table(
                values='Return Moyen (%)',
                index='Portfolio Initial',
                columns='Mise Initiale',
                aggfunc='mean'
            )

            fig_heatmap.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,  
                    hovertemplate='Portfolio: %{y:,.0f}€<br>Mise: %{x}€<br>Return: %{z:.2f}%<extra></extra>',
                    colorscale='RdYlGn',
                    text=pivot_data.values,
                    texttemplate='%{text:.1f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="Return (%)", x=1.02 + idx*0.3),
                    showscale=True
                ),
                row=1, col=idx+1
            )

            fig_heatmap.update_xaxes(title_text="Mise Initiale (€)", row=1, col=idx+1)

            portfolio_values = sorted(pivot_data.index)
            fig_heatmap.update_yaxes(
                title_text="Portfolio Initial (€)",
                type="log",
                row=1, col=idx+1,
                tickmode='array',
                tickvals=portfolio_values,  
                ticktext=[f'{val:,.0f}' if val < 1000 else f'{val/1000:.0f}K' if val < 1000000 else f'{val/1000000:.1f}M' 
                         for val in portfolio_values]  
            )

        fig_heatmap.update_layout(
            height=450,
            title_text="Heatmap du Return Moyen en (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, key="mc_heatmap")
        
        with st.expander("Détail des résultats VectorBT", expanded=False):
            st.dataframe(df_resultats.style.highlight_max(subset=['Return Moyen (%)', 'Sharpe Ratio', '% Gain'], axis=0), use_container_width=True)
        
        best_sharpe = df_resultats.loc[df_resultats['Sharpe Ratio'].idxmax()]
        worst_return = df_resultats.loc[df_resultats['Return Moyen (%)'].idxmin()]

        c_best, c_worst = st.columns(2)
        with c_best:
            st.success(f"**Meilleure combinaison** :\n* Portfolio: **{best_sharpe['Portfolio Initial']:.0f}€**\n* Mise: **{best_sharpe['Mise Initiale']:.0f}€**\n* Return: **{best_sharpe['Return Moyen (%)']:.2f}%**\n* Ratio Sharpe: **{best_sharpe['Sharpe Ratio']:.3f}**")
        with c_worst:
            st.error(f" **Pire combinaison** :\n* Portfolio: **{worst_return['Portfolio Initial']:.0f}€**\n* Mise: **{worst_return['Mise Initiale']:.0f}€**\n* Return: **{worst_return['Return Moyen (%)']:.2f}%**\n* Ratio Sharpe: **{worst_return['Sharpe Ratio']:.3f}**")
