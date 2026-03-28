import streamlit as st
import scipy.optimize as sco
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# --- CONFIGURATION ---
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"
CAPITAL_INITIAL = 10000
tickers = ['SPY', 'QQQ', 'GC=F', 'BTC-USD', '^FCHI']

# --- DATA FETCHING ---
@st.cache_data
def fetch_data(tickers):
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, interval=INTERVAL)['Adj Close']
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- 1. FONCTIONS MATHÉMATIQUES ---
def portfolio_performance(weights, returns, cov_matrix):
    # Le rendement du portefeuille = somme(poids * rendements moyens)
    # * 252 car il y a 252 jours de bourse dans une année (annualisation)
    mean_returns = returns.mean()
    p_ret = np.sum(mean_returns * weights) * 252
    
    # La volatilité = racine_carrée(Poids_Transposés * Matrice_Covariance * Poids)
    # C'est la formule classique de Markowitz !
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return p_ret, p_vol

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    # Calcul des performances temporaires
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    
    # Ratio de Sharpe classique : (Rendement - TauxSansRisque) / Risque (Volatilité)
    sharpe = (p_ret - risk_free_rate) / p_vol
    
    # ASTUCE SCIPY : SciPy sait seulement "minimiser" des choses.
    # Pour MAXIMISER le Sharpe, on demande à SciPy de MINIMISER le [-Sharpe].
    return -sharpe


# --- 2. FONCTION D'OPTIMISATION SCIPY ---
def optimize_portfolio(data): 
    # 2.0 : Préparation des données matricielles
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    num_assets = len(tickers)

    # 2.1 : Poids de départ (guess). On donne le même poids à tout le monde
    initial_weights = np.array(num_assets * [1. / num_assets])
    
    # 2.2 : Les Limites (Bounds). Aucun actif ne peut être shorté (0 à 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 2.3 : Les Contraintes. La somme totale des poids doit faire 100% (1.0).
    # 'type': 'eq' signifie que l'équation suivante devra être ÉGALE à 0.
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 2.4 : Let's go !! On dit à SciPy de trouver les meilleurs poids !
    # 'SLSQP' est l'algorithme parfait pour les problèmes bousculés par des limites et contraintes.
    result = sco.minimize(
        neg_sharpe_ratio,               # La fonction à embêter / minimiser
        initial_weights,                # D'où on part
        args=(returns, cov_matrix),     # Les paramètres à envoyer à notre fonction
        method='SLSQP',                 # L'algorithme
        bounds=bounds,                  # Nos limites (pas de short)
        constraints=constraints         # Notre contrainte (somme = 1)
    )
    
    return result, returns, cov_matrix


# --- 3. RENDU INTERFACE UTILISATEUR (UI) ---
def render():
    st.title("Portfolio Optimizer (SciPy)")
    st.markdown("L'objectif est d'optimiser la pondération de chaque indice pour construire le portefeuille le plus robuste possible (**Max Sharpe**).")
    st.markdown("Actifs étudiés : **" + " / ".join(tickers) + "**")
    st.markdown(f"Période d'historique : du **{START_DATE}** au **{END_DATE}**")
    
    st.divider()

    data = fetch_data(tickers)
    if data.empty:
        st.warning("Aucune donnée récupérée.")
        return
        
    # Lancement des calculs en backend
    opt_result, returns, cov_matrix = optimize_portfolio(data)
    
    # Les poids parfaits trouvés par l'algorithme ! (Dans la variable .x du résultat)
    optimal_weights = opt_result.x
    opt_ret, opt_vol = portfolio_performance(optimal_weights, returns, cov_matrix)
    opt_sharpe = opt_ret / opt_vol
    
    # Poids pour un portefeuille équipondéré (pour la comparaison)
    eq_weights = np.array(len(tickers) * [1. / len(tickers)])
    eq_ret, eq_std = portfolio_performance(eq_weights, returns, cov_matrix)
    eq_sharpe = eq_ret / eq_std

    # Affichage intelligent
    col_data, col_chart = st.columns([2, 1])
    
    with col_data:
        st.markdown("### Performances (Portefeuille Poids Optimisés)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ratio de Sharpe", f"{opt_sharpe:.2f}", delta=f"{(opt_sharpe - eq_sharpe):.2f} (vs équipondéré)")
        c2.metric("Rendement (Annuel)", f"{opt_ret*100:.2f} %", delta=f"{(opt_ret - eq_ret)*100:.2f}%")
        c3.metric("Risque / Volatilité", f"{opt_vol*100:.2f} %", delta=f"{(opt_vol - eq_std)*100:.2f}%", delta_color="inverse")
        
        st.markdown("### Évolution historique (Base 100)")
        # Ligne de cours normalisée pour la lecture (chaque actif commence à 100)
        st.line_chart(data / data.iloc[0] * 100)

    with col_chart:
        st.markdown("### L'Allocation Idéale")
        # Un beau camembert avec Plotly
        fig = go.Figure(data=[go.Pie(labels=tickers, values=optimal_weights, hole=.4)])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Petit tableau montrant les pondérations exactes
        df_w = pd.DataFrame(optimal_weights * 100, index=tickers, columns=['Poids (%)']).round(2)
        st.dataframe(df_w)
