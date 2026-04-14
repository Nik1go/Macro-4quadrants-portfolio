import streamlit as st
import scipy.optimize as sco
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# --- CONFIGURATION ---
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"
CAPITAL_INITIAL = 10000
tickers = [
    'VWRL.AS', # Global Dividend Benchmark (Vanguard All-World)
    'CEM.PA',  # Indep. Mid Cap Benchmark (Amundi MSCI Europe Small Cap)
    'AEEM.PA', # Carmignac Emergent Benchmark (Amundi MSCI EM)
    'INR.PA',  # EdR India Benchmark (Amundi MSCI India)
    'PHPM.MI', # Métaux Benchmark (WisdomTree Physical Precious Metals)
    'IQQH.DE', # Clean Energy Benchmark (iShares Global Clean Energy)
    'VGEA.DE'  # Bonds d'Etat Benchmark (Vanguard Gov Bond)
]

TICKER_NAMES = {
    'VWRL.AS': 'Global Dividend (All-World)',
    'CEM.PA': 'Mid Cap Europe (MSCI)',
    'AEEM.PA': 'EM (Amundi MSCI)',
    'INR.PA': 'India (Amundi MSCI)',
    'PHPM.MI': 'Métaux (WisdomTree)',
    'IQQH.DE': 'Clean Energy (iShares)',
    'VGEA.DE': 'Bonds d’Etat (Vanguard)'
}


# --- DATA FETCHING ---
@st.cache_data
def fetch_data(tickers):
    try:
        raw = yf.download(tickers, start=START_DATE, end=END_DATE, interval=INTERVAL, progress=False)
        if raw.empty:
            return pd.DataFrame()

        # Handle different column structures
        if isinstance(raw.columns, pd.MultiIndex):
            # Case 1: Multiple tickers with MultiIndex (Metric, Ticker)
            # Find the best available price metric
            metrics = raw.columns.levels[0].unique()
            if 'Adj Close' in metrics:
                data = raw['Adj Close']
            elif 'Close' in metrics:
                data = raw['Close']
            else:
                # Fallback to the first available metric
                data = raw[metrics[0]]
        else:
            # Case 2: Single ticker or flattened columns
            available_cols = raw.columns.tolist()
            if 'Adj Close' in available_cols:
                data = raw[['Adj Close']]
            elif 'Close' in available_cols:
                data = raw[['Close']]
            else:
                # Take the first numeric column
                num_cols = raw.select_dtypes(include=[np.number]).columns
                if not num_cols.empty:
                    data = raw[[num_cols[0]]]
                else:
                    return pd.DataFrame()
            
            # For consistent downstream processing (ensure it's a DataFrame with ticker name as column)
            if len(tickers) == 1:
                data.columns = [tickers[0]]

        return data.dropna()
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def portfolio_performance(weights, returns, cov_matrix):
    # Le rendement du portefeuille = somme(poids * rendements moyens)
    mean_returns = returns.mean()
    p_ret = np.sum(mean_returns * weights) * 252
    
    # La volatilité = racine_carrée(Poids_Transposés * Matrice_Covariance * Poids) Markowitz
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
    num_assets = len(data.columns)

    # 2.1 : Poids de départ (guess). On donne le même poids à tout le monde
    initial_weights = np.array(num_assets * [1. / num_assets])

    # 2.2 : Les Limites (Bounds). L'utilisateur demande 5% min et 30% max par fonds
    bounds = tuple((0.05, 0.30) for _ in range(num_assets))

    
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


# --- EXTRA ANALYTICS ---
def generate_random_portfolios(returns, cov_matrix, num_portfolios=5000):
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
        results[0,i] = p_ret
        results[1,i] = p_vol
        results[2,i] = p_ret / p_vol
        
    return results

def find_min_vol_portfolio(returns, cov_matrix):
    num_assets = len(returns.columns)
    initial_weights = np.array(num_assets * [1. / num_assets])
    # On applique les mêmes limites (5% - 30%)
    bounds = tuple((0.05, 0.30) for _ in range(num_assets))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    result = sco.minimize(
        lambda w, r, c: portfolio_performance(w, r, c)[1],
        initial_weights,
        args=(returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result


# --- 3. RENDU INTERFACE UTILISATEUR (UI) ---
def render():
    st.title("Portfolio Optimizer (SciPy)")
    st.markdown("A des fin d'entrenement je cherche a optimiser la pondération de chaque actif dans un portefeuille qui maximiserais le ratio de Sharpe (rendement ajusté au risque).")
    st.markdown("J'ai pris ici des etf diversifié pour l'exemple. L'intérêt est de repliquer par la suite cette méthode dans mon projet macro 4 saisons.")

    st.markdown("Pour cela j'utilise donc scipy et markowitz")
    st.markdown("Actifs étudiés :")
    st.write(", ".join([f"{TICKER_NAMES.get(t, t)} ({t})" for t in tickers]))
    st.markdown(f"Période configurée : du **{START_DATE}** au **{END_DATE}**")

    
    st.divider()

    data = fetch_data(tickers)
    # Get the actual tickers downloaded (in case some failed)
    active_tickers = data.columns.tolist()
    
    if data.empty:
        st.warning("Aucune donnée récupérée.")
        return
    
    if len(active_tickers) < len(tickers):
        missing = set(tickers) - set(active_tickers)
        st.warning(f"Certains actifs n'ont pas pu être récupérés : {', '.join(missing)}")

    # Analyse de la disponibilité des données
    data_info = []
    for t in active_tickers:
        first_date = data[t].dropna().index.min()
        last_date = data[t].dropna().index.max()
        data_info.append({
            "Fonds": TICKER_NAMES.get(t, t),
            "Début": first_date.date() if pd.notnull(first_date) else "N/A",
            "Fin": last_date.date() if pd.notnull(last_date) else "N/A",
            "Jours": len(data[t].dropna())
        })
    
    df_info = pd.DataFrame(data_info)
    
    # On restreint les données à la période commune
    common_data = data.dropna()
    if common_data.empty:
        st.error("L'intersection des données pour tous les fonds est vide. Impossible d'optimiser car certains fonds sont trop récents ou n'ont pas de données communes.")
        st.dataframe(df_info)
        return

    common_start = common_data.index.min().date()
    st.info(f"Optimisation basée sur la période commune : du **{common_start}** au **{common_data.index.max().date()}**")
    
    with st.expander("Détails de disponibilité des données par fonds"):
        st.dataframe(df_info, use_container_width=True)

    # Lancement des calculs en backend
    opt_result, returns, cov_matrix = optimize_portfolio(common_data)

    
    # Calcul de la variance minimale pour comparaison
    min_vol_result = find_min_vol_portfolio(returns, cov_matrix)
    min_vol_ret, min_vol_std = portfolio_performance(min_vol_result.x, returns, cov_matrix)

    # Les poids parfaits trouvés par l'algorithme ! (Dans la variable .x du résultat)
    optimal_weights = opt_result.x
    opt_ret, opt_vol = portfolio_performance(optimal_weights, returns, cov_matrix)
    opt_sharpe = opt_ret / opt_vol
    
    # Poids pour un portefeuille équipondéré (pour la comparaison)
    eq_weights = np.array(len(active_tickers) * [1. / len(active_tickers)])
    eq_ret, eq_std = portfolio_performance(eq_weights, returns, cov_matrix)
    eq_sharpe = eq_ret / eq_std

    # Affichage intelligent
    col_data, col_chart = st.columns([2, 1])
    
    with col_data:
        st.markdown("### Performances (Portefeuille Poids Optimisés)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ratio de Sharpe", f"{opt_sharpe:.2f}", delta=f"{(opt_sharpe - eq_sharpe):.2f} (vs équipondéré)")
        c2.metric("Rendement (Annuel)", f"{opt_ret*100:.2f} %", delta=f"{(opt_ret - eq_ret)*100:.2f}%")
        c3.metric("Risque / Volatilité", f"{opt_vol*100:.2f} %", delta=f"{(opt_vol - min_vol_std)*100:.2f}%", delta_color="inverse")
        
        st.markdown("### Évolution historique (Base 100)")
        # Ligne de cours normalisée pour la lecture (chaque actif commence à 100)
        st.line_chart(data / data.iloc[0] * 100)

    with col_chart:
        st.markdown("### L'Allocation Idéale")
        # Un beau camembert avec Plotly
        fig = go.Figure(data=[go.Pie(
            labels=[TICKER_NAMES.get(t, t) for t in active_tickers], 
            values=optimal_weights, 
            hole=.4
        )])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Petit tableau montrant les pondérations exactes
        df_w = pd.DataFrame(optimal_weights * 100, index=[TICKER_NAMES.get(t, t) for t in active_tickers], columns=['Poids (%)']).round(2)
        st.dataframe(df_w)


    st.divider()
    
    # --- MONTE CARLO SECTION ---
    st.markdown("### Simulation Monte-Carlo & Frontière Efficiente")
    st.markdown("Cette simulation génère 5 000 combinaisons de portefeuilles aléatoires pour visualiser le sharpe ratio.")
    
    with st.spinner("Génération des portefeuilles aléatoires..."):
        mc_results = generate_random_portfolios(returns, cov_matrix, num_portfolios=5000)
        
    fig_mc = go.Figure()
    
    # Nuage de points
    fig_mc.add_trace(go.Scatter(
        x=mc_results[1, :] * 100,
        y=mc_results[0, :] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=mc_results[2, :], # Sharpe ratio
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(color='white')),
                tickfont=dict(color='white')
            ),
        ),
        name="Portefeuilles Simulés",
        hovertemplate="Volatilité: %{x:.2f}%<br>Rendement: %{y:.2f}%<extra></extra>"
    ))
    
    # Point Max Sharpe
    fig_mc.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name="MAX SHARPE",
        hovertemplate="MAX SHARPE<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
    ))
    
    # Point Min Vol
    fig_mc.add_trace(go.Scatter(
        x=[min_vol_std * 100],
        y=[min_vol_ret * 100],
        mode='markers',
        marker=dict(color='orange', size=12, symbol='diamond'),
        name="MIN VOLATILITÉ",
        hovertemplate="MIN VOLATILITE<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
    ))
    
    fig_mc.update_layout(
        xaxis_title="Volatilité Annuelle (%)",
        yaxis_title="Rendement Annuel (%)",
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(gridcolor='#3d4263', zerolinecolor='white'),
        yaxis=dict(gridcolor='#3d4263', zerolinecolor='white')
    )
    
    st.plotly_chart(fig_mc, use_container_width=True)
