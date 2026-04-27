import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURATION (from notebook) ---
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"
ZSCORE_WINDOW = 50
CAPITAL_INITIAL = 10000
POSITION_SIZE = 100
FLAT_FEE = 2.0

FIXED_LEVELS = [20, 40, 60]
FIXED_STOPS = [30, 50, 70]

Z_ENTRY_LEVELS = [2, 3, 4]
Z_STOP_LEVELS = [3, 4, 5]

class Position:
    def __init__(self, entry_price, size, side, level_idx, entry_date):
        self.entry_price = entry_price
        self.size = size
        self.side = side  # 1 for Long, -1 for Short
        self.level_idx = level_idx
        self.entry_date = entry_date

    def calculate_pnl(self, current_price):
        return (current_price - self.entry_price) * self.side * self.size

def run_strategy(df, strategy_type='fixed'):
    cash = CAPITAL_INITIAL
    equity = []
    positions = []
    trade_history = []
    
    for date, row in df.iterrows():
        spread_val = row['Spread']
        z_val = row['ZScore']
        
        signal_val = spread_val if strategy_type == 'fixed' else z_val
        
        # close trade tp sl
        for pos in positions[:]:
            pnl_gross = pos.calculate_pnl(spread_val)
            close_position = False
            reason = ""
            
            # Stop Loss
            if strategy_type == 'fixed':
                if pos.side == 1:  # Long
                    if spread_val <= -FIXED_STOPS[pos.level_idx]:
                        close_position = True
                        reason = "SL"
                else:  # Short
                    if spread_val >= FIXED_STOPS[pos.level_idx]:
                        close_position = True
                        reason = "SL"
            else:  # Dynamic
                if pos.side == 1:
                    if z_val <= -Z_STOP_LEVELS[pos.level_idx]:
                        close_position = True
                        reason = "SL"
                else:
                    if z_val >= Z_STOP_LEVELS[pos.level_idx]:
                        close_position = True
                        reason = "SL"
            
            # Take Profit
            if not close_position:
                if strategy_type == 'fixed':
                    if pos.side == 1 and spread_val >= 0:
                        close_position = True
                        reason = "TP"
                    elif pos.side == -1 and spread_val <= 0:
                        close_position = True
                        reason = "TP"
                else:  # Dynamic
                    if pos.side == 1 and z_val >= 0:
                        close_position = True
                        reason = "TP"
                    elif pos.side == -1 and z_val <= 0:
                        close_position = True
                        reason = "TP"
            
            if close_position:
                cash += pnl_gross - FLAT_FEE
                trade_history.append({
                    'Date': date, 'Type': 'Exit', 'Reason': reason, 
                    'PnL': pnl_gross - FLAT_FEE, 'Level': pos.level_idx, 'Side': pos.side
                })
                positions.remove(pos)
        
        # --- ENTREES ---
        levels = FIXED_LEVELS if strategy_type == 'fixed' else Z_ENTRY_LEVELS
        
        for idx, level in enumerate(levels):
            # SHORT (spread trop haut)
            if signal_val >= level:
                has_pos = any(p.level_idx == idx and p.side == -1 for p in positions)
                if not has_pos:
                    positions.append(Position(spread_val, POSITION_SIZE, -1, idx, date))
                    cash -= FLAT_FEE
                    trade_history.append({
                        'Date': date, 'Type': 'Entry', 'Side': 'Short', 
                        'Level': idx, 'Price': signal_val
                    })
            
            # LONG (spread trop bas)
            if signal_val <= -level:
                has_pos = any(p.level_idx == idx and p.side == 1 for p in positions)
                if not has_pos:
                    positions.append(Position(spread_val, POSITION_SIZE, 1, idx, date))
                    cash -= FLAT_FEE
                    trade_history.append({
                        'Date': date, 'Type': 'Entry', 'Side': 'Long', 
                        'Level': idx, 'Price': signal_val
                    })
        
        # --- VALORISATION ---
        latent_pnl = sum([p.calculate_pnl(spread_val) for p in positions])
        current_equity = cash + latent_pnl
        equity.append({'Date': date, 'Equity': current_equity})

    return pd.DataFrame(equity).set_index('Date'), pd.DataFrame(trade_history)
    
def calculate_stats(equity_df, trades_df):
    if equity_df.empty or trades_df.empty or 'PnL' not in trades_df.columns:
        return {"Win Rate": "0.0%", "Sharpe": "0.00", "Max DD": "0.0%", "Profit Factor": "0.00"}
    exits = trades_df[trades_df['Type'] == 'Exit']
    win_rate = (len(exits[exits['PnL'] > 0]) / len(exits)) * 100 if len(exits) > 0 else 0
    wins = exits[exits['PnL'] > 0]['PnL'].sum()
    losses = abs(exits[exits['PnL'] <= 0]['PnL'].sum())
    profit_factor = wins / losses if losses > 0 else (wins if wins > 0 else 0)
    returns = equity_df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    peak = equity_df['Equity'].cummax()
    max_dd = ((equity_df['Equity'] - peak) / peak).min() * 100
    return {"Win Rate": f"{win_rate:.1f}%", "Sharpe": f"{sharpe:.2f}", "Max DD": f"{max_dd:.1f}%", "Profit Factor": f"{profit_factor:.2f}"}

@st.cache_data(show_spinner=False)
def compute_rolling_adf(spread_values, window_size):
    pvals = [np.nan] * window_size
    for i in range(window_size, len(spread_values)):
        try:
            p = adfuller(spread_values[i-window_size:i])[1]
        except:
            p = np.nan
        pvals.append(p)
    return pvals

def render():
    st.title("Arbitrage Statistique : Paire trading")
    st.markdown("---")

    with st.expander("Comprendre la Stratégie (Introduction)", expanded=True):
        st.markdown("""
        L'arbitrage de paires repose sur le principe de **mean reversion**.
        Si deux actifs sont liés économiquement, leur écart (spread) devrait revenir à sa moyenne. 
        Arbitrer, c'est donc parier sur la convergence de cet écart lorsqu'il devient anormalement haut ou bas.
        <br> **Cette page n'a pas pour but de trouvé un réel alpha, mais plutot d'illustrer et de comprendre 
        comment fonctionne ce type de stratégie.** <br>
        """, unsafe_allow_html=True)

    # --- SELECTION DE LA PAIRE ---
    PAIRS = {
        "Or vs Argent (GLD / SLV)": {"asset1": "Or", "tk1": "GLD", "color1": "gold", "asset2": "Argent", "tk2": "SLV", "color2": "silver"},
        "Coca-Cola vs Pepsi (KO / PEP)": {"asset1": "Coca", "tk1": "KO", "color1": "red", "asset2": "Pepsi", "tk2": "PEP", "color2": "blue"},
        "Visa vs Mastercard (V / MA)": {"asset1": "Visa", "tk1": "V", "color1": "blue", "asset2": "Mastercard", "tk2": "MA", "color2": "orange"}
    }

    selected_pair_key = option_menu(
        menu_title=None,
        options=list(PAIRS.keys()),
        icons=["coin", "droplet-half", "credit-card"], 
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "#0a0e27",
                "border": "none",
                "border-radius": "0",
                "box-shadow": "none"
            },
            "icon": {"color": "#00d4ff", "font-size": "15px"},
            "nav-link": {
                "color": "#e8e8e8",
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#1e2139",
            },
            "nav-link-selected": {"background-color": "#00d4ff", "color": "white"},
        }
    )
    pair_info = PAIRS[selected_pair_key]
    
    asset1 = pair_info["asset1"]
    tk1 = pair_info["tk1"]
    color1 = pair_info["color1"]
    
    asset2 = pair_info["asset2"]
    tk2 = pair_info["tk2"]
    color2 = pair_info["color2"]

    # --- DATA FETCHING ---
    with st.spinner("Téléchargement des données..."):
        tickers = f"{tk1} {tk2}"
        data = yf.download(tickers, start=START_DATE, end=END_DATE, interval=INTERVAL, progress=False)
        if data.empty or 'Close' not in data.columns:
            st.error("Données indisponibles.")
            return
            
        a1_data = data['Close'][tk1] if tk1 in data['Close'] else data[tk1]
        a2_data = data['Close'][tk2] if tk2 in data['Close'] else data[tk2]
        
        df = pd.DataFrame({asset1: a1_data, asset2: a2_data}).dropna()

    # --- CALCULATIONS ---
    X_reg = df[[asset2]].values
    y_reg = df[asset1].values
    model = LinearRegression().fit(X_reg, y_reg)
    hedge_ratio = model.coef_[0]
    intercept = model.intercept_
    df['Spread'] = y_reg - model.predict(X_reg)
    df['ZScore'] = (df['Spread'] - df['Spread'].rolling(ZSCORE_WINDOW).mean()) / df['Spread'].rolling(ZSCORE_WINDOW).std()
    df['Correlation'] = df[asset1].rolling(252).corr(df[asset2])
    df = df.dropna()

    # --- STEP 1: VISUALISATION & CORRELATION ---
    st.header("Partie 1 : Validation de la Paire (Corrélation)")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df.index, y=df[asset1], name=asset1, line=dict(color=color1)))
        fig_p.add_trace(go.Scatter(x=df.index, y=df[asset2], name=asset2, line=dict(color=color2), yaxis="y2"))
        fig_p.update_layout(yaxis2=dict(overlaying="y", side="right"), template="plotly_dark", height=350)
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        st.metric("Corrélation (252j)", f"{df['Correlation'].iloc[-1]:.2f}")
        with st.expander("Pourquoi la Corrélation ?", expanded=True):
            st.write("C'est la première étape du filtrage. La corrélation est une **condition nécessaire mais non suffisante** au trading de paires.")

    # --- STEP 2: REGRESSION ---
    st.divider()
    st.header("Partie 2 : Construction du Modèle de Spread (Régression OLS)")
    c3, c4 = st.columns([1, 2])
    with c3:
        st.metric("Hedge Ratio", f"{hedge_ratio:.2f}")
        with st.expander("Le Ratio de Couverture", expanded=True):
            st.markdown(f"""
            Le Hedge Ratio (actuellement `{hedge_ratio:.2f}`) nous dit combien d'unités de {asset2} il faut vendre pour chaque unité de {asset1} achetée afin d'être "neutre au marché".
            
            **Le Spread** est l'écart résiduel :
            $$Spread = {asset1} - ({hedge_ratio:.2f} \\times {asset2} + {intercept:.2f})$$
            """)
    with c4:
        fig_s_adj = go.Figure()
        fig_s_adj.add_trace(go.Scatter(x=df.index, y=df['Spread'], name="Spread Ajusté", line=dict(color='#00d4ff')))
        fig_s_adj.update_layout(title=f"Le Spread Ajusté ({asset1} vs {asset2})", template="plotly_dark", height=300)
        st.plotly_chart(fig_s_adj, use_container_width=True)
        st.write("""Graphique du spread brut sans standardisation,<br> sklearn : LinearRegression().fit(df[['Asset2']],df['Asset1']).<br>""", unsafe_allow_html=True)

    # --- STEP 3: COINTEGRATION ---
    st.divider()
    st.header("Partie 3 : Cointégration et Fonctionnement Interne du Test (ADF)")
    
    with st.expander("Comprendre Mathématiquement le Test ADF", expanded=False):
        st.markdown(r"""
        Pour prouver mathématiquement que la paire est cointégrée, on utilise la librairie `statsmodels`. Le code exécuté sur notre spread calculé est le suivant :  
        `score, pvalue = adfuller(df['Spread'])[0:2]`

        **Le fonctionnement interne de la régression Augmented Dickey-Fuller (ADF) :**
        
        La formule modélisée par le test est la suivante :
        $$\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$
        
        * $y_t$ : Le **Spread** de notre paire.
        * $\alpha$ (La Constante / Drift) : C'est l'origine, déterminée par une méthode des moindres carrés (OLS) mise à jour chaque jour. Elle indique si le spread a une tendance naturelle à monter ou descendre (même sans un lien élastique).
        * $\gamma$ (Le Coefficient de Stationnarité) : C'est le **cœur du test**. Il lie le niveau d'hier ($y_{t-1}$) au mouvement d'aujourd'hui $\Delta y_t$. Plus il tire à l'inverse de la valeur, plus la force de rappel vers la moyenne est forte.
        * $\sum \delta_i \Delta y_{t-i}$ : On regarde les variations des jours précédents pour nettoyer le bruit et s'assurer que notre test n'est pas faussé par des mouvements très récents ou de la forte autocorrélation.
        * $\epsilon_t$ : Le reste en erreur de la régression (le bruit que l'on ne peut pas expliquer).

        **Obtention de la P-Value :**
        La valeur de la `p-value` de ce test précis est identifiée grâce au **T-Score** (score de confiance lié à $\gamma$) en le comparant dans la "table de Dickey Fuller" :
        - **H0** (Hypothèse Nulle : P-Value > 0.05) : Ce n'est pas stationnaire / il y a présomption de dérive aléatoire. L'élastique ne fonctionne pas à coup sûr.
        - **H1** (Hypothèse Alternative : P-Value <= 0.05) : On a de la stationnarité prouvée avec un fort niveau de certitude ($\gamma < 0$), l'élastique marchera.
        """)

    with st.spinner("Calcul glissant de la P-Value (100 jours) en cours..."):
        pvals = compute_rolling_adf(df['Spread'].to_numpy(), 100)
        df['Rolling_PValue'] = pvals

    fig_pval = go.Figure()
    fig_pval.add_trace(go.Scatter(x=df.index, y=df['Rolling_PValue'], name="P-Value (100j)", line=dict(color='#ab63fa')))
    fig_pval.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Seuil 0.05 (H1 acceptée)", annotation_position="bottom right")
    fig_pval.update_layout(title="Évolution de la P-Value au fil du temps (Fenêtre Glissante de 100 jours)", template="plotly_dark", height=300, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_pval, use_container_width=True)

    mean_pvalue = df['Rolling_PValue'].mean()
    last_pvalue = df['Rolling_PValue'].dropna().iloc[-1] if not df['Rolling_PValue'].dropna().empty else np.nan
    
    c5, c6 = st.columns(2)
    with c5:
        st.metric("Mean P-Value (Historique de la paire)", f"{mean_pvalue:.4f}")
        if mean_pvalue > 0.05:
            st.error("En moyenne, **H0 est majoritaire** (> 0.05). La paire est globalement **faiblement cointégrée** historiquement.")
        else:
            st.success("En moyenne, **H1 est validée** (<= 0.05). La paire est historiquement **bien cointégrée** et l'élastique opère.")
    with c6:
        st.metric("Valeur du test (100 derniers jours disponibles)", f"{last_pvalue:.4f}")
        if np.isnan(last_pvalue):
            st.write("Données insuffisantes.")
        elif last_pvalue > 0.05:
            st.warning("Sur sa partie récente étudiée, la P-Value a grimpé > 0.05. L'élastique actuel est peut-être distendu ou rompu.")
        else:
            st.success("Test réussi ! Récemment, l'élastique est solidement en place (P-Value <= 0.05) pour valider une stationnarité.")

    # --- STEP 4: SIGNALS ---
    st.divider()
    st.header("Partie 4 : Exécution de la Stratégie (Z-Score)")
    
    st.info("Le Z-Score mesure l'écart actuel par rapport à sa moyenne en unités d'écart-type. Un Z-Score > +2 (Vente) ou < -2 (Achat) indique un écart anormalement élevé.")

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=df.index, y=df['ZScore'], name="Z-Score", line=dict(color='#ff7f0e')))
    for l in [2, -2]: fig_z.add_hline(y=l, line_dash="dot", line_color="white")
    
    # Ajout des marqueurs Achat/Vente (croisements)
    buy_cond = (df['ZScore'] <= -2) & (df['ZScore'].shift(1) > -2)
    sell_cond = (df['ZScore'] >= 2) & (df['ZScore'].shift(1) < 2)
    buy_signals = df[buy_cond]
    sell_signals = df[sell_cond]
    fig_z.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['ZScore'], mode='markers', marker=dict(color='#00ff00', size=8, symbol='circle'), name="Achat"))
    fig_z.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['ZScore'], mode='markers', marker=dict(color='#ff0000', size=8, symbol='circle'), name="Vente"))

    fig_z.update_layout(title="Z-Score (Standardisation de l'écart)", template="plotly_dark", height=300)
    st.plotly_chart(fig_z, use_container_width=True)

    # --- STEP 5: PERFORMANCE ---
    st.divider()
    st.header("Partie 5 : Bilan de Performance et Analyse Post-Mortem")
    
    with st.expander("Stratégie Fixe", expanded=False  ):
        st.write("**Stratégie Fixe** : Se base sur des niveaux d'écart (spread) constants et prédéfinis. Les signaux d'achat et de vente sont déclenchés dès que l'écart atteint une valeur fixe (ex: +20 ou -20), sans tenir compte de l'évolution de la volatilité du marché.")
        st.write("*Exemple concret* : Si le Spread atteint 20, on vend. Ce seuil reste identique que le marché soit calme ou très agité.")
    
    with st.expander("Strategie Z-Score", expanded=False):
        st.write("**Stratégie Z-Score** : Se base sur un écart standardisé (le nombre d'écarts-types par rapport à la moyenne). Elle s'adapte automatiquement à la volatilité du marché.")
        st.write("*Exemple concret* : Si l'écart type du spread est de 2, un Z-Score de 2 déclenche une vente à un spread de +4 au-dessus de la moyenne. Si le marché devient nerveux et que l'écart type monte à 5, le même Z-Score de 2 ne déclenchera une vente qu'à +10. Cela permet d'éviter d'entrer trop tôt quand le marché est agité.")
    
   

    eq_f, tr_f = run_strategy(df, 'fixed')
    eq_d, tr_d = run_strategy(df, 'dynamic')
    s_f, s_d = calculate_stats(eq_f, tr_f), calculate_stats(eq_d, tr_d)

    col_f, col_d = st.columns(2)
    for col, name, stats, trades in zip([col_f, col_d], ["Fixe", "Z-Score"], [s_f, s_d], [tr_f, tr_d]):
        with col:
            st.subheader(f"Stratégie {name}")
            m1, m2 = st.columns(2)
            m1.metric("Win Rate", stats["Win Rate"])
            m2.metric("Sharpe", stats["Sharpe"])
            m3, m4 = st.columns(2)
            m3.metric("Max DD", stats["Max DD"])
            m4.metric("Profit Factor", stats["Profit Factor"])
            with st.expander(f"Historique des Trades ({name})"):
                if not trades.empty: st.dataframe(trades, use_container_width=True)
                else: st.write("Aucun trade.")

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=eq_f.index, y=eq_f['Equity'], name="Fixe"))
    fig_res.add_trace(go.Scatter(x=eq_d.index, y=eq_d['Equity'], name="Z-Score"))
    fig_res.update_layout(title="Equity Curves", template="plotly_dark", height=400)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("### Analyse des Résultats et Interprétation")
    if mean_pvalue > 0.05:
        st.write("L'instabilité ou la dérive de l'equity curve illustre le risque d'arbitrer une paire avec une P-Value élevée (non cointégrée). L'élastique finit par rompre ou s'étendre trop longtemps. Le spread dérive indéfiniment. C'est la preuve que la validation de la cointégration est l'étape la plus critique, bien plus que la corrélation.")
    else:
        st.write("Ici la cointégration est probante (P-Value <= 0.05), ce qui donne aux stratégies de retour à la moyenne (comme le Z-Score) une meilleure probabilité de succès en maintenant une performance plus stable. L'élastique ramène systématiquement l'écart à sa moyenne globale.")

if __name__ == "__main__":
    render()
