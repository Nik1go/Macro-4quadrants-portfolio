import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURATION (from notebook) ---
PERIOD = "5y"
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
    
def calculate_stats(equity_df, trades_df):
    if equity_df.empty or trades_df.empty:
        return {"Win Rate": "0%", "Sharpe": "0.00", "Max DD": "0%", "Profit Factor": "0.00"}
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

def render():
    st.title("Arbitrage Statistique : Or vs Argent")
    st.markdown("---")

    with st.expander("Comprendre la Stratégie (Introduction)", expanded=False):
        st.markdown("""
        L'arbitrage de paires repose sur le principe de **retour à la moyenne**. Si deux actifs sont liés économiquement, leur écart (spread) devrait rester stable. 
        Trader l'arbitrage, c'est parier sur la convergence de cet écart lorsqu'il devient anormalement haut ou bas.
        """)

    # --- DATA FETCHING ---
    with st.spinner("Téléchargement des données..."):
        data = yf.download("GLD SLV", period=PERIOD, interval=INTERVAL, progress=False)
        if data.empty or 'Close' not in data.columns:
            st.error("Données indisponibles.")
            return
        gld = data['Close']['GLD'] if 'GLD' in data['Close'] else data['GLD']
        slv = data['Close']['SLV'] if 'SLV' in data['Close'] else data['SLV']
        df = pd.DataFrame({'Or': gld, 'Argent': slv}).dropna()

    # --- CALCULATIONS ---
    X_reg = df[['Argent']].values
    y_reg = df['Or'].values
    model = LinearRegression().fit(X_reg, y_reg)
    hedge_ratio = model.coef_[0]
    intercept = model.intercept_
    df['Spread'] = y_reg - model.predict(X_reg)
    df['ZScore'] = (df['Spread'] - df['Spread'].rolling(ZSCORE_WINDOW).mean()) / df['Spread'].rolling(ZSCORE_WINDOW).std()
    df['Correlation'] = df['Or'].rolling(50).corr(df['Argent'])
    df = df.dropna()

    # --- STEP 1: VISUALISATION & CORRELATION ---
    st.header("1. Corrélation : Sont-ils liés ?")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df.index, y=df['Or'], name="Or", line=dict(color='gold')))
        fig_p.add_trace(go.Scatter(x=df.index, y=df['Argent'], name="Argent", line=dict(color='silver'), yaxis="y2"))
        fig_p.update_layout(yaxis2=dict(overlaying="y", side="right"), template="plotly_dark", height=350)
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        st.metric("Corrélation (50j)", f"{df['Correlation'].iloc[-1]:.2f}")
        with st.expander("Pourquoi la corrélation ?", expanded=True):
            st.write("La corrélation doit être forte (>0.80) pour justifier un arbitrage. Si elle chute, les actifs se déconnectent.")

    # --- STEP 2: REGRESSION ---
    st.divider()
    st.header("2. Régression : Quel est le Ratio ?")
    c3, c4 = st.columns([1, 2])
    with c3:
        st.metric("Hedge Ratio", f"{hedge_ratio:.2f}")
        with st.expander("Le Ratio de Couverture", expanded=False):
            st.markdown(f"""
            Le Hedge Ratio (`{hedge_ratio:.2f}`) nous dit combien d'unités d'Argent il faut vendre pour chaque unité d'Or achetée afin d'être "neutre au marché".
            
            **Le Spread** est l'écart résiduel :
            $$Spread = Or - ({hedge_ratio:.2f} \\times Argent + {intercept:.2f})$$
            """)
    with c4:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df.index, y=df['Spread'], name="Spread", line=dict(color='#00d4ff')))
        fig_s.update_layout(title="Le Spread (Ecart reel en $)", template="plotly_dark", height=300)
        st.plotly_chart(fig_s, use_container_width=True)

    # --- STEP 3: COINTEGRATION ---
    st.divider()
    st.header("3. Cointégration : Retourne-t-il au centre ?")
    score, pvalue, *unused = adfuller(df['Spread'])
    c5, c6 = st.columns(2)
    with c5:
        st.metric("P-Value (ADF)", f"{pvalue:.4f}")
        if pvalue < 0.05: st.success("P-Value < 0.05 : Le spread est stable.")
        else: st.error("P-Value > 0.05 : Le spread dérive.")
    with c6:
        with st.expander("C'est quoi la cointégration ?", expanded=True):
            st.write("C'est la certitude mathématique que l'écart finira par revenir à sa moyenne. C'est plus fort que la corrélation.")

    # --- STEP 4: SIGNALS ---
    st.divider()
    st.header("4. Signaux : Quand entrer ?")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=df.index, y=df['ZScore'], name="Z-Score", line=dict(color='#ff7f0e')))
    for l in [2, -2]: fig_z.add_hline(y=l, line_dash="dot", line_color="red")
    fig_z.update_layout(title="Z-Score (Standardisation de l'écart)", template="plotly_dark", height=300)
    st.plotly_chart(fig_z, use_container_width=True)

    # --- STEP 5: PERFORMANCE ---
    st.divider()
    st.header("5. Performance & Journal de Bord")
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

if __name__ == "__main__":
    render()
