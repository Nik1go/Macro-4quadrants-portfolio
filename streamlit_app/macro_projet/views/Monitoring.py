"""
Page 1: Monitoring Live
Real-time macroeconomic situation, quadrant scatter plot, and allocation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_loader import QUADRANT_NAMES, QUADRANT_COLORS, ALLOCATIONS


def render(data):
    st.header("Situation Macroeconomique Actuelle")

    # === 18-Day Trend (Last 18 days) ===
    st.subheader("Tendance Recente (18 derniers jours - Fenetre de Lissage)")
    if data['quadrants'] is not None:
        last_18 = data['quadrants'].tail(18).copy()

        q_counts = last_18['assigned_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)

        fig_trend = go.Figure(data=[go.Bar(
            x=[f"Q{i} {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=q_counts.values,
            textposition='auto',
        )])

        fig_trend.update_layout(
            title="Repartition des Quadrants (Brut) sur 18 jours",
            yaxis_title="Nombre de Jours",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        dominant_q = q_counts.idxmax()
    st.info(f"Le modele selectionne le **Mode (Valeur la plus frequente)** sur 18 jours. Tendance actuelle : **Q{dominant_q} {QUADRANT_NAMES.get(dominant_q)}** avec {q_counts.max()} jours.")

    st.divider()


    # === Smooth Quadrant Distribution (from Backtest) ===
    st.subheader("Repartition des Quadrants Lisses (Backtest Complet)")
    if data['backtest'] is not None and 'smooth_quadrant' in data['backtest'].columns:
        smooth_q_counts = data['backtest']['smooth_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        total_days = smooth_q_counts.sum()

        fig_smooth = go.Figure(data=[go.Bar(
            x=[f"Q{i} {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=smooth_q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=[f"{v} ({v / total_days * 100:.1f}%)" for v in smooth_q_counts.values],
            textposition='auto',
        )])

        fig_smooth.update_layout(
            title=f"Repartition des Quadrants - {total_days} jours de trading",
            yaxis_title="Nombre de Jours",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_smooth, use_container_width=True)

        dominant_smooth_q = smooth_q_counts.idxmax()
        start_date = data['backtest']['date'].min().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        end_date = data['backtest']['date'].max().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        st.info(f"Periode: **{start_date}** -> **{end_date}** | Regime dominant (lisse): **Q{dominant_smooth_q} {QUADRANT_NAMES.get(dominant_smooth_q)}** ({smooth_q_counts.max() / total_days * 100:.1f}%)")
    else:
        st.warning("Donnees smooth_quadrant non disponibles. Lancez le backtest pour generer ces donnees.")

    st.divider()

    # === Scatter Plot + Allocation ===
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Position dans le Cycle")
        if data['quadrants'] is not None:
            df_q = data['quadrants']

            fig = go.Figure()

            if 'MACRO_GROWTH_SCORE' in df_q.columns and 'MACRO_INFLATION_SCORE' in df_q.columns:
                inflation_hist = df_q['MACRO_INFLATION_SCORE']
                growth_hist = df_q['MACRO_GROWTH_SCORE']
                latest = df_q.iloc[-1]
                cur_inflation = latest['MACRO_INFLATION_SCORE']
                cur_growth = latest['MACRO_GROWTH_SCORE']
                x_title = "Weighted Inflation Score (New Logic) ->"
                y_title = "Weighted Growth Score (New Logic) ->"
                st.caption("Affichage base sur la nouvelle logique **2-Axes (Moyenne Ponderee)**")
            else:
                inflation_hist = df_q['score_Q2'] - df_q['score_Q4']
                growth_hist = df_q['score_Q1'] - df_q['score_Q3']
                latest = df_q.iloc[-1]
                cur_inflation = latest['score_Q2'] - latest['score_Q4']
                cur_growth = latest['score_Q1'] - latest['score_Q3']
                x_title = "Score d'Inflation ->"
                y_title = "Score de Croissance ->"

            max_val = max(abs(inflation_hist.min()), abs(inflation_hist.max()), abs(growth_hist.min()), abs(growth_hist.max()))
            limit = max(5.0, max_val * 1.2)

            # Quadrant backgrounds
            fig.add_shape(type="rect", x0=-limit, y0=0, x1=0, y1=limit, fillcolor="rgba(0,255,0,0.1)", line_width=0)
            fig.add_shape(type="rect", x0=0, y0=0, x1=limit, y1=limit, fillcolor="rgba(255,165,0,0.1)", line_width=0)
            fig.add_shape(type="rect", x0=0, y0=-limit, x1=limit, y1=0, fillcolor="rgba(255,0,0,0.1)", line_width=0)
            fig.add_shape(type="rect", x0=-limit, y0=-limit, x1=0, y1=0, fillcolor="rgba(0,0,255,0.1)", line_width=0)

            fig.add_trace(go.Scatter(
                x=inflation_hist, y=growth_hist, mode='markers',
                marker=dict(size=4, color=list(range(len(df_q))), colorscale='Blues', opacity=0.4, showscale=False),
                name='Historique complet',
                hovertext=df_q['date'].dt.strftime('%Y-%m-%d') if 'date' in df_q.columns else None
            ))

            df_recent = df_q.tail(90)
            if 'MACRO_GROWTH_SCORE' in df_recent.columns:
                recent_x = df_recent['MACRO_INFLATION_SCORE']
                recent_y = df_recent['MACRO_GROWTH_SCORE']
            else:
                recent_x = df_recent['score_Q2'] - df_recent['score_Q4']
                recent_y = df_recent['score_Q1'] - df_recent['score_Q3']

            fig.add_trace(go.Scatter(
                x=recent_x, y=recent_y,
                mode='lines+markers',
                marker=dict(size=6, color='yellow', opacity=0.8, line=dict(width=0.5, color='black')),
                line=dict(color='yellow', width=1, dash='solid'),
                name='90 derniers jours'
            ))

            fig.add_trace(go.Scatter(
                x=[cur_inflation], y=[cur_growth], mode='markers',
                marker=dict(size=20, color='red', symbol='star'), name='Actuel'
            ))

            fig.update_layout(
                xaxis_title=x_title, yaxis_title=y_title, height=500, showlegend=True,
                xaxis=dict(range=[-limit, limit], zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'), 
                yaxis=dict(range=[-limit, limit], zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
                margin=dict(l=0, r=0, t=30, b=0)
            )

            fig.add_annotation(x=-limit*0.5, y=limit*0.5, text="Q1: Croissance", showarrow=False, font=dict(size=14, color="rgba(255,255,255,0.6)"))
            fig.add_annotation(x=limit*0.5, y=limit*0.5, text="Q2: Inflation", showarrow=False, font=dict(size=14, color="rgba(255,255,255,0.6)"))
            fig.add_annotation(x=limit*0.5, y=-limit*0.5, text="Q3: Stagflation", showarrow=False, font=dict(size=14, color="rgba(255,255,255,0.6)"))
            fig.add_annotation(x=-limit*0.5, y=-limit*0.5, text="Q4: Deflation", showarrow=False, font=dict(size=14, color="rgba(255,255,255,0.6)"))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Donnees quadrants non disponibles")

    with c2:
        st.subheader("Allocation Actuelle")

        if data['backtest'] is not None:
            current_bt_q = int(data['backtest'].iloc[-1].get('smooth_quadrant', 1))
            alloc = ALLOCATIONS.get(current_bt_q, ALLOCATIONS[1])
            st.caption(f"Base sur le **Regime Modele Q{current_bt_q}** (Lisse)")
            alloc_df = pd.DataFrame({'Asset': alloc.keys(), 'Weight': alloc.values()})
            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        elif data['quadrants'] is not None:
            current_q = int(data['quadrants'].iloc[-1].get('assigned_quadrant', 1))
            alloc = ALLOCATIONS.get(current_q, ALLOCATIONS[1])
            st.warning(f"Base sur le Regime Brut Q{current_q} (Donnees Backtest manquantes)")
            alloc_df = pd.DataFrame({'Asset': alloc.keys(), 'Weight': alloc.values()})
            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Allocation non disponible")

    # === Recent Regime History ===
    st.subheader("Historique recent des Regimes")
    if data['quadrants'] is not None:
        recent = data['quadrants'].tail(20)[['date', 'assigned_quadrant', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']]
        recent['Regime'] = recent['assigned_quadrant'].map(QUADRANT_NAMES)
        st.dataframe(recent[['date', 'Regime', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']], use_container_width=True)

    st.divider()

    # === IBKR Paper Trading Dashboard ===
    st.header("📈 Compte IBKR Paper Trading")
    render_ibkr_dashboard(data)

def render_ibkr_dashboard(data):
    # Attempt to import PortfolioManager
    try:
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            
        from ibkr.portfolio import PortfolioManager
        has_ibkr_module = True
    except ImportError:
        has_ibkr_module = False
        
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Positions Actuelles (Live ou Dernier Log)")
        positions_df = None
        portfolio_val = None
        
        if has_ibkr_module:
            try:
                # Use a specific client ID for Streamlit to avoid conflicts
                pm = PortfolioManager(client_id=123, timeout=3)
                if pm.connect():
                    positions = pm.get_positions()
                    portfolio_val = pm.get_portfolio_value()
                    cash = pm.get_cash_balance()
                    pm.disconnect()
                    
                    if positions:
                        pos_list = []
                        for asset, info in positions.items():
                            pos_list.append({
                                'Asset': asset,
                                'Ticker': info['symbol'],
                                'Shares': info['shares'],
                                'Market Value ($)': round(info['market_value'], 2),
                                'Unrealized PNL ($)': round(info.get('unrealized_pnl', 0), 2)
                            })
                        positions_df = pd.DataFrame(pos_list)
                    st.success("✅ Connecté à IB Gateway (Live Data)")
                else:
                    st.warning("⚠️ Impossible de se connecter à IB Gateway. Affichage des données du dernier log.")
            except Exception as e:
                st.error(f"Erreur de connexion IBKR: {e}")
                
        # Fallback to logs if live fails
        if portfolio_val is None and 'ibkr_nav' in data and not data['ibkr_nav'].empty:
            portfolio_val = data['ibkr_nav'].iloc[-1]['nav']
            
        if portfolio_val is not None:
            st.metric("Portfolio Value (Net Liquidation)", f"${portfolio_val:,.2f}")
            
        if positions_df is not None and not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("Aucune position trouvée en direct.")
            
    with c2:
        st.subheader("Performance Historique")
        if 'ibkr_nav' in data and not data['ibkr_nav'].empty:
            nav_df = data['ibkr_nav'].copy()
            nav_df.set_index('date', inplace=True)
            
            # Plot NAV
            fig_nav = px.line(nav_df, y='nav', title="Evolution du Portefeuille (Logs)", markers=True)
            fig_nav.update_layout(yaxis_title="Valeur ($)", xaxis_title="Date", height=300)
            st.plotly_chart(fig_nav, use_container_width=True)
            
            # Simple stats
            first_val = nav_df['nav'].iloc[0]
            last_val = nav_df['nav'].iloc[-1]
            total_return = (last_val / first_val - 1) * 100 if first_val > 0 else 0
            
            st.metric("Total Return (depuis 1er log)", f"{total_return:.2f}%")
        else:
            st.info("Pas d'historique de NAV (Logs d'exécution introuvables)")

    st.subheader("Dernières Transactions (Logs)")
    if 'ibkr_orders' in data and not data['ibkr_orders'].empty:
        st.dataframe(data['ibkr_orders'].sort_values('Date', ascending=False).head(20), use_container_width=True)
    else:
        st.info("Aucune transaction trouvée dans les logs.")
