"""
Page 1: Monitoring Live
Real-time macroeconomic situation, quadrant scatter plot, and allocation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_loader import QUADRANT_NAMES, QUADRANT_COLORS, ALLOCATIONS

# --- Friendly display names for indicator columns ---
INDICATOR_LABELS = {
    "INFLATION":               "Inflation (CPI YoY)",
    "BREAKEVEN_10Y":           "Breakeven 10Y",
    "High_Yield_Bond_SPREAD":  "High Yield Spread",
    "10-2Year_Treasury_Yield_Spread": "Courbe des Taux (10-2Y)",
    "CONSUMER_SENTIMENT":      "Sentiment Consommateur",
    "INITIAL_CLAIMS":          "Inscriptions Chômage",
    "US_DOLLAR_INDEX":         "Dollar Index (DXY)",
    "WTI_CRUDE_OIL":           "WTI Pétrole",
    "COPPER":                  "Cuivre",
    "VIX":                     "VIX",
    "HOUSING_PERMITS":         "Permis Construire",
    "IND_PRODUCTION":          "Production Industrielle",
    "NFCI":                    "NFCI (Conditions Fin.)",
    "NET_LIQUIDITY":           "Net Liquidity Fed",
    "WALCL":                   "Bilan Fed (WALCL)",
}

SKIP_COLUMNS = {"date", "TAUX_ECB", "TAUX_BOJ", "TAUX_BOC", "TAUX_RBA", "TAUX_BCB",
                "WTREGEN", "RRPONTSYD", "WALCL","COPPER","US_DOLLAR_INDEX","VIX"}


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
                        st.warning("⚠️ Connecté à IB Gateway, mais aucune position correspondante trouvée (vérifier ETF_MAPPING).")
                else:
                    st.warning("⚠️ Impossible de se connecter à IB Gateway. Affichage des données du dernier log.")
            except Exception as e:
                st.error(f"Erreur de connexion IBKR: {e}")
                
        # --- Fallback to logs if live fails or returns nothing ---
        # 1. Fallback for Portfolio Value
        if portfolio_val is None:
            if data.get('ibkr_last_portfolio_val'):
                portfolio_val = data['ibkr_last_portfolio_val']
            elif 'ibkr_nav' in data and not data['ibkr_nav'].empty:
                portfolio_val = data['ibkr_nav'].iloc[-1]['nav']
            
        if portfolio_val is not None:
            st.metric("Valeur du Portefeuille", f"${portfolio_val:,.2f}")
            
        # 2. Fallback for Positions
        if positions_df is not None and not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        elif data.get('ibkr_last_positions'):
            st.info("Affichage des derniers poids connus (Logs)")
            last_pos = data['ibkr_last_positions']
            # Convert weights dict to DataFrame
            weights_df = pd.DataFrame([
                {'Asset': k, 'Weight (%)': round(v * 100, 2)} 
                for k, v in last_pos.items() if v > 0
            ])
            st.dataframe(weights_df, use_container_width=True)
        else:
            st.info("Aucune position trouvée (Direct ou Logs).")
            
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
        df = data['ibkr_orders'].sort_values('Date', ascending=False).head(40)
        # Reorder columns for better readability
        cols = ['Date', 'Action', 'Asset', 'Status', 'Shares', 'Estimated Value ($)', 'Error', 'Reason']
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        st.dataframe(
            df, 
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="Order execution status"),
                "Error": st.column_config.TextColumn("Détails Erreur", help="Raison du rejet IBKR"),
                "Estimated Value ($)": st.column_config.NumberColumn("Valeur ($)", format="$%.2f")
            }
        )
    else:
        st.info("Aucune transaction trouvée dans les logs.")



def _render_last_indicators(data):
    """Display the 5 most recently updated macro indicators using raw backup files."""
    recent_info = data.get("recent_indicators", [])
    if not recent_info:
        st.warning("Données d'indicateurs non disponibles ou historique insuffisant.")
        return

    top5 = recent_info[:5]

    cols = st.columns(len(top5))
    for i, info in enumerate(top5):
        with cols[i]:
            label = INDICATOR_LABELS.get(info["col"], info["col"].replace("_", " ").title())
            
            # Format values: show as % if small absolute value (e.g. rates/spreads), else 2 dec
            def _fmt(v):
                return f"{v:.3f}" if abs(v) < 10 else f"{v:,.2f}"

            delta_str = f"{info['pct_change']:+.2f}%  (anc.: {_fmt(info['prev_val'])})"
            delta_color = "normal" if info["pct_change"] >= 0 else "inverse"

            st.metric(
                label=label,
                value=_fmt(info["last_val"]),
                delta=delta_str,
                delta_color=delta_color,
            )
            st.caption(f"Publié le {info['last_date'].strftime('%Y-%m-%d')}")



def render(data):
    # === IBKR Paper Trading Dashboard ===
    st.header("Compte IBKR Paper Trading")
    render_ibkr_dashboard(data)
    st.divider()

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
    st.info(f"Le modele selectionne le **Mode (Valeur la plus frequente)** sur 5 jours glissants. Tendance actuelle : **Q{dominant_q} {QUADRANT_NAMES.get(dominant_q)}** avec {q_counts.max()} jours.")

    st.divider()

    # === Last 5 Fetched Indicators ===
    st.subheader("Derniers Indicateurs Fetchés")
    _render_last_indicators(data)

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
                x_title = "Weighted Inflation Score "
                y_title = "Weighted Growth Score "
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

            # Ligne fine noire pour le trajet
            fig.add_trace(go.Scatter(
                x=recent_x, y=recent_y,
                mode='lines',
                line=dict(color='black', width=1, dash='solid'),
                name='Trajectoire (90 derniers jours)'
            ))

            # Ajouter des flèches directionnelles le long du trajet (1 toutes les 4 périodes pour la lisibilité)
            step = 4
            for i in range(0, len(recent_x) - 1, step):
                fig.add_annotation(
                    x=recent_x.iloc[i+1], y=recent_y.iloc[i+1],
                    ax=recent_x.iloc[i], ay=recent_y.iloc[i],
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1,
                    arrowcolor='yellow'
                )
            # S'assurer qu'il y a une flèche sur le dernier segment si non couvert
            if len(recent_x) > 1 and (len(recent_x) - 2) % step != 0:
                fig.add_annotation(
                    x=recent_x.iloc[-1], y=recent_y.iloc[-1],
                    ax=recent_x.iloc[-2], ay=recent_y.iloc[-2],
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1,
                    arrowcolor='yellow'
                )

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
            last_row = data['backtest'].iloc[-1]
            current_bt_q = int(last_row.get('smooth_quadrant', 1))
            st.caption(f"Base sur le **Regime Modele Q{current_bt_q}** (Lisse)")
            
            # Extract actual weights from backtest result columns
            weight_cols = [c for c in data['backtest'].columns if c.endswith('_base_weight') and '_hc_' not in c]
            weights = last_row[weight_cols]
            weights = weights[weights > 0.005] # Filter tiny values
            
            if not weights.empty:
                alloc_df = pd.DataFrame({
                    'Asset': [c.replace('_base_weight', '').replace('_weight', '') for c in weights.index],
                    'Weight': weights.values
                })
                fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_traces(textinfo='percent+label', hole=.45)
                fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                # Fallback to hardcoded if columns missing
                alloc = ALLOCATIONS.get(current_bt_q, ALLOCATIONS[1])
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
    with st.expander("Historique recent des Regimes", expanded=False):
        if data['quadrants'] is not None:
            recent = data['quadrants'].tail(20)[['date', 'assigned_quadrant', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']]
            recent['Regime'] = recent['assigned_quadrant'].map(QUADRANT_NAMES)
            st.dataframe(recent[['date', 'Regime', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']], use_container_width=True)


