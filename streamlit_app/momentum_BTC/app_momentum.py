import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def render():
    import momentum_BTC.momentum_utils as mu

    st.title("Crypto Momentum Trading")
    st.markdown("**Exploitation des effets de mode et de momentum sur les cryptomonnaies.**")
    st.markdown("---")

    # Custom CSS for dropdowns
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
        
        /* Expander styling */
        div[data-testid="stExpander"] details {
            background-color: #1a1d35 !important;
            border: 1px solid #3d4263 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stExpander"] summary {
            background-color: #1e2139 !important;
            color: #00d4ff !important;
            padding: 15px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: #2a2d45 !important;
            border-color: #00d4ff !important;
        }
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #0a0e27 !important;
            padding: 20px !important;
            border-top: 1px solid #3d4263 !important;
        }
        
        /* Tab navigation styling - bigger + separated */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px !important;
            border-bottom: 2px solid #3d4263 !important;
            padding-bottom: 0 !important;
            width: 100% !important;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1 !important;
            justify-content: center !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            padding: 14px 28px !important;
            color: #8890b5 !important;
            border-right: 2px solid #3d4263 !important;
        }
        .stTabs [data-baseweb="tab"]:last-child {
            border-right: none !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #00d4ff !important;
            background: rgba(0, 212, 255, 0.08) !important;
            border-bottom: 3px solid #00d4ff !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #00d4ff !important;
            background: rgba(0, 212, 255, 0.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # === TABS ===
    tab_strat, tab_live, tab_backtest = st.tabs(["STRATEGIE", "LIVE PIPELINE MONITORING", "BACKTEST & PERF"])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: STRATEGY EXPLANATION
    # ══════════════════════════════════════════════════════════════
    with tab_strat:
        st.markdown("###  Hypothèse de la Stratégie")
        st.markdown("""
        Les altcoins sont des actifs dont la valorisation repose, selon moi, essentiellement sur la **psychologie des investisseurs** et les dynamiques de flux. 
        
        L'objectif de ce projet est de déterminer s'il existe des **effets de mode**, sur les altcoins et s'il est possible de définir une méthodologie pour les exploiter.

        **Le mécanisme observé est le suivant :**
        * **L'effet de richesse :** Lorsque le Bitcoin affiche une tendance haussière forte, les capitaux ont tendance à se déplacer vers les actifs plus risqués (**Altcoins**), créant des phénomènes de momentum à court et moyen terme.
        * **L'abandon massif :** À l'inverse, en période de chute du Bitcoin, les altcoins affichant une faiblesse relative sont délaissés prioritairement, offrant des opportunités de **Vente à découvert (Short)**.

        L'idée générale est donc d'identifier quelles cryptomonnaies captent l'attention du marché (ou sont délaissées) pour exploiter le **momentum** généré par ces flux de capitaux.
        """)

        st.markdown("---")

        with st.expander("Conditions d'Entrée & Sortie", expanded=False):
            # Long/Short side by side
            col_long, col_short = st.columns(2)
            

            with col_long:
                st.markdown("### Position Long (Achat)")
                st.markdown("""
                **Conditions d'Entrée :**
                1. **Univers :** Sélection des 20 altcoins avec le plus gros volume (hors stablecoins).
                2. **Régime de Distribution :** *120d Skewness > 0.15* (asymétrie positive significative).
                3. **Force du Mouvement :** *BTC 5D Return > Médiane + 0.5σ* (momentum haussier exceptionnel).
            4. **Confirmation Volume :** Volume BTC du jour > SMA(20) des volumes (le mouvement est soutenu par de vrais flux).
            5. **Confirmation BTC :** Prix > SMA depuis **2 jours consécutifs**.
            6. **Performance Relative :** Paire ALT/BTC > SMA depuis **2 jours consécutifs**.
            7. **Sélection finale :** L'actif ayant la **meilleure performance sur 3 jours** parmi les filtrés.
            8. **Allocation :** 50% du cash disponible.

            **Conditions de Sortie :**
            * L'actif sous-performe le panier moyen des altcoins pendant **3 jours** consécutifs.
            * **OU :** Le BTC repasse sous sa SMA pendant **2 jours** consécutifs.
            * **OU :** **Trailing Stop ATR** : le prix chute de plus de **2× ATR(14)** depuis son plus haut en position.
            """)

        with col_short:
            st.markdown("### Position Short (Vente)")
            st.markdown("""
            **Conditions d'Entrée :**
            1. **Univers :** Sélection des 20 altcoins avec le plus gros volume (hors stablecoins).
            2. **Régime de Distribution :** *120d Skewness < -0.15* (asymétrie négative forte).
            3. **Force du Mouvement :** *BTC 5D Return < Médiane - 0.5σ* (momentum baissier marqué).
            4. **Confirmation Volume :** Volume BTC du jour > SMA(20) des volumes (le mouvement est soutenu par de vrais flux).
            5. **Confirmation BTC :** Prix < SMA depuis **2 jours consécutifs**.
            6. **Performance Relative :** Paire ALT/BTC < SMA depuis **2 jours consécutifs**.
            7. **Sélection finale :** L'actif ayant la **pire performance sur 3 jours** parmi les filtrés.
            8. **Allocation :** 25% du cash disponible.

            **Conditions de Sortie :**
            * L'actif sur-performe le panier moyen pendant **3 jours** consécutifs.
            * **OU :** Le BTC repasse au-dessus de sa SMA pendant **2 jours** consécutifs.
            * **OU :** **Trailing Stop ATR** : le prix rebondit de plus de **2× ATR(14)** depuis son plus bas en position.
            """)

        st.markdown("---")
        with st.expander("1. Architecture Technique (Data Engineering Pipeline)", expanded=False):
            st.markdown("""
            L'ensemble du pipeline est entièrement automatisé et executé chaque jours à 00H05 UTC (01H05/02H05 heure de Paris) par **Apache Airflow**, via **Apache Spark**, de l'ingestion de la donnée jusqu'à l'exécution d'ordres de trading via l'API d'Interactive Brokers.
            """)
        
            try:
                import os
                from PIL import Image
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # current_dir is streamlit_app/macro_projet/views
                # project_root is streamlit_app/
                project_root = os.path.dirname(os.path.dirname(current_dir))
                image_path = os.path.join(project_root, "streamlit_app", "images", "momentum_workflow.png")
                img = Image.open(image_path)
                st.image(img, caption="Architecture Data Engineering & ML Pipeline", use_container_width=True)
            except Exception as e:
                st.error(f"Erreur d'ouverture de l'image : {e} (Chemin essayé : {image_path})")
            
        st.markdown("""
        **Pipeline ETL & ML :**
        1. **Ingestion (Task `fetch_data`)** : Récupération des données brutes via l'API FRED (macroéconomie) et Yahoo Finance (prix des actifs).
        2. **Feature Engineering (Task `prepare_indicators`)** : Nettoyage, synchronisation temporelle, et calcul des métriques dérivées.
        3. **Modélisation ML (Spark Job `train_model`)** : Entraînement distribué des classifieurs Random Forest (GridSearchCV + Walk-Forward test).
        4. **Inférence (Spark Job `compute_quadrants`)** : Prédiction mensuelle probabiliste et assignation au quadrant correspondant.
        5. **Simulation & Trading (Spark/IBKR)** : Backtesting de la stratégie avec prise en compte des coûts de transaction, exécution automatique des ordres de réallocation.
        """)

    # ══════════════════════════════════════════════════════════════
    # TAB 2: BACKTEST (Offline from Airflow pipeline)
    # ══════════════════════════════════════════════════════════════
    with tab_backtest:
        st.markdown("### Optimisation des Paramètres (Heatmap)")
        st.markdown("*Résultats de la stratégie Long/Short (Sizing 25%) pré-calculés hors ligne par Airflow via VectorBT.*")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        backtest_out_dir = os.path.join(project_root, "data", "crypto", "backtest_results")

        heatmap_path = os.path.join(backtest_out_dir, "heatmap.csv")
        equity_path = os.path.join(backtest_out_dir, "equity_curves.csv")
        summary_path = os.path.join(backtest_out_dir, "backtest_summary.json")
        trades_path = os.path.join(backtest_out_dir, "trades_log.csv")

        if not os.path.exists(heatmap_path) or not os.path.exists(summary_path):
            st.warning(" Les résultats du Backtest ne sont pas encore générés.")
            st.info("Lancez le script `pipeline_crypto_momentum/backtest/run_backtest.py` via Airflow pour calculer l'optimisation.")

        else:
            col_params, col_results = st.columns([1, 3])

            with col_params:
                import json
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                st.markdown("#### Informations du Backtest")
                st.markdown("**Univers :** Top 12 Cryptos (Filtre ALT/BTC)")
                st.markdown(f"**Date de début :** {summary.get('start_date', 'N/A')}")
                st.markdown(f"**Frais :** 6 bps | **Slippage :** 10 bps")

                st.markdown("---")
                st.markdown("### Meilleure Configuration")
                st.info(f" SMA : **{summary.get('best_sma', 'N/A')}**\n\n🏆 Lookback : **{summary.get('best_lookback', 'N/A')}**")

                st.markdown("### Métriques globales")
                st.metric("Rendement Total", f"{summary.get('tot_ret_pct', 0.0):.2f}%")
                st.metric("Win Rate", f"{summary.get('win_rate_pct', 0.0):.2f}%")
                st.metric("Ratio de Sharpe", f"{summary.get('sharpe', 0.0):.2f}")
                st.metric("Max Drawdown", f"{summary.get('max_dd_pct', 0.0):.2f}%")

            with col_results:
                # 1. Heatmap
                st.markdown("#### Matrice des Rendements par Paramètres")
                heatmap_df = pd.read_csv(heatmap_path, index_col=0)
                heatmap_df.columns = [int(float(c)) for c in heatmap_df.columns] # Clean column headers
                
                import plotly.express as px
                fig_hm = px.imshow(
                    heatmap_df,
                    labels=dict(x="Lookback (Jours)", y="SMA (Jours)", color="Rendement (%)"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                fig_hm.update_layout(
                    plot_bgcolor="#0a0e27",
                    paper_bgcolor="#0a0e27",
                    font_color="white",
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_hm, use_container_width=True)

                # 2. Equity Curve
                st.markdown("#### Évolution du Capital (vs Buy & Hold BTC)")
                if os.path.exists(equity_path):
                    equity_df = pd.read_csv(equity_path, index_col=0)
                    # Safe cast index to datetime if possible
                    try: 
                        equity_df.index = pd.to_datetime(equity_df.index)
                    except: pass
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(equity_df.index, equity_df["Momentum_Equity"], label="Capital (Momentum PnL)", color="#00ff00", linewidth=1.5)
                    ax.plot(equity_df.index, equity_df["Buy_and_Hold_BTC"], label="Buy & Hold (BTC)", color="#f7931a", linewidth=1.5, alpha=0.7)

                    fig.patch.set_facecolor('#0a0e27')
                    ax.set_facecolor('#0a0e27')
                    for spine in ['bottom', 'top', 'left', 'right']:
                        ax.spines[spine].set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Équité (USDT)')
                    ax.legend(facecolor='#0a0e27', edgecolor='white', labelcolor='white')

                    st.pyplot(fig)

                # 3. Trades log
                st.markdown("#### Aperçu des Derniers Trades Réalisés")
                if os.path.exists(trades_path):
                    trades_df = pd.read_csv(trades_path)
                    if not trades_df.empty and "Return" in trades_df.columns:
                        # Ensure numeric returns and clean column naming
                        trades_df["Return"] = pd.to_numeric(trades_df["Return"], errors="coerce")
                        
                        cols = ["Column", "Side", "Entry Timestamp", "Exit Timestamp", "Avg Entry Price", "Avg Exit Price", "Return", "Size", "BTC Skew", "Extrême Condition"]
                        disp_cols = [c for c in cols if c in trades_df.columns]
                        disp_df = trades_df[disp_cols].tail(20)
                        
                        def highlight_returns(val):
                            if pd.isna(val): return ""
                            return f'color: {"#00ff00" if val > 0 else "#ff4444"}; font-weight: bold;'
                        
                        styled_df = disp_df.style.applymap(highlight_returns, subset=["Return"])
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.dataframe(trades_df.tail(20))
                else:
                    st.warning("Aucun csv de trades trouvé.")

    # ══════════════════════════════════════════════════════════════
    # TAB 3: LIVE MONITORING
    # ══════════════════════════════════════════════════════════════
    with tab_live:
        st.markdown("### Live Pipeline Monitoring")

        # Resolve data paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        crypto_data_dir = os.path.join(project_root, "data", "crypto")
        alt_usdt_dir = os.path.join(crypto_data_dir, "ALT_USDT")
        alt_btc_dir = os.path.join(crypto_data_dir, "ALT_BTC")

        state_path = os.path.join(crypto_data_dir, "state.json")
        nav_path = os.path.join(crypto_data_dir, "nav_history.csv")

        # Check if data exists
        if not os.path.exists(alt_usdt_dir) or not os.listdir(alt_usdt_dir):
            st.warning("⚠️ Aucune donnée crypto. Lancez le DAG Airflow pour commencer.")
            st.code("airflow dags trigger dag_crypto_momentum", language="bash")
            return

        # ── Load indicators for live dashboard ──
        try:
            import sys
            momentum_dir = os.path.join(project_root, "streamlit_app", "momentum_BTC")
            if momentum_dir not in sys.path:
                sys.path.insert(0, momentum_dir)
            from indicators.calc_indicators import compute_all_indicators

            # Load optimal parameters from backtest summary
            summary_path = os.path.join(crypto_data_dir, "backtest_results", "backtest_summary.json")
            live_lookback = 120
            live_sma = 50
            if os.path.exists(summary_path):
                import json
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    live_lookback = summary.get("best_lookback", 120)
                    live_sma = summary.get("best_sma", 50)
            
            st.markdown(f"**Paramètres Actifs (issus du Backtest) :** SMA = {live_sma} | Lookback = {live_lookback}")
            st.markdown("---")

            indicators = compute_all_indicators(sma_period=live_sma, roll_lookback=live_lookback)
            today_idx = len(indicators["btc_close"]) - 1
            today = indicators["btc_close"].index[today_idx]
            data_loaded = True
        except Exception as e:
            st.error(f"Erreur de chargement des indicateurs: {e}")
            data_loaded = False

        if data_loaded:
            btc_close = float(indicators["btc_close"].iloc[today_idx])
            btc_sma = float(indicators["btc_sma"].iloc[today_idx])
            btc_ret = float(indicators["btc_ret_5d"].iloc[today_idx])
            btc_med = float(indicators["btc_median"].iloc[today_idx])
            btc_std = float(indicators["btc_std"].iloc[today_idx])
            btc_skew = float(indicators["btc_skew"].iloc[today_idx])
            above_2d = bool(indicators["btc_above_sma_2d"].iloc[today_idx])
            below_2d = bool(indicators["btc_below_sma_2d"].iloc[today_idx])

            long_threshold = btc_med + 0.5 * btc_std
            short_threshold = btc_med - 0.5 * btc_std
            long_momentum = btc_ret > long_threshold
            short_momentum = btc_ret < short_threshold

            # ══════════════════════════════════════════
            # SECTION 1: BTC CONDITIONS DASHBOARD
            # ══════════════════════════════════════════
            st.markdown("####  Conditions BTC")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("BTC Close", f"${btc_close:,.0f}")
            with col2:
                st.metric("SMA(50)", f"${btc_sma:,.0f}", delta=f"{'Above' if btc_close > btc_sma else 'Below'}")
            with col3:
                st.metric("5D Return", f"{btc_ret:.2%}")
            with col4:
                st.metric("120d Skewness", f"{btc_skew:.2f}", delta="Bullish Regime" if btc_skew > 0 else "Bearish Regime", delta_color="normal" if btc_skew > 0 else "inverse")
            with col5:
                trend = "🟢 Haussier" if above_2d else ("🔴 Baissier" if below_2d else "⚪ Neutre")
                st.metric("Trend (2d)", trend)

            st.markdown("---")

            # Entry conditions with visual indicators
            st.markdown("#### Conditions d'Entrée")

            col_long, col_short = st.columns(2)

            with col_long:
                st.markdown("**🟢 LONG**")
                c1_met = long_momentum
                c2_met = above_2d
                c3_met = btc_skew > 0.15
                c4_met = bool(indicators.get("btc_vol_confirm", pd.Series(True)).iloc[-1]) if "btc_vol_confirm" in indicators else True
                st.markdown(f"{'✅' if c3_met else '❌'} Régime Distribution: Skewness > 0.15")
                st.markdown(f"{'✅' if c1_met else '❌'} 5D Ret ({btc_ret:.2%}) > Seuil ({long_threshold:.2%})")
                st.markdown(f"{'✅' if c4_met else '❌'} Volume BTC > SMA(20) Volume")
                st.markdown(f"{'✅' if c2_met else '❌'} BTC > SMA depuis 2 jours")
                if c1_met and c2_met and c3_met and c4_met:
                    st.success("**Signal LONG activé !**")
                else:
                    st.info("Signal Long inactif")

            with col_short:
                st.markdown("**🔴 SHORT**")
                c1_met_s = short_momentum
                c2_met_s = below_2d
                c3_met_s = btc_skew < -0.15
                c4_met_s = bool(indicators.get("btc_vol_confirm", pd.Series(True)).iloc[-1]) if "btc_vol_confirm" in indicators else True
                st.markdown(f"{'✅' if c3_met_s else '❌'} Régime Distribution: Skewness < -0.15")
                st.markdown(f"{'✅' if c1_met_s else '❌'} 5D Ret ({btc_ret:.2%}) < Seuil ({short_threshold:.2%})")
                st.markdown(f"{'✅' if c4_met_s else '❌'} Volume BTC > SMA(20) Volume")
                st.markdown(f"{'✅' if c2_met_s else '❌'} BTC < SMA depuis 2 jours")
                if c1_met_s and c2_met_s and c3_met_s and c4_met_s:
                    st.success("**Signal SHORT activé !**")
                else:
                    st.info("Signal Short inactif")

            st.markdown("---")

            # ══════════════════════════════════════════
            # SECTION 2: ALT HEATMAP + SELECTION
            # ══════════════════════════════════════════
            st.markdown("####  Heatmap Performance Altcoins")

            import plotly.express as px
            import plotly.graph_objects as go

            alt_btc_cols = list(indicators["alt_btc_closes"].columns)
            alt_usdt_cols = [c for c in indicators["alt_usdt_closes"].columns if c != "BTCUSDT"]

            # Build heatmap data: ALT/BTC filter status + 3D returns
            heatmap_data = []
            long_candidates = []
            short_candidates = []

            for sym in alt_btc_cols:
                usdt_sym = sym.replace("BTC", "USDT")
                base = sym.replace("BTC", "")

                ret_3d = None
                if usdt_sym in indicators["ret_3d"].columns:
                    ret_3d_val = indicators["ret_3d"].at[today, usdt_sym]
                    if pd.notna(ret_3d_val):
                        ret_3d = float(ret_3d_val)

                above_2d = False
                below_2d = False
                if sym in indicators["alt_btc_above_sma_2d"].columns:
                    above_2d = bool(indicators["alt_btc_above_sma_2d"].at[today, sym])
                if sym in indicators["alt_btc_below_sma_2d"].columns:
                    below_2d = bool(indicators["alt_btc_below_sma_2d"].at[today, sym])

                heatmap_data.append({
                    "Crypto": base,
                    "3D Return (%)": round(ret_3d * 100, 2) if ret_3d else 0,
                    "ALT/BTC > SMA (2d)": above_2d,
                    "ALT/BTC < SMA (2d)": below_2d,
                })

                if above_2d and ret_3d is not None:
                    long_candidates.append((usdt_sym, ret_3d))
                if below_2d and ret_3d is not None:
                    short_candidates.append((usdt_sym, ret_3d))

            hm_df = pd.DataFrame(heatmap_data)

            if not hm_df.empty:
                hm_df = hm_df.sort_values("3D Return (%)", ascending=False)

                # Keep top 10 long candidates (highest positive return) and top 10 short candidates (lowest negative return)
                df_long = hm_df[hm_df["ALT/BTC > SMA (2d)"]].nlargest(10, "3D Return (%)")
                df_short = hm_df[hm_df["ALT/BTC < SMA (2d)"]].nsmallest(10, "3D Return (%)")
                hm_df = pd.concat([df_long, df_short]).sort_values("3D Return (%)", ascending=False)

                # Color-coded bar chart of 3D returns
                colors = []
                for _, row in hm_df.iterrows():
                    if row["ALT/BTC > SMA (2d)"]:
                        colors.append("#00ff88")  # green = long candidate
                    else:
                        colors.append("#ff4444")  # red = short candidate

                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=hm_df["Crypto"],
                        y=hm_df["3D Return (%)"],
                        marker_color=colors,
                        text=[f"{v:+.1f}%" for v in hm_df["3D Return (%)"]],
                        textposition="outside",
                        textfont=dict(color="white", size=10),
                    )
                ])
                fig_bar.update_layout(
                    plot_bgcolor="#0a0e27",
                    paper_bgcolor="#0a0e27",
                    font_color="white",
                    xaxis_title="",
                    yaxis_title="3D Return (%)",
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Signal Preview ──
            st.markdown("---")
            st.markdown("####  Crypto Sélectionnée")

            col_star, col_skull = st.columns(2)

            with col_star:
                if long_candidates:
                    long_candidates.sort(key=lambda x: x[1], reverse=True)
                    star_sym, star_ret = long_candidates[0]
                    st.markdown(f"###  Star du moment")
                    st.markdown(f"**{star_sym}** — 3D ret: **{star_ret:+.2%}**")
                    st.markdown(f"*{len(long_candidates)} cryptos passent le filtre Long*")
                    if long_momentum and above_2d:
                        st.success("🟢 → Trade LONG actif !")
                    else:
                        st.info("Conditions BTC non remplies (pas de trade)")
                else:
                    st.markdown("###  Star du moment")
                    st.info("Aucune crypto ne passe le filtre ALT/BTC > SMA (2d)")

            with col_skull:
                if short_candidates:
                    short_candidates.sort(key=lambda x: x[1])
                    skull_sym, skull_ret = short_candidates[0]
                    st.markdown(f"###  Absente du moment")
                    st.markdown(f"**{skull_sym}** — 3D ret: **{skull_ret:+.2%}**")
                    st.markdown(f"*{len(short_candidates)} cryptos passent le filtre Short*")
                    if short_momentum and below_2d:
                        st.success("🔴 → Trade SHORT actif !")
                    else:
                        st.info("Conditions BTC non remplies (pas de trade)")
                else:
                    st.markdown("###  Absente du moment")
                    st.info("Aucune crypto ne passe le filtre ALT/BTC < SMA (2d)")

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECTION 3: PORTFOLIO STATE
        # ══════════════════════════════════════════
        st.markdown("####  État du Portefeuille")

        if os.path.exists(state_path):
            import json
            with open(state_path, "r") as f:
                state = json.load(f)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(" Cash Disponible", f"${state.get('cash', 0):,.2f}")
            with col_b:
                n_pos = len(state.get("positions", []))
                st.metric(" Positions Ouvertes", n_pos)
            with col_c:
                initial_cash = state.get('initial_cash', 10000)
                current_cash = state.get('cash', 0)
                realized_pnl = current_cash - initial_cash
                
                st.metric("P/L Réalisé", f"${realized_pnl:+,.2f}", 
                          delta=f"{(realized_pnl / initial_cash):+.2%}" if initial_cash > 0 else "0.00%")

            positions = state.get("positions", [])
            if positions:
                st.markdown("##### Positions Actives")
                pos_df = pd.DataFrame(positions)
                st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("Portefeuille non initialisé.")

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECTION 4: NAV CHART
        # ══════════════════════════════════════════
        if os.path.exists(nav_path):
            nav_df = pd.read_csv(nav_path, parse_dates=["date"])
            if not nav_df.empty:
                st.markdown("#### Évolution de la NAV")
                import plotly.graph_objects as go
                fig_nav = go.Figure()
                fig_nav.add_trace(go.Scatter(
                    x=nav_df["date"], y=nav_df["nav"],
                    mode="lines+markers", name="NAV",
                    line=dict(color="#00d4ff", width=2),
                    marker=dict(size=4)
                ))
                fig_nav.update_layout(
                    plot_bgcolor="#0a0e27", paper_bgcolor="#0a0e27",
                    font_color="white", xaxis_title="Date", yaxis_title="NAV ($)",
                    margin=dict(l=20, r=20, t=30, b=20), height=350,
                )
                st.plotly_chart(fig_nav, use_container_width=True)

        # ══════════════════════════════════════════
        # SECTION 5: TRADE HISTORY
        # ══════════════════════════════════════════
        st.markdown("####  Trades Récents")
        exec_logs_dir = os.path.join(crypto_data_dir, "execution_logs")
        if os.path.exists(exec_logs_dir):
            import glob, json
            log_files = sorted(glob.glob(os.path.join(exec_logs_dir, "*.json")))[-30:]
            all_trades = []
            for lf in log_files:
                try:
                    with open(lf, "r") as f:
                        log = json.load(f)
                    for order in log.get("orders", []):
                        all_trades.append({
                            "Date": log["date"],
                            "Symbol": order.get("symbol"),
                            "Side": order.get("side"),
                            "Qty": order.get("quantity"),
                            "Type": order.get("signal_type"),
                            "Reason": order.get("reason", "—"),
                            "Status": order.get("status"),
                        })
                except Exception:
                    continue

            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                
                # Format the table for Live Executions
                def format_status(val):
                    return 'color: #00ff00;' if str(val).lower() == 'filled' else 'color: #f7931a;'
                
                def format_side(val):
                    v = str(val).lower()
                    if "buy" in v or "long" in v: return 'color: #00ff00; font-weight: bold;'
                    elif "sell" in v or "short" in v: return 'color: #ff4444; font-weight: bold;'
                    return ''

                styled_live = trades_df.style \
                    .applymap(format_side, subset=["Side"]) \
                    .applymap(format_status, subset=["Status"])
                    
                st.dataframe(styled_live, use_container_width=True)
            else:
                st.info("Aucun trade exécuté pour le moment.")
        else:
            st.info("Aucun log d'exécution trouvé.")
