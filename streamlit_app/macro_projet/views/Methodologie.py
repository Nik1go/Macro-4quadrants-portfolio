"""
Page 4: Méthodologie & Synthèse Technique
Documentation technique à destination des investisseurs, quants et recruteurs.
"""

import streamlit as st

def render(data):
    st.title("Méthodologie & Architecture")
    
    st.markdown("""
    Ce projet repose sur la détection algorithmique de régimes macroéconomiques à l'aide d modèles de **Machine Learning (Classification Binaire)**.
    L'objectif est d'allouer dynamiquement le capital sur 4 quadrants macroéconomiques (Goldilocks, Reflation, Stagflation, Déflation) afin d'optimiser le ratio de Sharpe face aux stratégies passives (Buy & Hold).
    """)

    with st.expander("1. Architecture Technique (Data Engineering Pipeline)", expanded=True):
        st.markdown("""
        L'ensemble du pipeline est entièrement automatisé et executé chaque jours à 16H30 UTC+1 par **Apache Airflow**, via **Apache Spark**, de l'ingestion de la donnée jusqu'à l'exécution d'ordres de trading via l'API d'Interactive Brokers.
        """)
        
        try:
            import os
            from PIL import Image
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # current_dir is streamlit_app/macro_projet/views
            # project_root is streamlit_app/
            project_root = os.path.dirname(os.path.dirname(current_dir))
            image_path = os.path.join(project_root, "images", "architecture.png")
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
        
    with st.expander("2. Cadre Macroéconomique (Feature Engineering)"):
        st.markdown("""
        Pour définir la position économique les **Quadrants**, j'utilise un cadre bifactoriel : **Croissance** et **Inflation**.

        **L'Axe Croissance mesure la probabilité d'un régime Risk-On / Risk-Off**.
        On utilise comme target le **High Yield Bond Spread** (différence entre les taux d'obligations d'entreprises risquées et les taux sans risque de référence).
        Il représente, selon moi, un excellent proxy de marché de la croissance et notamment de l'appétit des investisseurs pour les actifs risqués (risk-on / risk-off).

        **L'Axe Inflation mesure la dynamique des prix et le risque de surchauffe économique**.
        On utilise comme target le **10Y Breakeven Inflation Rate** (la différence de rendement entre les Bons du Trésor classiques à 10 ans et les TIPS indexés sur l'inflation).
        Contrairement à l'Indice des Prix à la Consommation (CPI) qui est publié mensuellement avec du retard (lag) et souvent révisé, le Breakeven reflète les anticipations d'inflation "pricées" en temps réel par les investisseurs obligataires. C'est un excellent proxy car c'est le signal de marché le plus pur et réactif pour capter une tendance inflationniste (Reflation) ou déflationniste avant même les statistiques officielles.
        
        **Pour entraîner le modèle**, j'ai décidé d'utiliser un ensemble de features (indicateurs) complémentaires, soigneusement sélectionnés pour éviter le "bruit" et la malédiction de la dimensionnalité :
        
        **Les indicateurs de marché (réactifs) :**
        - Matières premières : **Copper** (Momentum 3M - Demande industrielle globale), **WTI Crude Oil** (Niveau et Momentum 3M - Choc énergétique).
        - Taux et Conditions Financières : **Yield Curve 10-2Y** (Pente de la courbe), **VIX** (Stress du marché), **NFCI** (Conditions financières et bancaires de Chicago).
        - Devises : **DXY** (Momentum 3M - Tendance de la liquidité globale).
        *(Note : Le High Yield Spread et le 10Y Breakeven agissent comme cibles principales, mais ils sont aussi utilisés de manière croisée ("cross-confirmation") comme features : le Spread aide à prédire l'Inflation, et le Breakeven aide à prédire le Risque. Ils ne sont jamais utilisés pour se prédire eux-mêmes afin d'éviter toute fuite de données).*

        **Les indicateurs macroéconomiques (structurels) :**
        Ils sont intégrés pour que le modèle trouve des corrélations de fond et détecte avec moins de bruit les véritables changements de régime économique :
        - Économie et Emploi : **Initial Claims** (Demandes de chômage en temps réel), **Industrial Production** (Production industrielle brute).
        - Consommation et Immobilier : **Consumer Sentiment** (Confiance des consommateurs), **Housing Permits** (Demandes de permis de construire).
        - Monétaire et Inflation : **Inflation YoY** (Tendance structurelle), **Net Liquidity** (Soutien en liquidité des Banques Centrales), **Real Rates** (Véritable coût du capital).

        **Détermination du Régime (Lissage des Probabilités sur 5 jours) :**
        Pour éviter des allers-retours incessants dans le portefeuille à chaque bruit statistique du marché, le quadrant final n'est pas choisi sur une seule journée isolée. 
        Les prédictions brutes de chaque régime émanant du modèle sont lissées via une Moyenne Mobile Exponentielle (EMA) sur **5 jours**. L'algorithme exige ainsi qu'une convergence probabiliste s'installe fermement (une confirmation sur une semaine entière) avant d'activer officiellement un basculement de régime macroéconomique.

        **Gestion des données (Look-Ahead Bias) :**
        Les données proviennent de Yahoo Finance et de FRED. Contrairement au marché actions, les données macroéconomiques (FRED) sont publiées avec du retard (ex : l'inflation de février est annoncée mi-mars). 
        J'ai donc appliqué des **lags de publication** (+30 jours pour le CPI, +5 jours pour le chômage...) afin d'éviter d'entraîner le modèle sur des données du futur.

        **Transformations & Normalisation :**
        - **Z-Scores** : Application de scores standardisés glissants (expanding z-scores) pour normaliser les données et capter les changements de régime.
        - **Momentum (3MoM-6MoM) & Volatilité** : Calcul de la dynamique à 3 mois et de la volatilité historique pour lisser le bruit et dégager des signaux clairs.
      
        """)
        st.latex(r'''
        Score_{position} = \frac{X_t - \mu_{expanding}}{\sigma_{expanding}}
        ''')
        st.info(" Je vous invite par la suite à consulter la page **'Correlations'** pour observer la matrice de corrélation exploratoire de ces indicateurs et valider la solidité des indicateurs retenus.")

    with st.expander("3. Modélisation Machine Learning"):
        st.markdown("""
        **Approche Algorithmique :** Classification binaire séparée pour le Risque (Croissance) et l'Inflation en utilisant des algorithmes d'ensemble (**Random Forest Classifier**).

        **Choix des Cibles (Targets) :**
        Pour piloter l'allocation, j'ai sélectionné deux actifs de marché extrêmement réactifs qui agissent comme *proxies* pour la Croissance et l'Inflation. Cela permet d'éviter l'utilisation de données macroéconomiques classiques (PIB, CPI) souvent des données mensuelles/trimestrielles retardées et sujettes à de fortes révisions ( révision non disponible avec FRED).
        - **Cible Risque (Proxy Croissance)** : Le **High Yield Bond Spread**. Cet indicateur mesure la prime de risque exigée pour prêter aux entreprises fragiles. C'est un baromètre direct du stress financier et de la confiance des marchés dans l'économie.
        - **Cible Taux (Proxy Inflation)** : Le **10Y Breakeven Inflation Rate**. Cet actif représente l'inflation "pricée" en temps réel par les investisseurs obligataires. C'est le signal le plus pur pour capter la tendance inflationniste bien avant les annonces officielles.

        **Logique de Tendance (Croisement de Moyennes Mobiles - SMA) :**
        Plutôt que d'utiliser une valeur absolue ou une médiane historique fixe, les régimes sont définis par le momentum. Le modèle compare le court terme (1 mois) à la tendance de fond (3 mois) pour identifier les points de basculement :
        - `TARGET_RISK_CLASS = 1` (Risk-On) : si `SMA_1M < SMA_3M` du **High Yield Spread**, 0 sinon. Un spread qui baisse à court terme signale une détente des conditions de crédit, propice aux actifs risqués.
        - `TARGET_INFLATION_CLASS = 1` (Reflation) : si `SMA_1M > SMA_3M` du **10Y Breakeven**, 0 sinon. Un breakeven qui monte à court terme indique une accélération soudaine des anticipations d'inflation.

        **Validation & Évaluation :**
        - **Walk-Forward Validation** : Le modèle subit un "backtest ML" glissant annuel (entraînement sur les données historiques $T-n$, test sur l'année $T$), garantissant l'absence temporelle de biais de présentation (look-ahead bias).
        - **Prédictions (Lissage)** : Les probabilités brutes (`predict_proba()`) sont lissées avec une Moyenne Mobile Exponentielle (EMA span=5) avant d'être classées par un seuil de décision de $0.5$.
        
        """)
        st.info(" Les métriques détaillées d'entraînement (Accuracy, Precision, Recall, Matrice de Confusion) sont disponibles et analysables sur la page **'ML Performance'**.")

    with st.expander("4. Stratégie d'Allocation"):
        st.markdown("""
        L'actif final est un portefeuille avec une gestion des ordres et de la réallocation **quotidienne**, pilotée à la fois par les probabilités macroéconomiques (lissées par une EMA de 5 jours) et un suivi de tendance (Trend Following). 
        La logique fondamentale d'allocation s'inspire de l'approche "All Weather" mais de manière nettement plus dynamique et réactive aux signaux de marché.

        **1. Allocation de base par Régime Macro :**

        L'allocation a été optimisée pour maximiser le rendement et la protection selon la dynamique de chaque régime :

        *   **Q1: Growth (Croissance Saine / Goldilocks)** : Maximal Risk-On sur les actions et l'immobilier.
            *   **S&P 500** : 40% | **NASDAQ 100** : 40% | **US REIT (VNQ)** : 20%
        *   **Q2: Inflation (Reflation)** : Phase d'expansion où les prix augmentent. Protection contre l'inflation avec maintien d'une exposition actions forte (Tech/Croissance).
            *   **Or (GOLD)** : 40% | **NASDAQ 100** : 38% | **Matières Premières** : 11% | **S&P 500** : 11%
        *   **Q3: Stagflation** : Défense. Baisse de la croissance et hausse des prix. Rotation vers les refuges monétaires et pari à la baisse sur les actions.
            *   **USD/JPY** : 40% | **Short S&P 500** : 32% | **USD/EUR** : 18% | **Matières Premières** : 10%
        *   **Q4: Deflation** : Le Bunker (Crash Déflationniste). Risk-OFF total, protection massive via la dette d'État.
            *   **Treasury 10Y** : 40% | **Or (GOLD)** : 38% | **Obligations (IG)** : 22%
            
        **2. Stratégie "High Conviction" (Levier 2x) :**
        
        Une variante algorithmique "High Conviction" (HC) est disponible dans le backtest. Cette stratégie applique un **effet de levier 2x** sur le portefeuille. Un coût d'emprunt (borrowing cost) de 5% annualisé est appliqué dynamiquement, uniquement sur les jours de trading actifs, pour refléter la réalité d'un compte sur marge.

        **3. Filtre de Tendance et Mécanisme Risk-Off (Trend Following Overlay) :**
        
        Afin de limiter les drawdowns extrêmes lors de krachs boursiers subits, un filtre de suivi de tendance (**MA200**) est superposé en temps réel aux allocations ci-dessus :
        - Les actifs majeurs (**S&P 500, NASDAQ 100 et Or**) sont constamment surveillés par rapport à leur Moyenne Mobile à 200 jours.
        - **Déclenchement Risk-Off :** Si un de ces actifs clôture sous sa MA200 pendant **5 jours consécutifs**, l'intégralité de son allocation prévue est coupée et automatiquement transférée vers la sécurité des **Bons du Trésor à 10 ans (Treasuries)**.
        - **Reprise Risk-On :** La position est restaurée soit lorsque l'actif repasse au-dessus de sa MA200 pendant 5 jours, soit lors d'un grand basculement de quadrant macroéconomique.
        """)
        st.info(" Les résultats de ces pondérations confrontées au marché (rendements composés, Max Drawdown, Ratio de Sharpe) sont consultables dans la page **'Backtest'**.")

    with st.expander("5. Mesure de Robustesse et Confiance Statistique (Bootstrap)"):
        st.markdown("""
        Pour s'assurer que les performances de notre modèle ne sont pas de simples anomalies (overfitting sur des valeurs extrêmes isolées), l'application propose plusieurs métriques et un test de confiance par Bootstrap :
        Pour la construction de l'allocations, le modèle s'appuie sur la confiance statistique des performances des actifs en fonction des quadrants.
        **Les Métriques de Performance :**
        - **Sharpe Ratio** : Évalue le rendement ajusté de la volatilité totale. 
        - **Sortino Ratio** : Variante du Sharpe qui ne pénalise que la volatilité *à la baisse* (Downside Deviation). 
        - **Win Rate (%)** : Probabilité qu'une journée soit positive. 

        **La Confiance Statistique (Block Bootstrap) :**
        Au lieu de tester la stratégie sur de l'aléatoire avec un montecarlo par exemple (ce qui n'aurait pas de sens économique avec un model macros comme celui-ci), l'application teste la **fiabilité des rendements**.
        Pour un actif donné dans un quadrant précis (ex: l'Or en Q4), le système :
        1. Isole tous les rendements quotidiens réels de l'actif durant ce régime.
        2. Tire aléatoirement 500 échantillons distincts de cette série de rendements (avec remise).
        3. Calcule la métrique (ex: Sortino) pour ces 500 scénarios ainsi que l'écart-type des rendements des échantillons.
        4. Détermine le **Niveau de Confiance** (ex: 95%) : pourcentage des échantillons confirmant la même polarité (gain ou perte) que la performance historique observée sur cet actif.
        
        *Plus la confiance est proche de 100%, plus la performance observée (qu'elle soit positive ou négative) est robuste et non due à quelques jours d'anomalies statistiques extrêmes isolés.*
        """)

    with st.expander("6. Limitations du Modèle (Real-World Constraints)"):
        st.markdown("""
        Afin de rester le plus réaliste possible face au terrain, le projet a été confronté et adapté à plusieurs limites inhérentes à la data financière et au ML :
        
        *   **Disponibilité et Révisions des Données Macro (FRED API)** : Les données macroéconomiques ne sont pas diffusées en *live*. Pour contrer ce phénomène (Look-Ahead Bias), le pipeline implémente des **lags stricts** : il simule artificiellement des retards de publication réalistes avant de fournir les caractéristiques au modèle en phase de test.
        *   **Absence de Données Consensus** : Contrairement aux places institutionnelles, l'API ne permet pas de capturer facilement le consensus de marché sur les parutions économiques, limitant ainsi la "surprise factor".
        *   **Limites du Machine Learning** : Les algorithmes d'apprentissage reposent sur la base que le passé rime avec le futur. Les chocs exogènes imprévisibles (guerre, pandémie non structurelle) restent difficiles à modéliser sans une composante de NLP (Sentiment Analysis quantitatif - potentiellement, une V2 de ce projet).
        """)
