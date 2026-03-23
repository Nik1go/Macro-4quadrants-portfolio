"""
Page 4: Méthodologie & Synthèse Technique
Documentation technique à destination des investisseurs, quants et recruteurs.
"""

import streamlit as st

def render(data):
    st.title("Méthodologie & Architecture")
    
    st.markdown("""
    Bienvenue dans la section technique de ce portfolio dynamique. Ce projet repose sur la détection algorithmique de régimes macroéconomiques à l'aide de modèles de **Machine Learning (Classification Binaire)**.
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
            image_path = os.path.join(os.path.dirname(current_dir), "images", "architecture.png")
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
        5. **Simulation & Trading (Spark/IBKR)** : Backtesting de la stratégie avec prise en compte des coûts de transaction, suivi de l'exécution automatique.
        """)
        
    with st.expander("2. Cadre Macroéconomique (Feature Engineering)"):
        st.markdown("""
        Pour définir la position économique (le "Quadrant"), j'utilise un cadre bifactoriel : **Croissance** et **Inflation**.

        **Axe Croissance (Mesure du Risk-On / Risk-Off) :**
        Capture la dynamique de l'économie réelle et le sentiment de marché.
        - *Indicateurs positifs* : Housing Permits, Industrial Production, Consumer Sentiment, Copper, spread 10-2Y Yield Curve.
        - *Indicateurs négatifs* : Initial Claims, High Yield Spread (spread de crédit risqué vs OAT), VIX, Real Rates.

        **Axe Inflation :**
        Capture la dynamique des prix et les anticipations du marché.
        - *Indicateurs positifs* : CPI, 10Y Breakeven Inflation Rate (T10YIE - anticipation de marché), WTI Oil Prices.
        - *Indicateurs négatifs* : US Dollar Index (un dollar fort étant historiquement déflationniste).

        **Transformations & Normalisation :**
        - **Z-Scores** : Application de scores standardisés glissants (expanding z-scores) pour normaliser les données et capter les changements de régime.
        - **Momentum (1MoM-3MoM-6MoM selon la rapidité de l'indicateur) & Volatilité** : Calcul de la dynamique à 3 mois et de la volatilité historique pour lisser le bruit et dégager des signaux clairs.
        """)
        st.latex(r'''
        Score_{position} = \frac{X_t - \mu_{expanding}}{\sigma_{expanding}}
        ''')
        st.info("💡 Je vous invite à consulter la page **'Correlations'** pour observer la matrice de corrélation exploratoire de ces indicateurs et valider la solidité des indicateurs retenus.")

    with st.expander("3. Modélisation Machine Learning"):
        st.markdown("""
        **Approche Algorithmique :** Classification binaire séparée pour le Risque (Croissance) et l'Inflation en utilisant des algorithmes d'ensemble (**Random Forest Classifier**).

        **Cibles (Targets du modèle) :**
        La définition des régimes s'appuie sur le momentum des prix de marché (pricing) pour capturer les conditions de liquidité et le consensus des investisseurs, qui anticipent la macroéconomie réelle de 6 à 9 mois :
        - `TARGET_RISK_CLASS = 1` (Risk-On / Croissance) : si la Moyenne Mobile 1 mois du **High Yield Spread** < Moyenne Mobile 3 mois (les spreads de crédit se resserrent).
        - `TARGET_INFLATION_CLASS = 1` (Reflation) : si la Moyenne Mobile 1 mois du **10Y Breakeven Inflation** > Moyenne Mobile 3 mois (les anticipations d'inflation montent).

        **Validation & Évaluation :**
        - **Walk-Forward Validation** : Le modèle subit un "backtest ML" glissant annuel (entraînement sur les données historiques $T-n$, test sur l'année $T$), garantissant l'absence temporelle de biais de présentation (look-ahead bias).
        - **Prédictions (Lissage)** : Les probabilités brutes (`predict_proba()`) sont lissées avec une Moyenne Mobile Exponentielle (EMA span=5) avant d'être classées par un seuil de décision de $0.5$.
        
        """)
        st.info("📊 Les métriques détaillées d'entraînement (Accuracy, Precision, Recall, Matrice de Confusion) sont disponibles et analysables sur la page **'ML Performance'**.")

    with st.expander("4. Stratégie d'Allocation"):
        st.markdown("""
        L'actif final est un portefeuille systématiquement réalloué au premier jour de chaque mois selon les prédictions macro. La logique fondamentale d'allocation s'inspire du modèle "All Weather" mais de manière dynamique et directionnelle.

        | Quadrant | Logique Fondamentale | SP500 | NASDAQ | SmallCAP | GOLD | COMMODITIES | TREASURY |
        |----------|-----------------------|-------|--------|----------|------|-------------|----------|
        | **Q1: Growth** | L'économie est forte, l'inflation est contenue (Goldilocks). Maximal Risk-On. | 30% | **40%** | **30%** | 0% | 0% | 0% |
        | **Q2: Inflation** | Surchauffe (Reflation). Les actifs tangibles et matières premières performent. | **40%** | 10% | 0% | **30%** | **20%** | 0% |
        | **Q3: Stagflation** | Croissance faible, forte inflation. Hausse des taux, cash et refuges privilégiés. | 0% | 0% | 0% | **60%** | **20%** | **20%** |
        | **Q4: Deflation** | Choc déflationniste / Récession. Les obligations (Treasuries) jouent leur rôle de valeur refuge. | 0% | 0% | 0% | **40%** | 0% | **60%** |
        """)
        st.info("📈 Les résultats de ces pondérations confrontées au marché (rendements composés, Max Drawdown, Ratio de Sharpe) sont consultables dans la page **'Backtest'**.")

    with st.expander("5. Limitations du Modèle (Real-World Constraints)"):
        st.markdown("""
        Afin de rester le plus réaliste possible face au terrain, le projet a été confronté et adapté à plusieurs limites inhérentes à la data financière et au ML :
        
        *   **Disponibilité et Révisions des Données Macro (FRED API)** : Les données macroéconomiques ne sont pas diffusées en *live*. Pour contrer ce phénomène (Look-Ahead Bias), le pipeline implémente des **lags stricts** : il simule artificiellement des retards de publication réalistes avant de fournir les caractéristiques au modèle en phase de test.
        *   **Absence de Données Consensus** : Contrairement aux places institutionnelles, l'API ne permet pas de capturer facilement le consensus de marché sur les parutions économiques, limitant ainsi la "surprise factor".
        *   **Limites du Machine Learning** : Les algorithmes d'apprentissage reposent sur la base que le passé rime avec le futur. Les chocs exogènes imprévisibles (guerre, pandémie non structurelle) restent difficiles à modéliser sans une composante de NLP (Sentiment Analysis quantitatif - potentiellement, une V2 de ce projet).
        """)
