"""Strategy explanation view - Macro style."""

from __future__ import annotations
from pathlib import Path
import streamlit as st

def render_strategy_explanation() -> None:
    """Render architecture and strategy explanations."""
    
    st.header("Mécanique d'Arbitrage")

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Flux de Données & Calcul")
        st.markdown("""
        1. **Ingestion :** Collecte des prix Spot via Binance et des carnets d'ordres via Polymarket v2.
        2. **Modélisation :** Calcul des d1/d2 Black-Scholes pour options binaires afin d'obtenir une probabilité théorique.
        3. **Détection :** Identification des écarts entre la probabilité théorique et le prix du carnet d'ordres (Edge).
        4. **Nettoyage :** Déduction des frais (Matic + Binance + Poly) pour obtenir le **Spread Net**.
        5. **Sizing (Kelly) :** Calcul de la fraction optimale du capital à investir :
           $$f^* = \\frac{\\mu - r}{\\sigma^2}$$
           *Où $\\mu$ est l'espérance de gain, $r$ le taux sans risque, et $\\sigma$ la volatilité de l'actif spécifique (XRP, BTC, etc.).*
        """)
        
        st.subheader("Gestion des Risques")
        st.markdown("""
        - **Fraction de Kelly (1/4) :** On utilise une version "Fractionnaire" pour éviter la sur-exposition.
        - **Volatilité par Actif :** La taille est inversement proportionnelle au carré de la volatilité de la crypto traitée.
        - **Concentration Caps :** Limite max de 35% du capital par actif pour la diversification.
        - **Circuit Breaker :** Arrêt auto si le drawdown global (Equity totale) dépasse 50%.
        """)

    with col2:
        repo_root = Path(__file__).resolve().parents[3]
        architecture_image = repo_root / "polymarket_arbitrage" / "streamlit" / "assets" / "architecture_diagram.png"

        if architecture_image.exists():
            st.image(str(architecture_image), caption="Architecture du bot d'arbitrage", use_container_width=True)
        else:
            st.info("Diagramme d'architecture non trouvé (polymarket_arbitrage/streamlit/assets/architecture_diagram.png).")

    st.divider()
    
    st.subheader("Variantes Stratégiques")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Delta-Neutral")
        st.markdown("""
        Achat d'options Yes/No sur Polymarket contre une couverture 1:1 sur les contrats perpétuels Binance.
        **Objectif :** Générer un profit pur sur la convergence des prix sans dépendance à la direction du marché.
        """)

    with col_b:
        st.markdown("#### Directional Pricing")
        st.markdown("""
        Achat direct d'options Yes/No basé sur une supériorité prédictive du modèle théorique par rapport au marché.
        **Objectif :** Maximiser le rendement sur les inefficacités de probabilités pures.
        """)
