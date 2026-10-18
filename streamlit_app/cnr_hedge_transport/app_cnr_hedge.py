from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest_ui import render_backtest
from .contract import MINI_BRENT_BARRELS, simulate_contract
from .study import lag_regressions, load_fortnightly_study, load_official_study, load_study_csv
from .weights import (
    CONVOY_LABELS,
    SOURCE_URL,
    history_for_convoy,
    weight_for_contract,
)


ASSUMPTIONS = [

    {
            "Bloc": "Exposition du chargeur",
            "Hypothese": "La variable à couvrir est le supplément de facture de transport, pas une quantité de gazole achetée directement.",
            "Role": "Le type de convoi, le prix initial, la pondération CNR et les dates du contrat et du transport déterminent le montant exposé.",
            "Statut": "Variable du Contrat",
        },
        {
            "Bloc": "Pondération gazole CNR",
            "Hypothese": "La part du gazole selon le type de convoi et le millésime est déjà chargée depuis les données publiées par le CNR.",
            "Role": "Fournit le coefficient carburant vérifié pour calculer le pied de facture dans le scénario CNR.",
            "Statut": "data CNR",
        },
        {
            "Bloc": "Modalités de facturation",
            "Hypothese": "L'indice, la date de base et la fréquence de révision réellement inscrits au contrat peuvent différer du scénario CNR standard.",
            "Role": "Permet d'appliquer la formule réellement facturée au chargeur lorsqu'un contrat est disponible.",
            "Statut": "Variable du Contrat",
        },
    {
        "Bloc": "Brent : données et contrat",
        "Hypothese": "Le questionnaire illustre le mini Brent MBC de 100 barils ; le backtest suit des échéances ICE avec quantités fractionnaires théoriques.",
        "Role": "BZ=F converti en euros sert à estimer β et au questionnaire. Le backtest calcule séparément le P&L des futures ICE du classeur par échéance.",
        "Statut": "Simulation en MBC",
    },
    {
        "Bloc": "Devise du hedge",
        "Hypothese": "Le Brent est coté en USD, mais aucune couverture EUR/USD séparée n'est mise en place à ce stade.",
        "Role": "Le backtest convertit en EUR le P&L quotidien ICE au taux choisi ; aucune position de change autonome n'est calibrée.",
        "Statut": "A mesurer",
    },
    {
        "Bloc": "Roll des mini-futures Brent",
        "Hypothese": "La couverture envisagée utilise des mini-futures Brent successifs de 100 barils. Le passage entre échéances et la liquidité doivent être mesurés.",
        "Role": "le P&L du roll exige le suivi des contrats détenus (calcul Pnl Backwardation / contango)",
        "Statut": "À mesurer",
    },

]


ROADMAP = pd.DataFrame(
    [
        [
            "1 · Mesurer l'exposition transport", "Prototype",
            "Indice CNR mensuel, part gazole par convoi et facture simulée sur 12 mois.",
            "Renseigner les factures et les clauses de contrats de transport réels.",
        ],
        [
            "2 · Estimer le lien CNR / Brent", "Calculé",
            "Décalage et sensibilité β estimés sur les données antérieures au scénario.",
            "Tester la stabilité de β sur d'autres périodes, hors échantillon.",
        ],
        [
            "3 · Dimensionner la couverture", "Prototype",
            "Formule des mini-contrats, arrondi, position mensuelle et mois sans contrat.",
            "Confronter la taille proposée aux clauses réelles et aux contraintes de négociation.",
        ],
        [
            "4 · Lire le résultat mensuel", "Estimation",
            "Surcoût de facture, P&L estimé de la couverture Brent et P&L total vs budget.",
            "Recalculer le P&L avec les prix des futures réellement détenus.",
        ],
        [
            "5 · Obtenir les données du mini Brent", "À collecter",
            "La simulation utilise actuellement une série Brent continue en euros.",
            "Collecter les prix MBC par échéance et date, le change, les volumes et l'intérêt ouvert.",
        ],
        [
            "6 · Backtester les contrats futures", "Prototype ICE",
            "P&L quotidien par échéance ICE fixe, conversion EUR et coût résiduel par facture.",
            "Ajouter prix de règlement vérifiés, frais, liquidité et appels de marge.",
        ],
        [
            "7 · Valider l'efficacité du hedge", "À faire",
            "Le résultat présenté aujourd'hui est celui d'un scénario simulé.",
            "Comparer avec et sans couverture, hors échantillon : coût net, risque résiduel et liquidité.",
        ],
    ],
    columns=["Étape", "État actuel", "Disponible aujourd'hui", "Prochaine action concrète"],
)


PALETTE = {
    "cyan": "#00d4ff",
    "blue": "#2f80ed",
    "green": "#17c964",
    "orange": "#ffb020",
    "red": "#ff5c7a",
    "purple": "#9d86e0",
    "border": "#3d4263",
}


def _fr_number(value: float, decimals: int = 0, signed: bool = False) -> str:
    """Format a scenario amount with French separators."""
    sign = "+" if signed else ""
    return f"{value:{sign},.{decimals}f}".replace(",", "\u202f").replace(".", ",")


def _mini_contract_label(count: int) -> str:
    return f"{count} mini-contrat{'s' if count != 1 else ''}"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Page palette --------------------------------------------------- */
        :root {
            --cnr-bg: #0a0e27;
            --cnr-surface: #151a33;
            --cnr-surface-raised: #1b2140;
            --cnr-border: #3d4263;
            --cnr-text: #f4f6ff;
            --cnr-muted: #b8c1e3;
            --cnr-cyan: #43ddff;
            --cnr-orange: #ffcf75;
        }

        /* The portfolio theme targets every div/span. Restore readable
           colours on native Streamlit controls used by this page. */
        div[data-baseweb="select"] > div,
        .stDateInput div[data-baseweb="input"] {
            background: var(--cnr-surface) !important;
            border-color: var(--cnr-border) !important;
            color: var(--cnr-text) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        .stDateInput input,
        .stDateInput button,
        .stDateInput svg {
            color: var(--cnr-text) !important;
            fill: var(--cnr-text) !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        ul[data-baseweb="menu"],
        [role="listbox"] {
            background: var(--cnr-surface-raised) !important;
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li span,
        ul[data-baseweb="menu"] li,
        ul[data-baseweb="menu"] li span,
        [role="option"] {
            color: var(--cnr-text) !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li[aria-selected="true"],
        ul[data-baseweb="menu"] li:hover,
        ul[data-baseweb="menu"] li[aria-selected="true"],
        [role="option"][aria-selected="true"],
        [role="option"]:hover {
            background: #283052 !important;
        }

        /* Match the Macro project's prominent, separated tab navigation. */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0 !important;
            border-bottom: 2px solid var(--cnr-border) !important;
            padding-bottom: 0 !important;
            width: 100% !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--cnr-cyan) !important;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            min-height: 68px !important;
            justify-content: center !important;
            font-size: clamp(0.8rem, 1.15vw, 1rem) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.2px !important;
            padding: 10px 6px !important;
            color: #8890b5 !important;
            border-right: 2px solid var(--cnr-border) !important;
            white-space: normal !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
            font: inherit !important;
            line-height: 1.25 !important;
            text-align: center;
            white-space: normal !important;
        }
        .stTabs [data-baseweb="tab"]:last-child {
            border-right: none !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--cnr-cyan) !important;
            background: rgba(0, 212, 255, 0.08) !important;
            border-bottom: 3px solid var(--cnr-cyan) !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--cnr-cyan) !important;
            background: rgba(0, 212, 255, 0.05) !important;
        }

        /* Long KPI values must wrap instead of being replaced by an ellipsis. */
        div[data-testid="stMetric"] {
            min-width: 0;
            min-height: 116px;
            padding: 14px 16px;
            border: 1px solid var(--cnr-border);
            border-radius: 10px;
            background: var(--cnr-surface);
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
            color: #dbe2ff !important;
            opacity: 1 !important;
            font-weight: 650;
        }
        div[data-testid="stMetricValue"] {
            color: var(--cnr-text) !important;
            font-size: clamp(1.3rem, 2vw, 2rem) !important;
            line-height: 1.16 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }
        div[data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        .cnr-hero {
            border: 1px solid var(--cnr-border);
            border-radius: 12px;
            padding: 22px 24px;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.10), rgba(157, 134, 224, 0.08));
            margin-bottom: 18px;
        }
        .cnr-hero h3 { color: var(--cnr-cyan) !important; }
        .cnr-hero p { margin-bottom: 0; color: var(--cnr-text) !important; }
        .cnr-pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(255, 176, 32, 0.14);
            border: 1px solid rgba(255, 176, 32, 0.35);
            color: var(--cnr-orange) !important;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .cnr-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            align-items: stretch;
            gap: 18px;
            margin: 8px 0 24px;
        }
        .cnr-card {
            box-sizing: border-box;
            border: 1px solid var(--cnr-border);
            background: var(--cnr-surface);
            border-radius: 10px;
            padding: 20px 22px;
            min-width: 0;
            height: 100%;
        }
        .cnr-card h4 {
            color: var(--cnr-cyan) !important;
            margin: 10px 0 10px;
            overflow-wrap: anywhere;
        }
        .cnr-card p {
            color: var(--cnr-text) !important;
            margin: 0 0 10px;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }
        .cnr-tag {
            color: var(--cnr-orange) !important;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .cnr-card .cnr-muted,
        .cnr-card .cnr-muted strong { color: var(--cnr-muted) !important; }
        .cnr-step-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            align-items: stretch;
            gap: 16px;
            margin: 8px 0 24px;
        }
        .cnr-step {
            box-sizing: border-box;
            border: 1px solid var(--cnr-border);
            border-top: 4px solid var(--cnr-cyan);
            background: var(--cnr-surface);
            border-radius: 10px;
            padding: 18px;
            min-height: 180px;
            height: 100%;
        }
        .cnr-step-number { color: var(--cnr-orange) !important; font-size: 0.82rem; font-weight: 700; }
        .cnr-step h4 { color: var(--cnr-cyan) !important; margin: 8px 0; overflow-wrap: anywhere; }
        .cnr-step p { color: var(--cnr-text) !important; line-height: 1.5; }

        /* Plotly inherits conflicting colours from the portfolio theme. */
        .js-plotly-plot .modebar-btn path { fill: var(--cnr-muted) !important; }
        .js-plotly-plot .modebar-btn:hover path { fill: var(--cnr-cyan) !important; }
        .js-plotly-plot .legendtext,
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .g-xtitle text,
        .js-plotly-plot .g-ytitle text { fill: var(--cnr-text) !important; }

        /* This variance chart has a white background: keep its axes legible. */
        .st-key-cnr_variance_chart .js-plotly-plot .xtick text,
        .st-key-cnr_variance_chart .js-plotly-plot .ytick text,
        .st-key-cnr_variance_chart .js-plotly-plot .g-xtitle text,
        .st-key-cnr_variance_chart .js-plotly-plot .g-ytitle text {
            fill: #24324b !important;
        }

        @media (max-width: 1100px) {
            .cnr-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .cnr-step-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 700px) {
            .cnr-card-grid,
            .cnr-step-grid { grid-template-columns: 1fr; }
            .cnr-hero { padding: 18px; }
            div[data-testid="stMetric"] { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_assumption_cards() -> None:
    cards = "".join(
        f'<article class="cnr-card">'
        f'<span class="cnr-tag">{item["Statut"]}</span>'
        f'<h4>{item["Bloc"]}</h4>'
        f'<p>{item["Hypothese"]}</p>'
        f'<p class="cnr-muted"><strong>Rôle :</strong> {item["Role"]}</p>'
        f'</article>'
        for item in ASSUMPTIONS
    )
    st.markdown(
        f'<section class="cnr-card-grid">{cards}</section>',
        unsafe_allow_html=True,
    )


def _build_cnr_weight_chart(convoy_type: str, selected) -> go.Figure:
    history = history_for_convoy(convoy_type)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[item.effective_date for item in history],
            y=[item.diesel_weight_pct for item in history],
            mode="lines+markers",
            line=dict(color=PALETTE["cyan"], width=3, shape="hv"),
            marker=dict(size=8),
            name="Pondération gazole",
            hovertemplate="Date d'effet : %{x|%d/%m/%Y}<br>Pondération : %{y:.1f} %<extra></extra>",
        )
    )
    if selected is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected.effective_date],
                y=[selected.diesel_weight_pct],
                mode="markers",
                marker=dict(color=PALETTE["orange"], size=15, symbol="diamond"),
                name="Valeur du contrat",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=410,
        yaxis_title="Pondération gazole (%)",
        xaxis_title="Date d'effet CNR",
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f4f6ff", size=13),
        legend=dict(font=dict(color="#f4f6ff", size=13), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            color="#f4f6ff",
            gridcolor="rgba(61,66,99,0.65)",
            title_font=dict(color="#f4f6ff"),
            tickfont=dict(color="#f4f6ff"),
        ),
        yaxis=dict(
            color="#f4f6ff",
            gridcolor="rgba(61,66,99,0.65)",
            title_font=dict(color="#f4f6ff"),
            tickfont=dict(color="#f4f6ff"),
        ),
        hovermode="x unified",
    )
    return fig



def _render_fortnightly_study() -> None:
    st.markdown("### Quinzaines CNR : analyse exploratoire")
    levels = load_fortnightly_study()
    results, aligned = lag_regressions(
        levels, max_lag=2, lag_label="Décalage (quinzaines)", min_observations=6
    )
    st.markdown(
        "Les valeurs CNR couvrent les jours **1 à 15** et **16 à la fin du mois**. "
        "Le Brent en EUR est la moyenne des clôtures quotidiennes communes "
        "BZ=F et EURUSD=X sur les mêmes jours. Un décalage vaut une quinzaine."
    )
    st.warning(
        f"Historique disponible : {len(levels)} niveaux, de {levels.index.min()} à {levels.index.max()}, "
        f"soit seulement {len(aligned)} variations communes pour tester 0 à 2 décalages. "
        "Ces régressions descriptives ne permettent pas de conclure à une précision "
        "supérieure ni de recalibrer la couverture. Le choix du meilleur shift sur ce petit échantillon accroît le risque de surajustement."
    )
    st.dataframe(results.round(3), width="stretch", hide_index=True)
    fig = go.Figure(go.Bar(
        x=results["Décalage (quinzaines)"], y=results["R²"],
        marker_color=PALETTE["cyan"],
    ))
    fig.update_layout(
        title=f"R² descriptif sur {len(aligned)} variations communes",
        xaxis_title="Décalage Brent en quinzaines", yaxis_title="R²",
        template="plotly_dark", height=300,
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander("Voir les niveaux CNR et Brent par quinzaine"):
        st.dataframe(levels.round(2).reset_index(), width="stretch", hide_index=True)
    st.caption(
        "[CNR : historique des indices gazole bimensuels]"
        "(https://www.cnr.fr/espaces/13/indicateurs/26), "
        f"valeurs enregistrées jusque {levels.index.max()} ; Brent Yahoo Finance "
        "(BZ=F / EURUSD=X), clôtures quotidiennes par quinzaine. "
        "Les indices mensuels CNR restent les indices de référence."
    )


def _render_study() -> None:
    st.subheader("Étude du lien entre Brent et coût du gazole (CNR)")
    st.markdown(
        "L'objectif ici est de mesurer **l'indice CNR gazole professionnel "
        "mensuel officiel** Elle le "
        "rapproche du future Brent **BZ=F converti en EUR** "
        "Le mini-contrat MBC envisagé pour la couverture diffère de cette série. Le surcoût final du chargeur dépendra aussi du prix des transports, des "
        "pondérations CNR et des dates de facturation."
    )
    levels = load_official_study()
    results, aligned = lag_regressions(levels, max_lag=4)
    best = results.loc[results["R²"].idxmax()]
    lag = int(best["Décalage (mois)"])
    beta_cnr = float(best["β"])
    r2_cnr = float(best["R²"])
    st.markdown("### Résultats sur l'indice CNR officiel")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Meilleur shift testé", f"{lag} mois")
    r2.metric("R² CNR / Brent", f"{r2_cnr:.3f}")
    r3.metric("β CNR / Brent", f"{beta_cnr:.3f}")
    r4.metric("Mois comparables", str(len(aligned)))
    visible_results = results.loc[:, [
        "Décalage (mois)", "R²", "β", "N", "α (points de %)", "RMSE (points de %)"
    ]]
    st.dataframe(visible_results.round(4), width="stretch", hide_index=True)
    cnr_chart = go.Figure(
        go.Bar(
            x=results["Décalage (mois)"],
            y=results["R²"],
            marker_color=[PALETTE["orange"] if k == lag else PALETTE["cyan"] for k in results["Décalage (mois)"]],
        )
    )
    cnr_chart.update_layout(
        title="R² selon le décalage — CNR officiel",
        xaxis_title="Décalage du Brent (mois)",
        yaxis_title="R²",
        template="plotly_dark",
        height=330,
    )
    st.plotly_chart(cnr_chart, width="stretch")
    observed_x = aligned[f"brent_lag_{lag}"]
    observed_y = aligned["cnr_change"]
    fitted_y = float(best["α (points de %)"]) + beta_cnr * observed_x
    order = observed_x.argsort()
    regression_chart = go.Figure()
    regression_chart.add_trace(go.Scatter(
        x=observed_x,
        y=observed_y,
        mode="markers",
        name="Mois observés",
        marker=dict(color=PALETTE["cyan"], size=8),
    ))
    regression_chart.add_trace(go.Scatter(
        x=observed_x.iloc[order],
        y=fitted_y.iloc[order],
        mode="lines",
        name="Droite estimée",
        line=dict(color=PALETTE["orange"], width=3),
    ))
    regression_chart.update_layout(
        title=f"Régression CNR / Brent en EUR à k = {lag} mois",
        xaxis_title="Variation mensuelle Brent en EUR (%)",
        yaxis_title="Variation mensuelle indice CNR (%)",
        template="plotly_dark",
        height=370,
    )
    st.plotly_chart(regression_chart, width="stretch")
    st.caption(
        "Sources : [CNR, indice gazole professionnel mensuel]"
        "(https://www.cnr.fr/espaces/13/indicateurs/26), valeurs arrêtées à juillet "
        "2026, et Yahoo Finance (BZ=F / EURUSD=X), extrait le 23 septembre 2026. "
        "Le Brent en EUR est la moyenne mensuelle des cours "
        "quotidiens BZ divisés par EUR/USD. Séries communes de janvier 2020 à juillet "
        f"2026 ; mêmes {len(aligned)} mois de variation pour les shifts 0 à 4 "
        f"({aligned.index.min()} à {aligned.index.max()}), sans combler de mois manquant."
    )
    st.info(
        "Le meilleur ajustement est **contemporain (k = 0)** : une variation du "
        "Brent du mois t accompagne celle de l'indice CNR du même mois. Ce n'est "
        "pas une prévision connue au début du mois ; le backtest du hedge devra "
        "respecter les dates de position, de publication et de facture."
    )
    _render_fortnightly_study()

    st.markdown("### 1. Régression linéaire")
    st.latex(r"\Delta G_t = \alpha + \beta\,\Delta Brent_{t-k} + \varepsilon_t")
    st.markdown(
        "Les Δ sont les **variations mensuelles en %** de deux niveaux mensuels : "
        "indice du gazole et prix moyen du Brent. Faire la régression sur les niveaux "
        "pourrait donner une relation artificielle liée à leurs tendances communes. "
        "**β** mesure la variation moyenne du gazole associée à +1 % de Brent ; "
        "**R²** mesure la part de variation du gazole expliquée dans cet échantillon. "
        "Ni l'un ni l'autre ne prouve une causalité ou ne garantit une couverture future."
    )
    st.markdown("### 2. Recherche du décalage (« shift »)")
    st.markdown(
        "On teste **k de 0 à 4 mois** : la variation du Brent en t − k est "
        "rapprochée de celle du CNR en t. Les cinq modèles sont estimés sur "
        f"**les mêmes {len(aligned)} mois**. Le maximum de R² est à **k = 0**, alors que "
        "le notebook DGEC affichait k = 1 après avoir déjà décalé son gazole "
        "d'un mois pour la facture. L'alignement des deux séries explique cette "
        "différence ; il ne faut pas interpréter k = 1 comme un délai de marché établi."
    )
    st.markdown("### 3. Recherche du coefficient de hedge de variance minimale")
    st.latex(r"h^* = \frac{\operatorname{Cov}(\Delta G,\,\Delta Brent)}{\operatorname{Var}(\Delta Brent)} = \beta")
    st.markdown(
        "Si l'on représente **l'exposition gazole et le hedge Brent par des variations "
        "en pourcentage sur des notionnels comparables**, le β de la régression est "
        "aussi le coefficient qui minimise la variance du résidu dans l'échantillon. "
        f"Sur le CNR officiel, **le coefficient optimal ≈ {beta_cnr:.3f}** "
        f"et la variance résiduelle théorique est **1 − R² ≈ {100 * (1 - r2_cnr):.1f} %** de la variance "
        "initiale. C'est un résultat descriptif sur l'indice, pas un nombre de "
        "contrats ni une économie garantie. Il reste à appliquer la pondération "
        "gazole à chaque facture, convertir le montant exposé en notionnel du "
        "contrat Brent de 1 000 barils, puis tester le résultat hors échantillon."
    )
    hedge_values = [i / 100 for i in range(121)]
    variance_pct = [100 * (1 - r2_cnr * (2 * h / beta_cnr - (h / beta_cnr) ** 2)) for h in hedge_values]
    hedge_chart = go.Figure()
    hedge_chart.add_trace(
        go.Scatter(x=hedge_values, y=variance_pct, mode="lines", name="Variance relative", line=dict(color=PALETTE["cyan"], width=3))
    )
    hedge_chart.add_vline(x=beta_cnr, line_dash="dash", line_color=PALETTE["orange"])
    hedge_chart.update_layout(
        title="Variance résiduelle théorique du CNR selon le coefficient",
        xaxis_title="Coefficient h sur notionnel Brent comparable",
        yaxis_title="Variance du résidu / variance sans hedge (%)",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#24324b"),
        xaxis=dict(color="#24324b", linecolor="#64748b", showline=True),
        yaxis=dict(color="#24324b", linecolor="#64748b", showline=True),
        height=330,
    )
    st.plotly_chart(hedge_chart, width="stretch", theme=None, key="cnr_variance_chart")

    with st.expander("Comparer avec le notebook initial (proxy DGEC)"):
        notebook_results = pd.DataFrame(
            [
                (0, 0.0154, 0.062527),
                (1, 0.5824, 0.382867),
                (2, 0.0637, 0.124573),
                (3, 0.0004, -0.009529),
                (4, 0.0511, -0.112111),
            ],
            columns=["Shift Brent (mois)", "R²", "β"],
        )
        st.dataframe(notebook_results.round(4), width="stretch", hide_index=True)
        st.caption(
            "Sorties enregistrées de hedgeTransport.ipynb : gazole DGEC hors TVA "
            "décalé d'un mois pour simuler la facture. Ces chiffres ne sont plus "
            "utilisés pour le coefficient CNR présenté ci-dessus."
        )

    st.markdown("### 4. Ce que les résultats apportent à la suite")
    st.markdown(
        "- **Dimensionnement** : β et la pondération CNR relient une variation du Brent "
        "à un ordre de grandeur du surcoût en EUR ; le nombre de contrats doit encore "
        "être ajusté à la valeur du contrat Brent et à la devise.\n"
        "- **Calendrier** : le décalage renseigne sur le mois auquel le hedge doit être "
        "actif par rapport à la facture.\n"
        "- **Risque résiduel** : les écarts à la droite de régression montrent ce que le "
        "Brent ne compense pas ; on vérifiera leur impact hors échantillon.\n"
        "- **Backtest** : on ajoutera les P&L des contrats effectivement détenus, les "
        "roulements et leur conversion en EUR, puis on comparera la variance du budget."
    )

    st.markdown("### 5. Roll historique des futures Brent BZ")
    st.markdown(
        "À une date de roll donnée, on compare les prix **du contrat vendu et du "
        "contrat acheté le même jour**. Si le suivant est plus cher, la courbe est "
        "en contango ; s'il est moins cher, elle est en backwardation. Ce spread "
        "mesure l'écart de prix au remplacement, mais n'est pas une perte encaissée "
        "instantanément. Pour le **coût effectif "
        "de la stratégie**, il faut reconstituer les P&L quotidiens de chaque échéance "
        "détenue, avec les dates de roll, le nombre de contrats, les commissions et "
        "la conversion USD/EUR. Une seule série continue de Brent ne suffit pas. "
        "L'accès aux historiques par échéance est le principal travail de données."
    )
    st.info(
        "BZ=F dans yfinance donne une série continue du contrat proche, utile pour la "
        "régression. Elle ne fournit pas à elle seule les prix simultanés des échéances "
        "M1 et M2 nécessaires au contango et au roll. Les cotations BZ par échéance "
        "doivent être obtenues et contrôlées ; l'archive officielle CME DataMine "
        "permet ce calcul mais demande un accès aux données historiques."
    )

    st.markdown("### Tester un autre jeu de données")
    st.caption(
        "L'étude officielle ci-dessus est déjà calculée. Cet import facultatif permet "
        "de comparer un autre indice ou une autre fenêtre avec un Brent exprimé en EUR."
    )

    uploaded = st.file_uploader(
        "Importer des séries mensuelles (CSV : date, cnr, brent)",
        type="csv",
        key="cnr_brent_study_csv",
        help="Une ligne par mois ; date au format AAAA-MM, niveaux positifs, Brent en EUR/baril.",
    )
    if uploaded is None:
        return

    try:
        levels = load_study_csv(uploaded.getvalue())
        results, aligned = lag_regressions(levels)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Données non exploitables : {exc}")
        return

    best = results.loc[results["R²"].idxmax()]
    lag = int(best["Décalage (mois)"])
    st.caption(
        f"{len(aligned)} mois communs, de {aligned.index.min()} à {aligned.index.max()}. "
        "Classement descriptif sur l'échantillon importé ; à valider hors échantillon."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Décalage le mieux ajusté", f"{lag} mois")
    m2.metric("Coefficient h* = β", f"{best['β']:.3f}")
    m3.metric("R²", f"{best['R²']:.3f}")
    m4.metric("Erreur RMSE", f"{best['RMSE (points de %)']:.2f} pt")
    st.dataframe(results.round(3), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        curve = go.Figure(
            go.Bar(x=results["Décalage (mois)"], y=results["R²"], marker_color=PALETTE["cyan"])
        )
        curve.update_layout(
            title="Qualité d'ajustement par décalage",
            xaxis_title="Brent antérieur de k mois",
            yaxis_title="R²",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(curve, width="stretch")
    with right:
        x = aligned[f"brent_lag_{lag}"]
        fitted = best["α (points de %)"] + best["β"] * x
        scatter = go.Figure()
        scatter.add_trace(
            go.Scatter(x=x, y=aligned["cnr_change"], mode="markers", name="Observations")
        )
        order = x.argsort()
        scatter.add_trace(
            go.Scatter(x=x.iloc[order], y=fitted.iloc[order], mode="lines", name="Régression")
        )
        scatter.update_layout(
            title=f"Relation observée à k = {lag} mois",
            xaxis_title="Variation Brent (%)",
            yaxis_title="Variation CNR (%)",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(scatter, width="stretch")


def render() -> None:
    _inject_css()

    st.title("CNR Hedge Transport")
    st.markdown(
        """
        <div class="cnr-hero">
            <span class="cnr-pill">Pourquoi couvrir le coût du gazole ?</span>
            <h3>Du pied de facture au risque de marché</h3>
            <p>
                Depuis la loi n° 2006-10 du 5 janvier 2006, le prix d'un transport routier
                est révisé pour répercuter la variation du coût du carburant entre la commande
                et la réalisation du transport. Cette indexation, souvent appelée
                « pied de facture gazole », protège le transporteur d'une hausse du carburant,
                mais fait varier la facture payée par le chargeur.
            </p><br>
            <p>
                Lorsque le contrat ne définit pas sa propre formule, le Code des transports
                prévoit une référence aux indices gazole et à la part carburant publiés par le CNR.
                Le but de cette étude est d'abord de mesurer le surcoût subi par le chargeur,
                puis de rechercher la stratégie de couverture la plus efficace contre la hausse
                des matières premières énergétiques.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Cadre juridique : articles L. 3222-1 et L. 3222-2 du Code des transports. "
        "La formule contractuelle prévaut lorsqu'elle identifie les charges de carburant."
    )

    st.divider()

    tab_map, tab_hypotheses, tab_study, tab_weights, tab_scenario, tab_backtest, tab_results, tab_roadmap = st.tabs(
        ["Cartographie", "Hypothèses", "Étude Brent & hedge", "Pondérations CNR", "Contrat & couverture", "Backtest ICE", "Résultats & code", "Roadmap"]
    )

    with tab_weights:
        st.subheader("Pondération gazole officielle par date de contrat")
        st.markdown(
            "La pondération représente la part du gazole dans le coût total du transport "
            "pour le profil choisi. Le CNR révise cette structure **chaque année**. "
            "La date du contrat sélectionne donc le millésime de la même année civile, "
            "sans moyenne ni interpolation."
        )
        control_left, control_right = st.columns(2)
        with control_left:
            convoy_type = st.selectbox(
                "Type de convoi",
                options=list(CONVOY_LABELS),
                format_func=CONVOY_LABELS.get,
                key="cnr_convoy_type",
            )
        with control_right:
            contract_date = st.date_input(
                "Date du contrat",
                value=date.today(),
                min_value=date(2020, 1, 1),
                key="cnr_contract_date",
            )

        try:
            selected_weight = weight_for_contract(convoy_type, contract_date)
        except LookupError as exc:
            selected_weight = None
            st.warning(str(exc))
        else:
            metric_left, metric_right = st.columns(2)
            metric_left.metric(
                "Pondération gazole applicable",
                f"{selected_weight.diesel_weight_pct:.1f} %",
            )
            metric_right.metric("Millésime CNR retenu", selected_weight.effective_date.year)
            st.caption(
                f"Millésime {selected_weight.effective_date.year}, applicable à l'année "
                f"du contrat ({contract_date:%d/%m/%Y}). Aucune interpolation et aucun "
                "recours au millésime suivant."
            )

        st.plotly_chart(
            _build_cnr_weight_chart(convoy_type, selected_weight),
            width="stretch",
        )
        st.markdown(f"Source : [Comité national routier — historique officiel 2026]({SOURCE_URL})")

    with tab_map:
        st.subheader("Du contrat de transport à la couverture")
        st.markdown(
            "La démarche suit quatre étapes successives"
        )
        steps = [
            ("1", "Décrire le contrat", "Prix initial, dates et type de convoi."),
            ("2", "Calculer le pied de facture", "Appliquer la pondération annuelle et la variation de l'indice CNR."),
            ("3", "Mesurer la perte", "Comparer la facture révisée au budget initial du chargeur."),
            ("4", "Construire le hedge", "Tester l'instrument, le ratio et les dates de couverture."),
        ]
        step_cards = "".join(
            f'<article class="cnr-step">'
            f'<span class="cnr-step-number">ÉTAPE {number}</span>'
            f'<h4>{title}</h4><p>{description}</p>'
            f'</article>'
            for number, title, description in steps
        )
        st.markdown(
            f'<section class="cnr-step-grid">{step_cards}</section>',
            unsafe_allow_html=True,
        )

        st.markdown("### Comment lire les deux notions contractuelles")
        st.markdown(
            "**Exposition du chargeur.** Le transporteur achète le gazole ; le chargeur "
            "paie une prestation. Notre risque mesuré est donc le *supplément de facture* "
            "lié au carburant, pas les litres consommés. Les volumes de trajets serviront "
            "à agréger les factures mensuelles.\n\n"
            "**Indexation de la facture.** Par exemple, pour un transport initial de 1 000 €, une "
            "part gazole de 25 % et un indice qui monte de 20 % entre les dates prévues, "
            "le pied de facture est de **50 €**. Côté chargeur, ces 50 € sont précisément "
            "l'exposition à couvrir. Le contrat dispose des clauses réelles : indice, pondération, "
            "date de base, date d'application et éventuels plafonds. "
        )
        st.markdown("### Modélisation du surcoût ")
        st.latex(
            r"\mathrm{Surco\hat{u}t\ gazole} = P_0 \times "
            r"w_{\mathrm{gazole},\ \mathrm{CNR},\ \mathrm{ann\acute{e}e}} "
            r"\times \left(\frac{\mathrm{CNR}_{\mathrm{r\acute{e}alisation}}}"
            r"{\mathrm{CNR}_{\mathrm{commande}}} - 1\right)"
        )
        st.markdown(
            "- **P₀** : prix de transport initial ;\n"
            "- **w gazole CNR** : pondération annuelle du gazole correspondant au type de convoi ;\n"
            "- **CNR commande / CNR réalisation** : valeurs de l’indice gazole CNR aux deux dates.\n\n"
            "Si le résultat est positif, il représente le supplément payé par le chargeur. "
            "C'est cette exposition que le hedge cherchera à compenser."
        )

    with tab_hypotheses:
        st.subheader("Hypothèses du projet")
        _render_assumption_cards()

    with tab_study:
        _render_study()

    with tab_scenario:
        st.subheader("Questionnaire de contrat et couverture mini Brent MBC")
        st.markdown(
            "Choisissez une clause pour douze mois de transports. Les indices CNR mensuels "
            "déterminent les factures ; les observations par quinzaine servent uniquement à "
            "étudier la sensibilité du gazole. La couverture est maintenue pendant chaque "
            "mois exposé et réduite lorsque les prix des prestations sont fixés."
        )
        levels = load_official_study()
        available_starts = [
            month for month in levels.index[37:len(levels) - 11]
            if levels.loc[pd.period_range(month - 1, periods=13, freq="M"), ["cnr", "brent"]]
            .notna().all().all()
        ]
        left, right = st.columns(2)
        with left:
            start_month = st.selectbox(
                "Premier mois de transport",
                options=available_starts,
                index=len(available_starts) - 1,
                format_func=str,
                key="cnr_scenario_start",
            )
            scenario_convoy = st.selectbox(
                "Profil du convoi",
                options=list(CONVOY_LABELS),
                format_func=CONVOY_LABELS.get,
                key="cnr_scenario_convoy",
            )
            annual_budget = st.number_input(
                "Budget initial de transport sur 12 mois (€ HT)",
                min_value=1000.0,
                value=250000.0,
                step=1000.0,
                key="cnr_scenario_budget",
            )
        with right:
            billing_label = st.selectbox(
                "Facturation",
                options=["Mensuelle", "Annuelle"],
                key="cnr_scenario_billing",
            )
            revision_label = st.selectbox(
                "Révision de la part gazole",
                options=["Mensuelle", "Annuelle avec régularisation"],
                key="cnr_scenario_revision",
            )
            rounding_label = st.selectbox(
                "Conversion de la cible en contrats entiers",
                options=["À l'entier le plus proche", "Toujours à l'entier inférieur", "Toujours à l'entier supérieur"],
                key="cnr_scenario_rounding",
                help=(
                    "Exemple pour une cible de 4,20 : 4 contrats au plus proche ou à l'entier "
                    "inférieur, 5 à l'entier supérieur. Une cible de 0,71 donne 1 contrat "
                    "au plus proche ; une cible de 0,49 donne 0. À x,50, le calcul arrondit "
                    "vers l'entier supérieur."
                ),
            )
        annual_reference_label = st.selectbox(
            "Indice pour la révision annuelle",
            options=["Mois anniversaire", "Moyenne des 12 mois"],
            disabled=revision_label == "Mensuelle",
            key="cnr_scenario_annual_reference",
        )
        annual_reference = (
            "anniversary" if annual_reference_label == "Mois anniversaire" else "average"
        )
        billing = "monthly" if billing_label == "Mensuelle" else "annual"
        revision = "monthly" if revision_label == "Mensuelle" else "annual"
        rounding = {
            "À l'entier le plus proche": "nearest",
            "Toujours à l'entier inférieur": "floor",
            "Toujours à l'entier supérieur": "continuous",
        }[rounding_label]
        weight = weight_for_contract(
            scenario_convoy, date(start_month.year, start_month.month, 1)
        )
        training = levels.loc[:start_month - 1]
        fitted, training_months = lag_regressions(training, max_lag=0, min_observations=24)
        beta_raw = float(fitted.loc[0, "β"])
        beta = max(0.0, beta_raw)
        scenario = simulate_contract(
            levels, start_month, annual_budget, weight.diesel_weight,
            beta, billing, revision, rounding, annual_reference,
        )
        st.caption(
            f"Indice de base : {start_month - 1} ; poids gazole CNR : "
            f"{_fr_number(weight.diesel_weight_pct, 1)} % (millésime {weight.effective_date.year}). "
            f"Sensibilité β : {_fr_number(beta_raw, 4)}, estimée à partir de "
            f"{len(training_months)} variations mensuelles antérieures au contrat. "
            "Le Brent et l'indice CNR sont comparés sur le même mois (décalage de zéro mois)."
        )
        if beta_raw <= 0:
            st.warning("Le coefficient historique est négatif : aucune position longue Brent proposée.")
        st.info(
            "Clause simulée : révision mensuelle selon le CNR du mois, ou révision annuelle "
            "selon la référence choisie avec régularisation en fin de période. "
            "Le budget initial est réparti également entre les mois. La facturation annuelle "
            "regroupe les douze prestations en une facture finale."
        )
        first_brent = float(levels.loc[start_month - 1, "brent"])
        mini_notional_eur = MINI_BRENT_BARRELS * first_brent
        effective_coverage = (
            scenario.first_contracts / scenario.first_target_contracts
            if scenario.first_target_contracts > 0 else 0.0
        )
        st.markdown("### Comment est calculée la taille de la position ?")
        st.markdown(
            "On part du **budget des transports qui restent à réaliser**, pas de la seule "
            "facture du mois. On en retient la part gazole CNR, puis on applique β, la "
            "sensibilité estimée de l'indice CNR aux variations du Brent. On divise ce "
            "montant par la valeur de référence d'un mini-contrat Brent de 100 barils."
        )
        st.latex(
            r"N_t = \frac{B_t \times w_{\mathrm{gazole}} \times \beta}"
            r"{100 \times P_{\mathrm{Brent},t-1}}"
        )
        st.caption(
            "Nₜ = mini-contrats nécessaires avant arrondi ; Bₜ = budget des transports "
            "encore exposés ; w = part gazole CNR ; β = sensibilité CNR/Brent estimée ; "
            "P = Brent moyen du mois précédent en euros par baril ; 100 = barils par mini-contrat."
        )
        st.markdown(
            f"**Avec les valeurs de départ :** {_fr_number(annual_budget, 2)} € "
            f"× {_fr_number(weight.diesel_weight_pct, 1)} % × "
            f"{_fr_number(beta, 4)} ÷ (100 × {_fr_number(first_brent, 2)} €/baril) "
            f"= **{_fr_number(scenario.first_target_contracts, 2)} mini-contrats calculés**. "
            f"Un mini représente ici **{_fr_number(mini_notional_eur, 2)} €** de notionnel "
            "au prix Brent de référence."
        )
        st.caption(
            "Chaque mois, le budget encore exposé diminue d'un douzième après fixation "
            "du prix de la prestation ; le prix Brent de référence est actualisé. Le "
            "nombre indiqué est la position à détenir pendant ce mois, pas autant de "
            "nouveaux contrats à acheter chaque mois."
        )
        a, b, c, d = st.columns(4)
        invoice_result_label = (
            "Surcoût des factures avant couverture"
            if scenario.surcharge_eur >= 0 else "Économie sur les factures avant couverture"
        )
        a.metric(invoice_result_label, f"{_fr_number(abs(scenario.surcharge_eur))} €")
        if scenario.surcharge_eur >= 0:
            a.caption(
                f"Factures révisées ({_fr_number(scenario.annual_invoice_eur)} €) "
                f"− budget prévu ({_fr_number(annual_budget)} €)."
            )
        else:
            a.caption(
                f"Budget prévu ({_fr_number(annual_budget)} €) − factures révisées "
                f"({_fr_number(scenario.annual_invoice_eur)} €)."
            )
        b.metric("Mini-contrats calculés au départ", f"{_fr_number(scenario.first_target_contracts, 2)}")
        b.caption("Taille théorique de la position avant l'arrondi à un entier.")
        c.metric("Mini-contrats détenus au départ", str(scenario.first_contracts))
        c.caption(f"Résultat de la règle « {rounding_label.lower()} ».")
        d.metric("Part de la cible détenue au départ", f"{_fr_number(100 * effective_coverage)} %")
        if scenario.first_target_contracts > 0:
            d.caption(
                f"{_mini_contract_label(scenario.first_contracts)} ÷ "
                f"{_fr_number(scenario.first_target_contracts, 2)} calculés. "
                "C'est un rapport de tailles de position, pas le pourcentage du surcoût remboursé."
            )
        else:
            d.caption("Aucune position Brent positive n'est proposée avec ces paramètres.")
        if rounding == "nearest":
            st.info(
                f"**Arrondi choisi :** la cible de {_fr_number(scenario.first_target_contracts, 2)} "
                f"mini-contrats devient {_mini_contract_label(scenario.first_contracts)}. Par exemple, "
                "0,71 devient 1 ; 0,49 devient 0. Un nombre comme 4,20 ne peut pas être "
                "détenu directement : les contrats se négocient par unités entières."
            )
        elif rounding == "floor":
            st.info(
                "**Arrondi choisi :** on garde uniquement la partie entière de la cible. "
                "Cette règle évite de dépasser le nombre théorique, mais peut laisser "
                "des mois sans aucun contrat lorsque la cible est inférieure à 1."
            )
        else:
            st.info(
                "**Arrondi choisi :** toute cible positive est portée à l'entier supérieur. "
                "Cette règle maintient au moins un contrat chaque mois exposé, mais peut "
                "dépasser fortement la taille théorique en fin de période."
            )
        uncovered = scenario.ledger.loc[scenario.ledger["mini_contracts"] == 0, "month"].tolist()
        covered_months = 12 - len(uncovered)
        positioned_rows = [
            row for row in scenario.ledger.itertuples()
            if row.target_contracts > 0 and row.mini_contracts > 0
        ]
        max_row = max(
            positioned_rows,
            key=lambda row: row.mini_contracts / row.target_contracts,
            default=None,
        )
        st.markdown(
            f"**Position détenue pendant {covered_months} mois sur 12.** "
            + (
                f"Aucun mini-contrat en {', '.join(uncovered)} : ces mois ne reçoivent "
                "aucun gain ou perte de la couverture Brent."
                if uncovered else "Un contrat au moins est détenu chaque mois."
            )
        )
        if max_row is not None:
            max_coverage = max_row.mini_contracts / max_row.target_contracts
            st.markdown(
                f"**Plus forte position par rapport à la cible : {_fr_number(100 * max_coverage)} % "
                f"en {max_row.month}.** Ce mois-là, {_mini_contract_label(max_row.mini_contracts)} "
                f"{'est détenu' if max_row.mini_contracts == 1 else 'sont détenus'} "
                f"pour une cible calculée de "
                f"{_fr_number(max_row.target_contracts, 2)}. "
                "Ce pourcentage mesure le nominal détenu par rapport à la cible ; il ne "
                "mesure pas l'efficacité financière de la couverture."
            )
        if uncovered and beta > 0:
            st.warning(
                f"Sans contrat en {', '.join(uncovered)}, la facture de ces mois reste "
                "exposée aux variations de l'indice CNR. L'arrondi à l'entier supérieur "
                "maintiendrait une position, avec un risque de surcouverture."
            )
        if max_row is not None and max_coverage > 1.25:
            st.warning(
                f"En {max_row.month}, la position de {_mini_contract_label(max_row.mini_contracts)} "
                f"dépasse la cible de {_fr_number(100 * (max_coverage - 1))} %. "
                "La taille indivisible du mini-contrat explique cette surcouverture."
            )
        if scenario.first_contracts == 0:
            st.warning(
                "La cible est inférieure à un mini-contrat avec cet arrondi. "
                "Une couverture de 100 % du notionnel est impossible avec ce seul instrument."
            )
        st.markdown("### Le résultat financier de la couverture")
        st.markdown(
            "Le **P&L simulé de la couverture MBC** est le gain ou la perte attribué aux mini-contrats "
            "détenus pendant le mois. Un gain réduit le coût du transport ; une "
            "perte l'augmente. Le **P&L total du mois** compare ensuite le coût du transport "
            "et ce gain ou cette perte au budget prévu. Un total positif signifie que le "
            "mois finit sous budget ; un total négatif signifie qu'il reste un dépassement."
        )
        st.latex(
            r"R_t = B_t^{\mathrm{mois}} - C_t + H_t"
        )
        st.caption(
            "R = P&L total du mois ; B = budget mensuel ; C = coût du transport "
            "indexé du mois ; H = P&L simulé des mini-contrats MBC du mois."
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Factures révisées sur 12 mois", f"{_fr_number(scenario.annual_invoice_eur)} €")
        p1.caption("Total du coût de transport après application de l'indice CNR.")
        p2.metric("P&L simulé des mini-contrats MBC", f"{_fr_number(scenario.proxy_pnl_eur, signed=True)} €")
        p2.caption("Somme des gains et pertes mensuels de la position MBC simulée.")
        p3.metric(
            "P&L total sur 12 mois vs budget",
            f"{_fr_number(-scenario.residual_before_costs_eur, signed=True)} €",
        )
        p3.caption(
            f"Budget − factures + couverture = {_fr_number(annual_budget)} − "
            f"{_fr_number(scenario.annual_invoice_eur)} + "
            f"({_fr_number(scenario.proxy_pnl_eur, signed=True)}) €. "
            "Positif : sous budget ; négatif : dépassement restant."
        )
        st.markdown(
            "**Calcul du P&L simulé des MBC pour chaque mois :** "
            "`mini-contrats détenus × 100 barils × (Brent moyen du mois − Brent moyen du mois précédent)`. "
            "Il s'agit du résultat des mini-contrats MBC dans cette simulation. "
            "La série Brent continue en euros ne donne pas les prix effectivement négociés "
            "des mini-futures par échéance : les roulements, frais, marge et effets de "
            "conversion ne sont pas inclus."
        )
        first_row = scenario.ledger.iloc[0]
        st.caption(
            f"Exemple en {first_row['month']} : {_fr_number(first_row['base_eur'], 4)} € "
            f"de budget − {_fr_number(first_row['revised_charge_eur'], 4)} € de transport "
            f"+ ({_fr_number(first_row['proxy_pnl_eur'], 4, signed=True)} €) "
            f"de couverture ≈ {_fr_number(first_row['total_pnl_eur'], 2, signed=True)} € "
            "de P&L total du mois. Les calculs sont faits avant l'arrondi des colonnes à deux décimales."
        )
        cash_differs = not (
            (scenario.ledger["cash_invoice_eur"] - scenario.ledger["revised_charge_eur"])
            .abs().lt(0.005).all()
        )
        cost_label = "Coût du transport indexé (€)" if cash_differs else "Facture du mois (€)"
        ledger = pd.DataFrame({
            "Mois": scenario.ledger["month"],
            "P&L total du mois (€)": scenario.ledger["total_pnl_eur"],
            "Surcoût vs budget (€)": scenario.ledger["invoice_gap_eur"],
            "P&L couverture MBC simulée (€)": scenario.ledger["proxy_pnl_eur"],
            "Budget mensuel (€)": scenario.ledger["base_eur"],
            cost_label: scenario.ledger["revised_charge_eur"],
            "Mini nécessaires avant arrondi": scenario.ledger["target_contracts"],
            "Mini détenus": scenario.ledger["mini_contracts"],
            "Indice CNR": scenario.ledger["cnr"],
        })
        if cash_differs:
            ledger.insert(6, "Paiement facturé ce mois (€)", scenario.ledger["cash_invoice_eur"])
            st.caption(
                "La clause choisie décale l'encaissement : « coût du transport indexé » "
                "attribue le coût à la prestation du mois, tandis que « paiement facturé » "
                "montre le calendrier réel de facturation. Le surcoût et le P&L total "
                "utilisent le coût attribué au mois, afin de comparer une même période."
            )
        st.dataframe(ledger.round(2), width="stretch", hide_index=True)
        st.caption(
            "Lecture du tableau : surcoût = coût du transport indexé − budget mensuel ; "
            "P&L total = P&L de couverture estimé − surcoût. Le résultat est placé juste "
            "après le mois pour voir immédiatement si le coût net dépasse le budget."
        )
        with st.expander("Données gratuites pour mesurer les rolls"):
            st.markdown(
                "Le [bulletin quotidien CME](https://www.cmegroup.com/market-data/daily-bulletin.html) "
                "publie les prix de règlement par échéance et les volumes avec un délai. "
                "Il permet de commencer une collecte quotidienne gratuite sur MBC. "
                "Les historiques complets et propres par échéance sont proposés via "
                "[CME DataMine](https://www.cmegroup.com/datamine.html) ; leur disponibilité "
                "gratuite pour toute la période reste à vérifier."
            )
            st.markdown(
                "Au changement de contrat, enregistrer le prix de vente de la proche échéance, "
                "le prix du nouvel achat, les dates, les commissions et le change. "
                "Le spread entre échéances indique le contango ou la backwardation, mais "
                "ne constitue pas seul un débit immédiat. Le P&L réalisé vient des variations "
                "de chaque contrat détenu, puis des frais de transaction et de financement."
            )
            st.caption(
                "Le [règlement MBC](https://www.cmegroup.com/rulebook/NYMEX/9/992.pdf) "
                "définit un mini Brent de 100 barils réglé sur la moyenne mensuelle du Brent ICE. "
                "Ce sous-jacent diffère de la série continue BZ utilisée pour la régression. "
                "Le [bulletin CME](https://www.cmegroup.com/market-data/daily-bulletin.html) consulté le 3 octobre 2026 "
                "a affiché zéro transaction MBC et 231 contrats ouverts ; la liquidité et "
                "les conditions du courtier doivent être vérifiées avant toute exécution."
            )

    with tab_backtest:
        render_backtest()

    with tab_results:
        st.subheader("Presentation portfolio")
        st.markdown(
            """
            L'onglet Étude Brent estime maintenant la relation avec l'indice CNR gazole
            professionnel officiel. Le notebook DGEC reste visible comme comparaison.
            Le questionnaire calcule les factures et le nombre de mini-contrats.
            Le questionnaire garde son proxy mensuel Brent. L'onglet Backtest ICE
            suit les prix quotidiens des échéances fixes du classeur et calcule
            le coût net pour les positions saisies ou dimensionnées par β.
            """
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Resultat suivi", "Risque residuel")
        r2.metric("Indice étudié", "CNR gazole")
        r3.metric("Sortie finale", "Variance budget")

        code_col, result_col = st.columns(2)
        with code_col:
            st.markdown("### Maniere dont c'est code")
            st.markdown(
                """
                - les hypotheses sont stockees dans une liste structuree pour alimenter les cartes et le tableau ;
                - la régression utilise l'indice CNR officiel et une série Brent en EUR enregistrée avec ses sources ;
                - le questionnaire distingue facturation et révision mensuelles ou annuelles ;
                - les mini-contrats de 100 barils sont dimensionnés sur le budget encore exposé.
                """
            )
        with result_col:
            st.markdown("### Ce qu'il reste à valider")
            st.markdown(
                """
                - appliquer les clauses et factures réelles d'un chargeur ;
                - obtenir les cotations du mini Brent par échéance et dater les changements de contrat ;
                - valider les prix de règlement ICE, le change, les frais et la liquidité avant de conclure sur une exécution réelle ;
                - comparer le coût du transport avec et sans couverture sur une période non utilisée pour calibrer β.
                """
            )
    with tab_roadmap:
        st.subheader("Roadmap de la couverture du transport")
        st.markdown(
            "**État actuel.** Le calcul des factures CNR, la taille de la position et le "
            "P&L total mensuel fonctionnent dans le questionnaire. L'onglet Backtest ICE "
            "suit les échéances fixes et leur P&L quotidien. Le mini MBC réel reste "
            "à tester avec ses propres cotations et coûts d'exécution."
        )
        st.info(
            "**Priorité suivante :** obtenir les prix historiques du mini Brent MBC "
            "(100 barils) par échéance, les coûts d'exécution et les conditions de marge. "
            "Cela permettra de comparer le proxy ICE au mini réellement négociable."
        )
        for step in ROADMAP.to_dict("records"):
            available = step["Disponible aujourd'hui"]
            next_action = step["Prochaine action concrète"]
            with st.container(border=True):
                st.markdown(f"#### {step['Étape']} — {step['État actuel']}")
                st.markdown(
                    f"**Disponible aujourd'hui :** {available}\n\n"
                    f"**Prochaine action :** {next_action}"
                )
        st.caption(
            "Une conclusion sur l'efficacité de la couverture viendra après la "
            "comparaison hors échantillon du coût net avec et sans hedge, en tenant compte "
            "des frais, de la liquidité et des besoins de marge."
        )
