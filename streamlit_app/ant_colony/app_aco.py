"""Page pédagogique et laboratoire ACO du portfolio Streamlit."""
from pathlib import Path
import json
import hashlib
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from zipfile import BadZipFile
import streamlit as st

from .core import (ACOConfig, BENCHMARKS, START, benchmark_target, calendar_years,
                   optimize_aco, prepare_data, return_metrics, wealth_curves)
from .guide import INTRO, CONVENTIONS, PARAMETERS, PSEUDOCODE, SOURCES
from .snapshot import SNAPSHOT_PATH, read_snapshot
from .ratios import portfolio_ratios

HERE = Path(__file__).parent
NOTEBOOK = HERE / "optimisation_portefeuille_fourmis.ipynb"
COLORS = ["#50e3c2", "#f4c677", "#829bff"]


@st.cache_data(show_spinner=False, max_entries=2)
def load_data(snapshot_version: tuple[int, int]):
    # La version du fichier invalide le cache après une mise à jour locale.
    # Ne jamais ajouter de téléchargement de secours ici.
    return read_snapshot(SNAPSHOT_PATH)


def style_chart(fig, height=410):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(20,28,50,0.5)", font_color="#dce6f7",
                      height=height, margin=dict(l=15,r=15,t=55,b=25),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                                  title_text="", font=dict(color="#f0f5ff", size=14),
                                  bgcolor="rgba(17,26,46,0.95)",
                                  bordercolor="#30415f", borderwidth=1),
                      hovermode="x unified")
    return fig


def render_results(prices, benchmarks, universe, result, winner, stats):
    metrics = result["metrics"]
    target = result["target"]
    if result["feasible"]:
        st.success(f"Objectif historique atteint · CAGR {metrics['cagr']:.2%} ≥ {target:.2%} ({winner}).")
    else:
        st.warning(f"Aucune solution admissible trouvée dans ce budget de recherche. "
                   f"Déficit : {100*(target-metrics['cagr']):.2f} points de CAGR. "
                   "Cela ne prouve pas que le problème est infaisable.")
    st.caption("Résultat rétrospectif sur les données de sélection ; aucune performance hors échantillon.")
    a,b,c,d = st.columns(4)
    a.metric("Volatilité annualisée", f"{metrics['volatility']:.2%}")
    b.metric("CAGR du portefeuille", f"{metrics['cagr']:.2%}")
    c.metric("Performance cumulée", f"{metrics['total_return']:.2%}")
    d.metric("Baisse maximale", f"{metrics['max_drawdown']:.2%}")

    st.markdown("#### Trajectoires comparées · base 100")
    curves = wealth_curves(prices, benchmarks, result["tickers"])
    fig = px.line(curves, color_discrete_sequence=COLORS, labels={"value":"Valeur base 100", "index":"Date", "variable":""})
    fig.add_hline(y=100, line_dash="dot", line_color="#65728e")
    st.plotly_chart(style_chart(fig), width="stretch")
    st.caption("10 % maintenus dans chaque action par rééquilibrage quotidien théorique ; indices prix, dividendes non réinvestis et frais exclus.")

    left,right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Les 10 actions retenues")
        selected = universe.set_index("ticker").loc[result["tickers"], ["name", "index"]].reset_index()
        selected["Poids"] = .10
        selected.columns = ["Ticker", "Entreprise", "Indice", "Poids"]
        st.dataframe(selected, hide_index=True, width="stretch",
                     column_config={"Poids":st.column_config.NumberColumn(format="percent")})
        st.caption("100 % investis · 10 titres distincts · aucun poids optimisé.")
        st.download_button("Télécharger la sélection · CSV", selected.to_csv(index=False).encode("utf-8-sig"),
                           "selection_fourmis.csv", "text/csv", key="aco_selection_download")
    with right:
        st.markdown("#### Le rendement minimal à respecter")
        comparison = pd.concat([pd.DataFrame({"Portefeuille ACO":metrics}).T, stats])
        comparison = comparison.rename(columns={"cagr":"CAGR", "volatility":"Volatilité", "total_return":"Performance", "max_drawdown":"Baisse max."})
        st.dataframe(comparison.style.format("{:.2%}"), width="stretch")
        st.info(f"Le benchmark gagnant est **{winner}**. Le seuil minimal de CAGR vaut **{target:.2%}** sur cette période commune.")

        st.markdown("#### Rendement ajusté du risque · portefeuille")
        daily_returns = prices[result["tickers"]].pct_change(fill_method=None).iloc[1:].mean(axis=1)
        ratios = portfolio_ratios(daily_returns)
        sharpe, sortino = st.columns(2)
        sharpe.metric("Ratio de Sharpe", f"{ratios['sharpe']:.2f}" if np.isfinite(ratios["sharpe"]) else "N/D")
        sortino.metric("Ratio de Sortino", f"{ratios['sortino']:.2f}" if np.isfinite(ratios["sortino"]) else "N/D")
        st.caption("Ratios annualisés · 252 séances/an · taux sans risque  "
                   "Sortino : écarts sous 0 % rapportés à toutes les séances. "
                   "Même historique et rééquilibrage quotidien que ci-dessus ; N/D si le risque mesuré est nul.")

    st.markdown("#### Comment la colonie progresse")
    risk,feasibility = st.columns(2)
    history = result["history"]
    with risk:
        # Le risque du meilleur admissible est monotone ; avant faisabilité le déficit prime.
        plot = history.copy()
        plot["volatility"] = plot["volatility"].where(plot["feasible"])
        fig = px.line(plot, x="iteration", y="volatility", labels={"iteration":"Itération", "volatility":"Volatilité du meilleur admissible"}, color_discrete_sequence=COLORS)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style_chart(fig,300), width="stretch")
        st.caption("La courbe commence à la première solution admissible trouvée.")
    with feasibility:
        fig = px.line(history, x="iteration", y="feasible_ants", labels={"iteration":"Itération", "feasible_ants":"Fourmis respectant le seuil"}, color_discrete_sequence=[COLORS[1]])
        st.plotly_chart(style_chart(fig,300), width="stretch")
        st.caption(f"{result['unique_evaluated']:,} paniers distincts évalués sur {result['config']['n_ants'] * result['config']['n_iterations']:,} tirages.")
    st.download_button("Exporter les trajectoires base 100", curves.to_csv(index_label="Date"), "trajectoires_fourmis.csv", "text/csv")


def render():
    st.html("""<style>
    .aco-hero {padding:30px 34px; border:1px solid #314766; border-radius:16px;
      background:linear-gradient(120deg,#152d42,#182039 65%,#29233c); margin-bottom:22px;}
    .aco-hero .aco-eyebrow {color:#50e3c2 !important; letter-spacing:2px; font-size:.8rem; font-weight:700;}
    .aco-hero h1 {color:#f0f5ff !important; font-size:2.2rem; margin:10px 0;}
    .aco-hero p {color:#c0cee2 !important; max-width:900px; font-size:1.06rem;}
    .aco-step {background:#151e35;border:1px solid #30415f;border-top:3px solid #50e3c2;
      border-radius:10px;padding:18px 20px;min-height:165px;}
    .aco-step h4 {color:#50e3c2 !important; margin:0 0 9px;}
    .aco-step p {color:#bfcee0 !important;font-size:.96rem;}
    [data-testid="stMetric"] {background:#151e35;border:1px solid #30415f;border-radius:10px;padding:14px;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
      font-size:1.6rem !important;line-height:1.3 !important;
      white-space:normal !important;overflow-wrap:anywhere;
      overflow:visible !important;text-overflow:clip !important;}
    [data-testid="stCode"] {
      background:#111a2e !important;border:1px solid #30415f;border-radius:10px;}
    [data-testid="stCode"] pre {background:#111a2e !important;}
    [data-testid="stCode"] pre, [data-testid="stCode"] pre * {
      color:#e8eef8 !important;}
    /* JSON and exports must override the portfolio's global light text. */
    [data-testid="stJson"], [data-testid="stJson"] .react-json-view,
    [data-testid="stJson"] .react-json-view * {
      background-color:#111a2e !important;color:#e8eef8 !important;}
    [data-testid="stJson"] {
      border:1px solid #30415f;border-radius:10px;padding:16px;}
    [data-testid="stJson"] svg {fill:#e8eef8 !important;}
    [data-testid="stDownloadButton"] button {
      background:#203c50 !important;color:#f0f5ff !important;
      border:1px solid #50e3c2 !important;border-radius:9px;}
    [data-testid="stDownloadButton"] button * {color:inherit !important;}
    [data-testid="stDownloadButton"] button:hover {
      background:#2b5065 !important;border-color:#9bf3df !important;}
    [data-testid="stDownloadButton"] button:focus-visible {
      outline:3px solid #50e3c2 !important;outline-offset:3px;}
    </style><div class="aco-hero"><div class="aco-eyebrow">RECHERCHE OPÉRATIONNELLE · FINANCE QUANTITATIVE</div>
    <h1>Optimisation par colonie de fourmis</h1>
    <p>Choisir dix actions. Garder 10 % sur chacune. Chercher le risque le plus faible,
    avec un rendement au moins égal au meilleur du CAC 40 et du CAC Mid 60.</p></div>""")
    a,b,c,d = st.columns(4)
    a.metric("Univers de départ", "40 + 60 actions")
    b.metric("Allocation", "10 × 10 %")
    c.metric("Objectif", "Risque minimal")
    d.metric("Depuis", "01/01/2022")
    st.caption("Étude rétrospective · sélection des titres sans solveur · données de clôture · notebook autonome disponible.")

    understand,method,lab,data,notebook = st.tabs(["Comprendre", "Pseudo-code & paramètres", "Expérimentation", "Données & limites", "Notebook & code"])
    with understand:
        st.markdown("### Le problème à résoudre")
        st.markdown(INTRO)
        for column,title,text in zip(st.columns(3), ["01 · Construire", "02 · Comparer", "03 · Apprendre"],
                                    ["Chaque fourmi propose 10 titres distincts. Les poids sont déjà fixés à 10 %.",
                                     "Le panier doit franchir le seuil de rendement. Parmi les paniers admissibles, le moins volatil gagne.",
                                     "Les bons paniers déposent des phéromones. L'évaporation et les bornes préservent l'exploration."]):
            column.markdown(f'<div class="aco-step"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.markdown("### Une formulation précise")
        st.latex(r"\min_{S\subseteq U,\ |S|=10}\ \sigma(S)\quad\text{sous}\quad g(S)\geq g_{\min}")
        st.latex(r"r_{S,t}=\frac{1}{10}\sum_{i\in S}r_{i,t},\quad \sigma(S)=\sqrt{252}\,s(r_S),\quad g(S)=\left[\prod_t(1+r_{S,t})\right]^{1/Y}-1")
        st.latex(r"g_{\min}=\max\{g_{CAC\ 40},\ g_{CAC\ Mid\ 60}\}")
        st.markdown("**U** est l'univers éligible, **S** le panier de dix actions, **Y** la durée en années et **s** l'écart-type empirique. Le risque dépend aussi des corrélations entre actions.")
        st.info("Le rendement minimal est une contrainte historique, pas une promesse de rendement futur. Le meilleur panier trouvé n'est pas un optimum global certifié.")
        st.markdown("### Hypothèses de cette première version")
        st.markdown(CONVENTIONS)
    with method:
        st.markdown("### Un pseudo-code qui suit le code Python")
        st.code(PSEUDOCODE, language="text")
        st.markdown("### Notice des variables et paramètres")
        st.dataframe(pd.DataFrame(PARAMETERS, columns=["Variable", "Valeur par défaut", "Signification", "Effet / lecture"]),
                     hide_index=True, width="stretch", row_height=90, height=650)
        st.latex(r"p_i(S)=\frac{\tau_i^\alpha\eta_i^\beta}{\sum_{j\notin S}\tau_j^\alpha\eta_j^\beta}\quad(i\notin S)")
        st.latex(r"\tau_i\leftarrow\operatorname{clip}\left((1-\rho)\tau_i+\sum_{S\ni i}\Delta\tau(S),\tau_{min},\tau_{max}\right)")
        st.caption("Exemple : avant dépôt, τ = 1 et ρ = 0,20 donnent τ = 0,80. Dans cette adaptation, les traces sont attachées aux actions, et non aux arêtes d'un parcours.")
    with notebook:
        st.markdown("### Lire, exécuter et transmettre le projet")
        st.markdown("Le notebook contient les explications, le snapshot de l'univers, toutes les fonctions Python, le téléchargement des données et les graphiques. Il fonctionne sans importer de module de ce dépôt et conserve les résultats de son exécution de validation.")
        if NOTEBOOK.is_file():
            st.download_button("Télécharger le notebook autonome · .ipynb", NOTEBOOK.read_bytes(), NOTEBOOK.name,
                               "application/x-ipynb+json", type="primary", key="aco_notebook_download")
        st.code("pip install -r streamlit_app/ant_colony/requirements.txt\n# Ouvrir optimisation_portefeuille_fourmis.ipynb dans Jupyter ou VS Code\n# Puis : Restart Kernel & Run All", language="bash")
        with st.expander("Lire le moteur Python complet"):
            st.code((HERE / "core.py").read_text(encoding="utf-8"), language="python")
        st.download_button("Télécharger le moteur · core.py", (HERE / "core.py").read_bytes(), "core.py", "text/x-python")
        st.markdown("### Références")
        st.markdown(SOURCES)
    with data:
        st.markdown("### Sources, calendrier et limites")
        st.markdown(CONVENTIONS)
        st.markdown(SOURCES)
        st.markdown("### Un instantané pour une étude reproductible")
        st.markdown("La page lit uniquement des cours déjà enregistrés sur le serveur. Les visites et les relances de la colonie ne déclenchent aucun appel à Yahoo ou Euronext. La période reste fixe jusqu'à une mise à jour manuelle par le propriétaire du projet.")
    with lab:
        st.markdown("### Le laboratoire de sélection")
        st.caption("Le calcul initial utilise les paramètres par défaut. Modifie le budget ou le comportement de la colonie, puis relance la recherche.")
        try:
            with st.spinner("Lecture de l'instantané local et contrôle des deux indices exacts…"):
                file_info = SNAPSHOT_PATH.stat()
                raw,metadata,universe = load_data((file_info.st_mtime_ns, file_info.st_size))
                prices,benchmarks,audit = prepare_data(raw, universe)
        except (ValueError, KeyError, OSError, BadZipFile) as exc:
            st.error(f"Instantané local indisponible : {exc}")
            st.info("Le propriétaire du projet doit régénérer le fichier avec refresh_snapshot.py depuis son terminal. Aucun téléchargement automatique n'est lancé.")
            return
        st.info(f"Données figées · dernière clôture : {prices.index[-1]:%d/%m/%Y}. Vous pouvez relancer la colonie sur ce même historique ; la mise à jour des cours est réservée au propriétaire du projet.")
        target,winner,stats = benchmark_target(benchmarks)
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([prices,benchmarks],axis=1),index=True).values.tobytes()).hexdigest()
        st.caption(f"Période effective : {prices.index[0]:%d/%m/%Y} → {prices.index[-1]:%d/%m/%Y} · {len(prices):,} clôtures communes · {len(prices.columns)}/100 actions éligibles.")
        with st.form("aco_parameters"):
            first,second,third = st.columns(3)
            ants = first.slider("Nombre de fourmis", 10, 150, 40, 10)
            iterations = second.slider("Nombre d'itérations", 20, 200, 80, 10)
            rho = third.slider("ρ · taux d'évaporation", .05, .95, .20, .05)
            first,second,third = st.columns(3)
            alpha = first.slider("α · influence des phéromones", .0, 4., 1., .25)
            beta = second.slider("β · influence de l'heuristique", .0, 4., 2., .25)
            seed = third.number_input("Graine aléatoire", min_value=0, max_value=2**31-1, value=42, step=1)
            st.caption("Fixes dans cette version : K=10, n_elite=5, Q=1, τ initial=1, τ min=0,05, τ max=5. Le notebook permet aussi de les modifier.")
            submitted = st.form_submit_button("Lancer la colonie", type="primary", width="stretch")
        key = "aco_result"
        previous = st.session_state.get(key)
        if submitted or previous is None or previous["fingerprint"] != fingerprint:
            config = ACOConfig(n_ants=ants,n_iterations=iterations,rho=rho,alpha=alpha,beta=beta,seed=int(seed))
            progress = st.progress(0., text="La colonie explore les paniers…")
            result = optimize_aco(prices.pct_change(fill_method=None).iloc[1:],target,calendar_years(prices.index),config,
                                  progress=lambda value: progress.progress(value))
            progress.empty()
            st.session_state[key] = {"fingerprint":fingerprint, "result":result}
        result = st.session_state[key]["result"]
        render_results(prices,benchmarks,universe,result,winner,stats)
    with data:
        st.markdown("### Traçabilité du jeu chargé")
        st.json(metadata)
        if audit["large_moves"].sum():
            st.warning("Certaines actions présentent des variations de plus de 50 % sur une séance. L'audit les signale pour contrôler les restructurations et opérations sur titres ; elles ne sont pas automatiquement corrigées.")
        st.dataframe(audit, hide_index=True, width="stretch")
        st.download_button("Télécharger l'audit des 100 actions", audit.to_csv(index=False).encode("utf-8-sig"), "audit_univers_fourmis.csv", "text/csv")
        st.download_button("Exporter les clôtures utilisées", pd.concat([prices,benchmarks],axis=1).to_csv(index_label="Date"), "clotures_fourmis.csv", "text/csv")
