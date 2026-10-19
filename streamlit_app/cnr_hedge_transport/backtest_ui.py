"""Interface Streamlit du backtest éducatif Brent ICE / pied de facture CNR."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .backtest import (
    CONTRACT_BARRELS_ICE,
    CONTRACT_BARRELS_MINI,
    load_cnr_monthly,
    load_ecb_eurusd,
    load_ice_history,
    parse_eurusd_csv,
    run_strip_backtest,
)
from .sizing import estimate_pre_entry_beta, recommend_strip_positions
from .study import load_official_study
from .weights import CONVOY_LABELS, weight_for_contract


PALETTE = {
    "cyan": "#00d4ff",
    "orange": "#ffb020",
    "green": "#17c964",
    "red": "#ff5c7a",
    "border": "#3d4263",
}
REPO_WORKBOOK = Path(__file__).with_name("data") / "Brent Futures 2021.xlsx"
LOCAL_WORKBOOK = Path.home() / "Downloads" / "Brent Futures 2021.xlsx"


@st.cache_data(show_spinner=False)
def _cached_history(content: bytes):
    return load_ice_history(content)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _cached_ecb_fx(start: date, end: date):
    return load_ecb_eurusd(start, end)


def _style(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f4f6ff"),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        xaxis=dict(gridcolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"]),
    )
    return fig


def _cumulative_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = pd.to_datetime(monthly["month"])
    for column, label, color in (
        ("surcharge_eur_cum", "Surcoût CNR cumulé", PALETTE["red"]),
        ("pnl_eur_cum", "P&L hedge cumulé", PALETTE["green"]),
        ("residual_eur_cum", "Coût net cumulé", PALETTE["cyan"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=monthly[column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=3),
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f} €<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_yaxes(title_text="Montant cumulé (€)")
    return _style(fig)


def _contract_chart(daily: pd.DataFrame, month: str) -> go.Figure:
    path = daily.loc[daily["month"] == month]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["price_usd_bbl"],
            mode="lines",
            name="Prix du contrat ICE",
            line=dict(color=PALETTE["orange"], width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["pnl_eur_cum"],
            mode="lines",
            name="P&L cumulé de cette position",
            line=dict(color=PALETTE["cyan"], width=3),
        ),
        secondary_y=True,
    )
    _style(fig, height=390)
    fig.update_yaxes(title_text="Brent (USD/baril)", secondary_y=False)
    fig.update_yaxes(title_text="P&L (€)", secondary_y=True, gridcolor=PALETTE["border"])
    return fig


def _show_monthly_details(monthly: pd.DataFrame) -> None:
    labels = {
        "month": "Facture",
        "cnr_month": "Mois CNR utilisé",
        "invoice_eur": "Facture initiale (€)",
        "cnr": "CNR du mois",
        "surcharge_eur": "Écart de facture CNR (€)",
        "ticker": "Future ICE détenu",
        "entry_date": "Achat",
        "exit_date": "Sortie",
        "rank_entry": "CO à l'achat",
        "rank_exit": "CO à la sortie",
        "entry_price_usd_bbl": "Prix achat ($/bbl)",
        "exit_price_usd_bbl": "Prix sortie ($/bbl)",
        "term_spread_usd_bbl": "Écart vs CO1 à l'achat ($/bbl)",
        "curve_shape": "Pente à l'achat",
        "front_move_usd": "Mouvement CO1 ($)",
        "curve_effect_usd": "Évolution écart vs CO1 ($)",
        "barrels": "Barils couverts",
        "ice_contracts_fractional": "Contrats ICE théoriques",
        "mini_equivalents": "Équivalents mini 100 bbl",
        "pnl_usd": "P&L ($)",
        "pnl_eur": "P&L (€)",
        "residual_eur": "Coût net (€)",
    }
    detail = monthly.loc[:, list(labels)].rename(columns=labels).copy()
    detail["Achat"] = pd.to_datetime(detail["Achat"]).dt.strftime("%d/%m/%Y")
    detail["Sortie"] = pd.to_datetime(detail["Sortie"]).dt.strftime("%d/%m/%Y")
    numeric = detail.select_dtypes(include="number").columns
    detail[numeric] = detail[numeric].round(2)
    st.dataframe(detail, width="stretch", hide_index=True)
    st.download_button(
        "Télécharger le détail mensuel (CSV)",
        monthly.to_csv(index=False).encode("utf-8-sig"),
        file_name="backtest_brent_ice_cnr_mensuel.csv",
        mime="text/csv",
    )


def render_backtest() -> None:
    st.subheader("Backtest du pied de facture avec un strip Brent ICE")
    st.markdown(
        "Chaque ligne ci-dessous est une facture mensuelle. Le modèle achète en janvier "
        "un future ICE de livraison **deux mois après** le mois de facture, puis le suit "
        "jusqu'à la fin de ce mois. Les fractions de contrat servent à mesurer une "
        "couverture théorique ; aucun contrat fractionnaire n'est négocié dans cette simulation. "
        "Le nombre de barils est calculé depuis le β connu avant l'achat, ou saisi manuellement."
    )
    uploaded = st.file_uploader(
        "Exporter Bloomberg : Brent ICE CO1 à CO12 (.xlsx)", type="xlsx", key="ice_backtest_upload"
    )
    if uploaded is not None:
        content = uploaded.getvalue()
        st.caption("Source : classeur Bloomberg chargé ici.")
    elif REPO_WORKBOOK.is_file():
        content = REPO_WORKBOOK.read_bytes()
        st.caption("Source : classeur placé dans le dossier data du projet. Un fichier chargé ici le remplacera.")
    elif LOCAL_WORKBOOK.is_file():
        content = LOCAL_WORKBOOK.read_bytes()
        st.caption("Source : classeur local détecté dans Downloads. Un fichier chargé ici le remplacera.")
    else:
        st.info("Charge le classeur Brent ICE pour lancer le backtest.")
        return

    try:
        history = _cached_history(content)
        cnr = load_cnr_monthly()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Données impossibles à charger : {exc}")
        return

    market_years = set(history.prices.index.year)
    cnr_years = set(cnr.index.year)
    years = sorted(market_years & cnr_years)
    if not years:
        st.warning("Aucune année commune entre les cotations ICE et l'indice CNR.")
        return
    default_year = max(year for year in years if year < max(years)) if len(years) > 1 else years[0]

    controls_a, controls_b, controls_c = st.columns(3)
    with controls_a:
        year = st.selectbox("Année des factures", years, index=years.index(default_year))
    january_dates = [d.date() for d in history.prices.index if d.year == year and d.month == 1]
    if not january_dates:
        st.warning("Aucune séance disponible en janvier pour cette année.")
        return
    default_entry = next((i for i, d in enumerate(january_dates) if d.day >= 3), 0)
    with controls_b:
        entry_date = st.selectbox(
            "Achat du strip en janvier",
            january_dates,
            index=default_entry,
            format_func=lambda value: value.strftime("%d/%m/%Y"),
        )
    with controls_c:
        convoy_type = st.selectbox("Type de convoi", list(CONVOY_LABELS), format_func=CONVOY_LABELS.get, key="backtest_convoy")

    entry_month = pd.Period(entry_date, freq="M")
    base_candidates = [month for month in cnr.index if month <= entry_month - 2]
    if not base_candidates:
        st.warning("Aucun mois CNR suffisamment ancien pour servir de référence connue à l'achat.")
        return
    most_recent_base = base_candidates[-1]
    control_base, control_lag, control_weight, control_invoice = st.columns(4)
    with control_base:
        base_month = st.selectbox(
            "Mois CNR de référence du contrat",
            list(reversed(base_candidates[-24:])),
            format_func=str,
        )
    with control_lag:
        cnr_lag = st.selectbox("Décalage CNR de la facture (mois)", [0, 1, 2])
    default_weight = weight_for_contract(convoy_type, entry_date).diesel_weight_pct
    with control_weight:
        weight_pct = st.number_input(
            "Part gazole de la facture (%)", min_value=0.0, max_value=100.0,
            value=float(default_weight), step=0.1,
        )
    with control_invoice:
        invoice_default = st.number_input(
            "Facture initiale par mois (€)", min_value=0.0, value=50_000.0, step=1_000.0,
        )
    if base_month != most_recent_base:
        st.caption(f"Référence CNR choisie : {base_month}. La date effective de publication de cet indice n'est pas historisée.")
    else:
        st.caption("Référence CNR : deux mois avant l'achat, par prudence faute de dates historiques de publication.")

    invoice_months = [month for month in cnr.index if month.year == year and month >= pd.Period(entry_date, freq="M")]
    fx_mode = st.selectbox(
        "Conversion du P&L quotidien de $ en €",
        ["EUR/USD constant choisi", "Historique quotidien BCE", "Importer un CSV EUR/USD"],
    )
    if fx_mode == "EUR/USD constant choisi":
        eurusd = st.number_input("Hypothèse EUR/USD (dollars pour 1 €)", min_value=0.01, value=1.10, step=0.01)
        st.caption("Ce taux est constant sur toute la période : le P&L en euros est une simulation, pas un change historique.")
    elif fx_mode == "Historique quotidien BCE":
        try:
            eurusd = _cached_ecb_fx(entry_date, date(year, 12, 31))
        except Exception as exc:
            st.warning(f"Historique BCE indisponible ({exc}). Choisis un taux constant ou importe un CSV.")
            return
        st.caption("Taux de référence BCE USD par EUR ; le dernier taux connu est repris les jours sans publication BCE.")
    else:
        fx_file = st.file_uploader("CSV avec date,eurusd ou TIME_PERIOD,OBS_VALUE", type="csv", key="backtest_fx_upload")
        if fx_file is None:
            st.info("Charge un historique EUR/USD pour afficher les résultats en euros.")
            return
        try:
            eurusd = parse_eurusd_csv(pd.read_csv(BytesIO(fx_file.getvalue())))
        except ValueError as exc:
            st.error(str(exc))
            return

    try:
        calibration = estimate_pre_entry_beta(load_official_study(), entry_date)
    except (ValueError, OSError) as exc:
        calibration = None
        st.caption(f"Dimensionnement β indisponible pour cette date : {exc}")

    st.markdown("### Dimensionnement des positions")
    modes = ["Calculé avec le β connu avant l'achat", "Contrats saisis manuellement"] if calibration else ["Contrats saisis manuellement"]
    position_mode = st.radio("Méthode", modes, horizontal=True)
    if calibration and position_mode == modes[0]:
        st.caption(
            f"β pré-achat = {calibration.beta:.3f} ; R² = {calibration.r2:.3f} "
            f"sur {calibration.observations} variations mensuelles jusqu'à {calibration.last_month}. "
            "Le β calculé sur toute la série dans l'onglet Étude utilise des mois futurs pour les années passées ; "
            "il n'est donc pas repris tel quel ici."
        )
        invoice_inputs = pd.DataFrame(
            {
                "Mois de facture": [str(month) for month in invoice_months],
                "Facture initiale (€)": [invoice_default] * len(invoice_months),
            }
        )
        edited = st.data_editor(
            invoice_inputs,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["Mois de facture"],
            column_config={
                "Facture initiale (€)": st.column_config.NumberColumn(min_value=0.0, step=1_000.0, format="%.2f"),
            },
            key=f"backtest_auto_invoices_{year}",
        )
        try:
            positions = recommend_strip_positions(
                history,
                pd.DataFrame({"month": edited["Mois de facture"], "invoice_eur": edited["Facture initiale (€)"]}),
                entry_date=entry_date,
                diesel_weight=weight_pct / 100,
                beta=calibration.beta,
                eurusd=eurusd,
            )
        except (ValueError, KeyError) as exc:
            st.error(f"Dimensionnement impossible : {exc}")
            return
        shown = positions.loc[:, [
            "month", "ticker", "diesel_exposure_eur", "entry_price_usd_bbl",
            "barrels", "ice_contracts_fractional", "mini_equivalents",
        ]].rename(columns={
            "month": "Facture", "ticker": "Future ICE", "diesel_exposure_eur": "Montant gazole exposé (€)",
            "entry_price_usd_bbl": "Prix à l'achat ($/bbl)", "barrels": "Barils calculés",
            "ice_contracts_fractional": "ICE fractionnaire", "mini_equivalents": "Équivalents mini",
        })
        st.dataframe(shown.round(3), width="stretch", hide_index=True)
        st.caption(
            "Barils = facture initiale × part gazole × β × EUR/USD à l'achat ÷ prix de l'échéance ICE à l'achat. "
            "Ces quantités sont fractionnaires et indicatives : β provient du CNR et d'un proxy Brent en EUR, "
            "pas d'une optimisation sur les P&L de ces échéances ICE."
        )
    else:
        unit = st.radio(
            "Unité de la position saisie pour chaque facture",
            ["Équivalents mini (100 barils)", "Contrats ICE fractionnaires (1 000 barils)"],
            horizontal=True,
        )
        mini_unit = unit.startswith("Équivalents")
        suggested_counts = [0.0] * len(invoice_months)
        if calibration:
            try:
                suggested = recommend_strip_positions(
                    history,
                    pd.DataFrame({
                        "month": [str(month) for month in invoice_months],
                        "invoice_eur": [invoice_default] * len(invoice_months),
                    }),
                    entry_date=entry_date,
                    diesel_weight=weight_pct / 100,
                    beta=calibration.beta,
                    eurusd=eurusd,
                )
                count_column = "mini_equivalents" if mini_unit else "ice_contracts_fractional"
                suggested_counts = suggested[count_column].tolist()
            except (ValueError, KeyError):
                pass
        inputs = pd.DataFrame(
            {
                "Mois de facture": [str(month) for month in invoice_months],
                "Facture initiale (€)": [invoice_default] * len(invoice_months),
                "Nombre de contrats": suggested_counts,
            }
        )
        st.markdown("**Modifie les factures et les positions mois par mois.** Les quantités initiales viennent du β pré-achat lorsqu'il est disponible ; mets 0 pour laisser un mois sans couverture.")
        edited = st.data_editor(
            inputs,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["Mois de facture"],
            column_config={
                "Facture initiale (€)": st.column_config.NumberColumn(min_value=0.0, step=1_000.0, format="%.2f"),
                "Nombre de contrats": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.3f"),
            },
            key=f"backtest_positions_{year}_{'mini' if mini_unit else 'ice'}",
        )
        positions = pd.DataFrame(
            {
                "month": edited["Mois de facture"],
                "invoice_eur": edited["Facture initiale (€)"],
                "barrels": pd.to_numeric(edited["Nombre de contrats"], errors="coerce")
                * (CONTRACT_BARRELS_MINI if mini_unit else CONTRACT_BARRELS_ICE),
            }
        )

    try:
        result = run_strip_backtest(
            history, cnr, entry_date=entry_date, base_cnr_month=base_month,
            diesel_weight=weight_pct / 100, positions=positions, eurusd=eurusd,
            cnr_lag_months=cnr_lag,
        )
    except (ValueError, KeyError) as exc:
        st.error(f"Backtest impossible : {exc}")
        return

    monthly = result.monthly
    k1, k2, k3 = st.columns(3)
    k1.metric("Écart de facture CNR cumulé", f"{monthly['surcharge_eur'].sum():,.0f} €")
    k2.metric("P&L du hedge ICE cumulé", f"{monthly['pnl_eur'].sum():,.0f} €")
    k3.metric("Coût net après hedge", f"{monthly['residual_eur'].sum():,.0f} €")
    st.plotly_chart(_cumulative_chart(monthly), width="stretch")
    st.caption(
        "Coût net = variation des factures CNR − P&L du hedge. Une valeur négative est un gain net "
        "par rapport au budget de base ; aucun surcoût n'est masqué ou plafonné à zéro."
    )

    st.markdown("### Détail par facture et par échéance")
    _show_monthly_details(monthly)
    selected_month = st.selectbox("Suivre un contrat sur toute sa vie dans le strip", monthly["month"].tolist())
    selected = monthly.loc[monthly["month"] == selected_month].iloc[0]
    st.caption(
        f"Facture {selected_month} : achat de {selected['ticker']} le {selected['entry_date']:%d/%m/%Y}, "
        f"sortie le {selected['exit_date']:%d/%m/%Y}. Dans la série générique, son rang passe "
        f"de CO{selected['rank_entry']} à CO{selected['rank_exit']} sans vente ni roll intermédiaire."
    )
    st.plotly_chart(_contract_chart(result.daily, selected_month), width="stretch")
    with st.expander("Méthode et limites de cette simulation"):
        st.markdown(
            "- **Instrument :** ICE Brent standard 1 000 barils, avec quantité fractionnaire théorique. "
            "0,1 ICE correspond à 100 barils de risque, soit la taille d'un mini MBC, mais les règlements diffèrent.\n"
            "- **Dimensionnement :** le β CNR / Brent en EUR est réestimé sur les seuls mois antérieurs à l'achat, "
            "avec un mois de recul supplémentaire faute d'historique des dates de publication. "
            "Barils indicatifs = facture × part gazole × β × EUR/USD à l'achat / prix du future détenu. "
            "La saisie manuelle peut remplacer ces quantités.\n"
            "- **Échéance :** pour une facture du mois M, on suit le future ICE de livraison M+2. "
            "Les CO1 à CO12 sont des rangs mobiles : le contrat détenu est retrouvé à chaque date avec la liste des expirations.\n"
            "- **P&L :** variation quotidienne de PX_LAST × barils. Chaque variation en dollars est "
            "convertie en euros au taux du jour, ou au taux constant choisi. Aucun roll n'est simulé.\n"
            "- **Facture :** facture initiale × part gazole × (CNR du mois retenu, après "
            "décalage / CNR de base − 1). "
            "Le coût net est cette variation de facture moins le P&L du future.\n"
            "- **Écart de courbe :** la colonne 'Écart vs CO1' compare deux échéances au jour d'achat. "
            "En dollars, P&L = mouvement de CO1 + évolution de l'écart entre l'échéance détenue "
            "et CO1. La pente à l'achat indique contango ou backwardation entre ces deux échéances. "
            "Cette décomposition n'est pas un débit de roll ni un rendement garanti de la pente.\n"
            "- **Limites :** PX_LAST n'est pas vérifié ici comme prix officiel de règlement ni comme prix "
            "réellement négociable. Frais, spread achat/vente, marge, financement et publication "
            "effective du CNR sont exclus. Les dates d'expiration proviennent de la colonne "
            "'Expiration théo.' du classeur, corrigée lorsque l'expiration tombe la veille "
            "ouvrée de Noël ou du Nouvel An, puis ramenée à la dernière séance disponible "
            "si le jour théorique n'est pas coté."
        )
