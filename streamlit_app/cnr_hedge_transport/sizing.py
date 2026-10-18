"""Dimensionnement indicatif d'un strip Brent à partir du β CNR / Brent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import pandas as pd

from .backtest import (
    CONTRACT_BARRELS_ICE,
    CONTRACT_BARRELS_MINI,
    IceHistory,
    _contract_path,
    _fx_on_dates,
)
from .study import lag_regressions


@dataclass(frozen=True)
class BetaCalibration:
    beta: float
    r2: float
    observations: int
    last_month: pd.Period


def estimate_pre_entry_beta(levels: pd.DataFrame, entry_date: date) -> BetaCalibration:
    """Estime le β contemporain sur les seuls mois antérieurs à l'achat.

    Le mois immédiatement précédent est exclu par prudence : les dates de
    publication effectives du CNR ne figurent pas dans les données locales.
    """
    last_month = pd.Period(entry_date, freq="M") - 2
    known_levels = levels.loc[levels.index <= last_month]
    regressions, _ = lag_regressions(known_levels, max_lag=0, min_observations=24)
    row = regressions.iloc[0]
    beta = float(row["β"])
    r2 = float(row["R²"])
    if not isfinite(beta) or not isfinite(r2) or beta <= 0:
        raise ValueError("Le β pré-achat est nul, négatif ou non calculable : saisis les positions manuellement.")
    return BetaCalibration(beta, r2, int(row["N"]), last_month)


def recommend_strip_positions(
    history: IceHistory,
    invoices: pd.DataFrame,
    *,
    entry_date: date,
    diesel_weight: float,
    beta: float,
    eurusd: float | pd.Series,
) -> pd.DataFrame:
    """Convertit β et l'exposition de chaque facture en barils ICE théoriques.

    β relie des variations relatives du CNR et du Brent en EUR. Le montant
    sensible de la facture est facture × part gazole. Pour une petite variation
    du future, barils ≈ montant sensible × β / prix ICE en EUR par baril.
    """
    if not 0 <= diesel_weight <= 1 or not isfinite(beta) or beta < 0:
        raise ValueError("Part gazole ou β de dimensionnement invalide.")
    if not {"month", "invoice_eur"}.issubset(invoices.columns) or invoices.empty:
        raise ValueError("Il faut renseigner les mois et les factures initiales.")
    entry = pd.Timestamp(entry_date).normalize()
    fx_at_entry = float(_fx_on_dates(eurusd, pd.DatetimeIndex([entry])).iloc[0])
    rows = []
    for position in invoices.itertuples(index=False):
        month = pd.Period(position.month, freq="M")
        invoice = float(position.invoice_eur)
        if not isfinite(invoice) or invoice < 0:
            raise ValueError("Les factures initiales doivent être positives ou nulles.")
        path, ticker, _ = _contract_path(history, entry, month)
        price = float(path["price_usd_bbl"].iloc[0])
        exposure = invoice * diesel_weight
        barrels = exposure * beta * fx_at_entry / price
        rows.append(
            {
                "month": str(month),
                "invoice_eur": invoice,
                "ticker": ticker,
                "diesel_exposure_eur": exposure,
                "entry_price_usd_bbl": price,
                "eurusd_entry": fx_at_entry,
                "barrels": barrels,
                "ice_contracts_fractional": barrels / CONTRACT_BARRELS_ICE,
                "mini_equivalents": barrels / CONTRACT_BARRELS_MINI,
            }
        )
    return pd.DataFrame(rows)
