"""Illustrative transport contract and 100-barrel Brent hedge estimate."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import pandas as pd


MINI_BRENT_BARRELS = 100


@dataclass(frozen=True)
class ContractScenario:
    ledger: pd.DataFrame
    annual_base_eur: float
    annual_invoice_eur: float
    surcharge_eur: float
    first_contracts: int
    first_target_contracts: float
    proxy_pnl_eur: float
    residual_before_costs_eur: float


def _whole_contracts(target: float, rounding: str) -> int:
    if rounding == "floor":
        return floor(target)
    if rounding == "nearest":
        return floor(target + 0.5)
    if rounding == "continuous":
        return ceil(target)
    raise ValueError("Règle de dimensionnement inconnue.")


def simulate_contract(
    levels: pd.DataFrame,
    start: pd.Period,
    annual_base_eur: float,
    diesel_weight: float,
    beta: float,
    billing: str,
    revision: str,
    rounding: str = "floor",
    annual_reference: str = "anniversary",
) -> ContractScenario:
    """Model twelve monthly services, with monthly risk fixing and no gaps.

    The Brent hedge P&L is estimated from a continuous monthly EUR price
    series. Actual per-expiry futures marks, FX conversions and costs are absent.
    """
    if billing not in {"monthly", "annual"} or revision not in {"monthly", "annual"}:
        raise ValueError("La facturation et la révision doivent être mensuelles ou annuelles.")
    if annual_reference not in {"anniversary", "average"}:
        raise ValueError("Référence annuelle inconnue.")
    if annual_base_eur <= 0 or not 0 <= diesel_weight <= 1 or beta < 0:
        raise ValueError("Montant, pondération ou coefficient invalide.")
    if start.freqstr != "M":
        raise ValueError("La date de début doit être un mois.")
    months = pd.period_range(start, periods=12, freq="M")
    required = pd.period_range(start - 1, periods=13, freq="M")
    if not required.isin(levels.index).all() or levels.loc[required, ["cnr", "brent"]].isna().any().any():
        raise ValueError("Treize mois complets CNR et Brent sont nécessaires.")
    if (levels.loc[required, ["cnr", "brent"]] <= 0).any().any():
        raise ValueError("Les indices et les prix Brent doivent être positifs.")

    base_cnr = float(levels.loc[start - 1, "cnr"])
    monthly_base = annual_base_eur / 12
    year_cnr = levels.loc[months, "cnr"].astype(float)
    annual_cnr = float(year_cnr.iloc[-1] if annual_reference == "anniversary" else year_cnr.mean())
    rows = []
    for i, month in enumerate(months):
        cnr = float(levels.loc[month, "cnr"])
        brent_before = float(levels.loc[month - 1, "brent"])
        brent_after = float(levels.loc[month, "brent"])
        applicable_cnr = cnr if revision == "monthly" else annual_cnr
        charge = monthly_base * (1 + diesel_weight * (applicable_cnr / base_cnr - 1))
        remaining_base = monthly_base * (12 - i)
        target = remaining_base * diesel_weight * beta / (MINI_BRENT_BARRELS * brent_before)
        contracts = _whole_contracts(target, rounding)
        rows.append({
            "month": str(month),
            "cnr": cnr,
            "brent_eur_before": brent_before,
            "brent_eur_after": brent_after,
            "base_eur": monthly_base,
            "revised_charge_eur": charge,
            "target_contracts": target,
            "mini_contracts": contracts,
            "proxy_pnl_eur": contracts * MINI_BRENT_BARRELS * (brent_after - brent_before),
        })
    ledger = pd.DataFrame(rows)
    total_invoice = float(ledger["revised_charge_eur"].sum())
    if billing == "annual":
        ledger["cash_invoice_eur"] = [0.0] * 11 + [total_invoice]
    elif revision == "annual":
        adjustment = total_invoice - annual_base_eur
        ledger["cash_invoice_eur"] = [monthly_base] * 11 + [monthly_base + adjustment]
    else:
        ledger["cash_invoice_eur"] = ledger["revised_charge_eur"]
    ledger["invoice_gap_eur"] = ledger["revised_charge_eur"] - ledger["base_eur"]
    ledger["total_pnl_eur"] = ledger["proxy_pnl_eur"] - ledger["invoice_gap_eur"]
    surcharge = total_invoice - annual_base_eur
    proxy_pnl = float(ledger["proxy_pnl_eur"].sum())
    return ContractScenario(
        ledger=ledger,
        annual_base_eur=annual_base_eur,
        annual_invoice_eur=total_invoice,
        surcharge_eur=surcharge,
        first_contracts=int(ledger.loc[0, "mini_contracts"]),
        first_target_contracts=float(ledger.loc[0, "target_contracts"]),
        proxy_pnl_eur=proxy_pnl,
        residual_before_costs_eur=surcharge - proxy_pnl,
    )
