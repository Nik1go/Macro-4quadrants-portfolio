"""Pondérations gazole annuelles officielles du Comité national routier."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


CONVOY_LABELS = {
    "ld_ea_44t": "Ensemble articulé longue distance 44 t",
    "reg_ea_44t": "Ensemble articulé régional 44 t",
    "reg_porteur_19t": "Porteur régional 19 t",
}
DATA_PATH = Path(__file__).with_name("data") / "cnr_gazole_weights.csv"
SOURCE_URL = (
    "https://www.cnr.fr/download/file/publications/"
    "CNR%20-%20Historique%20des%20pond%C3%A9rations%20indiciaires%20TRM%20-%202026.pdf"
)


@dataclass(frozen=True)
class WeightObservation:
    convoy_type: str
    effective_date: date
    diesel_weight_pct: float

    @property
    def diesel_weight(self) -> float:
        return self.diesel_weight_pct / 100


def load_weights(path: Path = DATA_PATH) -> tuple[WeightObservation, ...]:
    observations = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = WeightObservation(
                convoy_type=row["convoy_type"],
                effective_date=date.fromisoformat(row["effective_date"]),
                diesel_weight_pct=float(row["diesel_weight_pct"]),
            )
            if item.convoy_type not in CONVOY_LABELS:
                raise ValueError(f"Type de convoi CNR inconnu : {item.convoy_type}")
            if not 0 <= item.diesel_weight_pct <= 100:
                raise ValueError("Une pondération doit être comprise entre 0 et 100 %")
            observations.append(item)

    observations.sort(key=lambda item: (item.convoy_type, item.effective_date))
    keys = {(item.convoy_type, item.effective_date) for item in observations}
    if len(keys) != len(observations):
        raise ValueError("La série CNR contient des doublons")
    return tuple(observations)


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def weight_for_contract(
    convoy_type: str,
    contract_date: date | datetime | str,
    observations: tuple[WeightObservation, ...] | None = None,
) -> WeightObservation:
    """Pondération du millésime applicable à la date du contrat.

    Les pondérations CNR étant publiées annuellement, chaque millésime est
    représenté à compter du 1er janvier, sans interpolation entre deux années.
    """
    if convoy_type not in CONVOY_LABELS:
        raise ValueError(f"Type de convoi inconnu : {convoy_type}")
    cutoff = _as_date(contract_date)
    eligible = [
        item
        for item in (observations if observations is not None else load_weights())
        if item.convoy_type == convoy_type and item.effective_date <= cutoff
    ]
    if not eligible:
        raise LookupError(
            f"Aucune pondération CNR disponible pour ce convoi au {cutoff:%d/%m/%Y}."
        )
    return max(eligible, key=lambda item: item.effective_date)


def history_for_convoy(convoy_type: str) -> tuple[WeightObservation, ...]:
    if convoy_type not in CONVOY_LABELS:
        raise ValueError(f"Type de convoi inconnu : {convoy_type}")
    return tuple(item for item in load_weights() if item.convoy_type == convoy_type)
