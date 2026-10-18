"""Étude descriptive du lien Brent / indice gazole CNR."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "cnr", "brent")
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_official_study() -> pd.DataFrame:
    """Align the published monthly CNR index with the saved Yahoo Brent/EUR series."""
    cnr = pd.read_csv(DATA_DIR / "cnr_gazole_professionnel_monthly.csv")
    brent = pd.read_csv(DATA_DIR / "brent_bz_eur_monthly.csv")
    if not {"date", "cnr"}.issubset(cnr.columns):
        raise ValueError("Le fichier CNR officiel est incomplet.")
    if not {"date", "brent_eur", "daily_observations"}.issubset(brent.columns):
        raise ValueError("Le fichier Brent en EUR est incomplet.")
    if (brent["daily_observations"] < 10).any():
        raise ValueError("Un mois de cotations Brent est incomplet.")
    monthly = cnr.merge(brent, on="date", how="inner", validate="one_to_one")
    monthly = monthly.rename(columns={"brent_eur": "brent"})
    levels = load_study_csv(monthly.loc[:, list(REQUIRED_COLUMNS)].to_csv(index=False).encode())
    if levels.isna().any().any():
        raise ValueError("Les séries CNR et Brent contiennent un mois manquant.")
    return levels


def load_study_csv(content: bytes) -> pd.DataFrame:
    """Lit des niveaux mensuels positifs, sans inventer les mois manquants."""
    frame = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    frame.columns = frame.columns.str.strip().str.lower()
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(sorted(missing))}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    for column in ("cnr", "brent"):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    if frame.isna().any().any():
        raise ValueError("Chaque ligne doit avoir une date et deux niveaux numériques valides.")
    if (frame[["cnr", "brent"]] <= 0).any().any():
        raise ValueError("Les niveaux CNR et Brent doivent être strictement positifs.")
    frame["date"] = frame["date"].dt.to_period("M")
    if frame["date"].duplicated().any():
        raise ValueError("Une seule observation est attendue par mois.")
    frame = frame.sort_values("date").set_index("date")
    months = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    return frame.reindex(months)


def lag_regressions(
    levels: pd.DataFrame, max_lag: int = 6, *,
    lag_label: str = "Décalage (mois)", min_observations: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS sur variations périodiques, avec un échantillon identique par décalage.

    k est un décalage de périodes dans la fréquence de la série fournie.
    Les niveaux ne sont pas régressés entre eux : leurs tendances communes
    pourraient créer une corrélation trompeuse.
    """
    if max_lag < 0:
        raise ValueError("Le décalage maximal doit être positif ou nul.")
    if not {"cnr", "brent"}.issubset(levels.columns):
        raise ValueError("Les colonnes cnr et brent sont requises.")
    changes = levels[["cnr", "brent"]].pct_change(fill_method=None) * 100
    aligned = pd.DataFrame({"cnr_change": changes["cnr"]})
    for lag in range(max_lag + 1):
        aligned[f"brent_lag_{lag}"] = changes["brent"].shift(lag)
    aligned = aligned.dropna()
    if len(aligned) < min_observations:
        raise ValueError(
            f"Il faut au moins {min_observations} périodes complètes communes après les {max_lag} décalages ; "
            f"{len(aligned)} sont disponibles."
        )

    y = aligned["cnr_change"].to_numpy(dtype=float)
    rows = []
    for lag in range(max_lag + 1):
        x = aligned[f"brent_lag_{lag}"].to_numpy(dtype=float)
        if np.allclose(x, x[0]):
            continue
        beta, alpha = np.polyfit(x, y, 1)
        fitted = alpha + beta * x
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - fitted) ** 2)) / total if total > 0 else float("nan")
        rows.append(
            {
                lag_label: lag,
                "N": len(y),
                "α (points de %)": float(alpha),
                "β": float(beta),
                "R²": float(r2),
                "RMSE (points de %)": float(np.sqrt(np.mean((y - fitted) ** 2))),
            }
        )
    if not rows:
        raise ValueError("La variation du Brent est constante : régression impossible.")
    return pd.DataFrame(rows), aligned


def load_fortnightly_study() -> pd.DataFrame:
    """Align CNR and Brent over calendar days 1-15 and 16-month end."""
    cnr = pd.read_csv(DATA_DIR / "cnr_gazole_professionnel_fortnightly.csv")
    brent = pd.read_csv(DATA_DIR / "brent_bz_eur_fortnightly.csv")
    if not {"date", "cnr"}.issubset(cnr.columns):
        raise ValueError("Le fichier CNR bimensuel est incomplet.")
    if not {"date", "brent_eur", "daily_observations"}.issubset(brent.columns):
        raise ValueError("Le fichier Brent bimensuel est incomplet.")
    if (brent["daily_observations"] < 6).any():
        raise ValueError("Une quinzaine Brent manque de cotations.")
    frame = cnr.merge(brent, on="date", how="outer", validate="one_to_one")
    if frame["date"].isna().any() or not frame["date"].str.fullmatch(r"\d{4}-\d{2}-H[12]").all():
        raise ValueError("Dates de quinzaine invalides.")
    for column in ("cnr", "brent_eur"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["cnr", "brent_eur"]].isna().any().any():
        raise ValueError("Les séries CNR et Brent ne couvrent pas les mêmes quinzaines.")
    if (frame[["cnr", "brent_eur"]] <= 0).any().any():
        raise ValueError("Les niveaux CNR et Brent doivent être positifs.")
    frame = frame.sort_values("date").set_index("date")
    months = pd.period_range(frame.index.min()[:7], frame.index.max()[:7], freq="M")
    expected = [f"{month}-H{half}" for month in months for half in (1, 2)]
    expected = [slot for slot in expected if frame.index.min() <= slot <= frame.index.max()]
    if frame.index.tolist() != expected:
        raise ValueError("Une quinzaine manque dans les séries CNR et Brent.")
    return frame.rename(columns={"brent_eur": "brent"})[["cnr", "brent"]]
