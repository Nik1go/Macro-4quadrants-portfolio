"""Instantané local lu par Streamlit, sans téléchargement ni date d'expiration."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

import pandas as pd

from .core import DATA_QUALITY_VERSION, correct_known_prices, prepare_data

SNAPSHOT_PATH = Path(__file__).parent / "snapshot" / "market_data.zip"
MEMBERS = {"prices.csv", "metadata.json", "universe.csv"}


def read_snapshot(path: Path = SNAPSHOT_PATH) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Lire seulement le fichier local ; un fichier absent ou invalide reste une erreur.

    Aucune solution de secours réseau, même si l'instantané est ancien.
    Le format contient uniquement des CSV et du JSON, jamais de pickle.
    """
    with ZipFile(path) as archive:
        if set(archive.namelist()) != MEMBERS:
            raise ValueError("L'instantané local doit contenir prices.csv, metadata.json et universe.csv.")
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("snapshot_schema") != 1:
            raise ValueError("Version de l'instantané local non prise en charge.")
        prices = pd.read_csv(BytesIO(archive.read("prices.csv")), index_col="Date", parse_dates=True)
        universe = pd.read_csv(BytesIO(archive.read("universe.csv")))
    return prices, metadata, universe


def write_snapshot(prices: pd.DataFrame, metadata: dict, universe: pd.DataFrame,
                   path: Path = SNAPSHOT_PATH) -> dict:
    """Valider puis remplacer atomiquement l'instantané depuis une commande locale.

    Une acquisition incomplète ne doit pas écraser le dernier fichier exploitable.
    Les cours et les métadonnées sont publiés ensemble dans une seule archive.
    """
    prices, corrections = correct_known_prices(prices)
    eligible, _, _ = prepare_data(prices, universe)
    previous = metadata.get("data_corrections", [])
    corrections = previous + [entry for entry in corrections if entry not in previous]
    metadata = {**metadata,
                "data_quality_version": DATA_QUALITY_VERSION,
                "data_corrections": corrections,
                "snapshot_schema": 1,
                "snapshot_created_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                "snapshot_first_session": eligible.index[0].date().isoformat(),
                "snapshot_last_session": eligible.index[-1].date().isoformat(),
                "snapshot_eligible_assets": eligible.shape[1],
                "update_policy": "Commande locale uniquement ; aucun téléchargement depuis Streamlit."}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Le fichier temporaire est sur le même volume pour permettre os.replace.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".snapshot-", suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr("prices.csv", prices.to_csv(index_label="Date"))
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            archive.writestr("universe.csv", universe.to_csv(index=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata
