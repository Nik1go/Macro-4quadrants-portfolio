"""Actualisation manuelle des cours : outil local, jamais appelé par Streamlit."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from streamlit_app.ant_colony.core import fetch_market_data, paris_today
from streamlit_app.ant_colony.snapshot import SNAPSHOT_PATH, write_snapshot


def main():
    parser = argparse.ArgumentParser(description="Mettre à jour manuellement l'instantané ACO Yahoo / Euronext.")
    parser.add_argument("--end-exclusive", type=date.fromisoformat,
                        help="Borne de fin exclusive YYYY-MM-DD ; jour courant à Paris par défaut.")
    arguments = parser.parse_args()
    end = arguments.end_exclusive.isoformat() if arguments.end_exclusive else paris_today()
    universe = pd.read_csv(Path(__file__).with_name("universe.csv"))
    # Le téléchargement existe uniquement dans cette commande et le notebook local.
    prices, metadata = fetch_market_data(universe, end=end)
    saved = write_snapshot(prices, metadata, universe)
    print(f"Instantané validé : {SNAPSHOT_PATH}")
    print(f"Période : {saved['snapshot_first_session']} → {saved['snapshot_last_session']}")
    print(f"Actions éligibles : {saved['snapshot_eligible_assets']}/100")


if __name__ == "__main__":
    main()
