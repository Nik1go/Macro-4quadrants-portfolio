"""Execute the original Risk interface on every Streamlit rerun.

The engine, notebook, dataset and analytical content are preserved; visual styling
is adapted to the portfolio.
Using runpy preserves __file__-relative data and export paths; a normal import
would render only once, breaking widgets on subsequent reruns.
"""
from pathlib import Path
import runpy
import sys

import streamlit as st


def render():
    directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, directory)
    try:
        runpy.run_path(str(Path(directory) / "app.py"), run_name="__portfolio_risk__")
    finally:
        sys.path.remove(directory)
        # The imported interface has its own page config; retain our shared icon.
        st.set_page_config(
            page_title="Portfolio Risk Engine · Finance Portfolio",
            page_icon=str(Path(directory).parent / "images/portfolio-brand.png"),
        )
