"""The supplied Greeks analyzer embedded in the portfolio's native navigation."""
from pathlib import Path
import re

import streamlit as st

ASSETS = Path(__file__).resolve().parent
SOURCE = (ASSETS / 'analyzer.html').read_text(encoding='utf-8')
CSS = re.search(r'<style>(.*?)</style>', SOURCE, re.S).group(1)
BODY = re.search(r'<body>(.*?)<script>', SOURCE, re.S).group(1)
SCRIPT = re.findall(r'<script>(.*?)</script>', SOURCE, re.S)[-1]

# Bundle D3 locally: the analyzer also works without a CDN connection. The
# CommonJS branch of its UMD build keeps the library local to this component.
D3 = (ASSETS / 'vendor/d3.v7.min.js').read_text(encoding='utf-8')
JS = (
    'const d3 = (() => { const module = {exports: {}}; '
    'const exports = module.exports;\n' + D3 + '\nreturn module.exports; })();\n'
    'export default function({ parentElement }) {\n'
    '  const document = {\n'
    '    getElementById: id => parentElement.querySelector(`[id="${id}"]`),\n'
    '    querySelectorAll: selector => parentElement.querySelectorAll(selector),\n'
    '    createElement: tag => globalThis.document.createElement(tag)\n'
    '  };\n' + SCRIPT + '\n'
    '  const root = document.getElementById("greeks-app");\n'
    '  let previousWidth = root.clientWidth;\n'
    '  const observer = new ResizeObserver(() => {\n'
    '    if (root.clientWidth !== previousWidth && root.clientWidth > 0) {\n'
    '      previousWidth = root.clientWidth; initCharts(); updateCharts();\n'
    '    }\n'
    '  });\n'
    '  observer.observe(root);\n'
    '  return () => { observer.disconnect(); };\n'
    '}\n'
)

analyzer = st.components.v2.component(
    'portfolio_greeks_revision', html=BODY, css=CSS, js=JS, isolate_styles=True,
)


def render():
    st.title('Greeks Revision')
    st.caption('Options européennes · Portefeuille multi-legs · Sensibilités individuelles et agrégées')
    analyzer(key='greeks_revision_analyzer', height='content')
