"""The retained LB identity, scoped to its own sidebar element."""
import streamlit as st


def render_portfolio_brand():
    st.markdown('''<style>
    .portfolio-brand {display:flex;gap:12px;align-items:center;margin:0 0 14px;}
    .portfolio-brand .brand-symbol {display:flex;align-items:center;gap:2px;
      background:#172b40;color:white !important;border-radius:11px;padding:13px 10px;
      font-family:'Segoe UI',sans-serif;font-weight:700;font-size:1.1rem;}
    .portfolio-brand .brand-symbol span {color:#61d3c5 !important;font-size:.7rem;}
    .portfolio-brand strong {display:block;color:#e8e8e8 !important;font-family:'Segoe UI',sans-serif;
      font-size:1.02rem;letter-spacing:-.025em;}
    .portfolio-brand small {display:block;color:#b8c1e3 !important;font-family:'Segoe UI',sans-serif;
      font-size:.54rem;font-weight:700;letter-spacing:.12em;margin-top:4px;}
    </style><div class="portfolio-brand"><div class="brand-symbol">LB<span>◈</span></div><div><strong>Léo Bertrand</strong><small>FINANCE PORTFOLIO</small></div></div>''', unsafe_allow_html=True)
