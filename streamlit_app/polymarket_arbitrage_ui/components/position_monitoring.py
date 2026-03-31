"""Position monitoring view - Macro style."""

from __future__ import annotations
import streamlit as st
from ..data_loader import load_open_positions, load_trades_history

def render_position_monitoring(db_path: str | None = None) -> None:
    """Render metrics and tables for current arbitrage activity."""
    
    st.header("Monitoring Temps Réel")

    # Load data
    positions_df = load_open_positions(db_path=db_path)
    trades_df = load_trades_history(db_path=db_path)

    # Summary Metrics Row
    m1, m2, m3 = st.columns(3)
    
    total_realized_pnl = trades_df["realized_pnl"].sum() if not trades_df.empty else 0.0
    total_fees = trades_df["fees_paid"].sum() if not trades_df.empty else 0.0
    win_rate = (trades_df["realized_pnl"] > 0).mean() if len(trades_df) > 0 else 0.0

    m1.metric("PnL Réalisé Global", f"${total_realized_pnl:,.2f}")
    m2.metric("Frais Totaux (Gas/Taker)", f"${total_fees:,.2f}", delta=f"{(total_fees / 10000):.2%}", delta_color="inverse")
    m3.metric("Win Rate Provisoire", f"{win_rate:.2%}")

    st.divider()

    # Open Positions
    st.subheader("Positions Ouvertes")
    if positions_df.empty:
        st.info("Aucune position ouverte actuellement.")
    else:
        st.dataframe(
            positions_df.style.format({
                "size_usd": "${:,.2f}",
                "entry_price": "{:.4f}",
                "current_price": "{:.4f}",
                "unrealized_pnl": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Trades History
    st.subheader("Historique des Trades")
    if trades_df.empty:
        st.info("Aucun trade exécuté encore.")
    else:
        st.dataframe(
            trades_df.style.format({
                "size": "${:,.2f}",
                "entry_price": "{:.4f}",
                "exit_price": "{:.4f}",
                "realized_pnl": "${:,.2f}",
                "fees_paid": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )
