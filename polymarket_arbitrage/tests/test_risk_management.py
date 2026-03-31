"""Unit tests for risk management module."""

from polymarket_arbitrage.risk_management import DrawdownManager, PositionSizer, RiskManager


def test_kelly_fraction_non_negative() -> None:
    sizer = PositionSizer(max_kelly_fraction=0.5)
    frac = sizer.calculate_kelly_fraction(win_probability=0.4, market_price=0.8)
    assert frac >= 0.0


def test_drawdown_breach_detection() -> None:
    manager = DrawdownManager(initial_capital=1000.0, max_drawdown_pct=0.2)
    manager.update_drawdown(1000.0)
    state = manager.update_drawdown(700.0)
    assert state["breached"] is True


def test_liquidity_and_slippage() -> None:
    rm = RiskManager()
    orderbook = {"asks": [[0.55, 10], [0.56, 10]], "bids": [[0.54, 10], [0.53, 10]]}
    avg_price, ok = rm.check_liquidity_and_slippage(orderbook, target_size=5, side="buy")
    assert ok is True
    assert 0.0 < avg_price < 1.0
