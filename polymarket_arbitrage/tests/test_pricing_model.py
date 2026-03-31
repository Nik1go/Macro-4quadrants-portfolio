"""Unit tests for the pricing model module."""

from datetime import datetime, timedelta, timezone

from polymarket_arbitrage.pricing_model import PricingModel


def test_binary_call_bounds_and_ordering() -> None:
    model = PricingModel()
    deep_itm = model.binary_call_option_price(s=200.0, k=100.0, t=0.5, r=0.05, sigma=0.5)
    deep_otm = model.binary_call_option_price(s=80.0, k=100.0, t=0.5, r=0.05, sigma=0.5)

    assert 0.0 < deep_itm < 1.0
    assert 0.0 < deep_otm < 1.0
    assert deep_itm > deep_otm


def test_time_to_maturity_non_negative() -> None:
    model = PricingModel()
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=7)

    assert model.calculate_time_to_maturity(past) == 0.0
    assert model.calculate_time_to_maturity(future) > 0.0


def test_implied_volatility_returns_value() -> None:
    model = PricingModel()
    s, k, t, r, sigma = 100.0, 100.0, 0.25, 0.03, 0.6
    market = model.binary_call_option_price(s=s, k=k, t=t, r=r, sigma=sigma)
    iv = model.calculate_implied_volatility(market_price=market, s=s, k=k, t=t, r=r)

    assert iv is not None
    assert 0.0 < iv < 5.0
