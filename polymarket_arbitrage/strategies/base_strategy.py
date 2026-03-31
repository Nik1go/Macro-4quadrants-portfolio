"""Base abstractions for arbitrage strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StrategyContext:
    """Container for per-market inputs used by strategy decision logic."""

    asset: str
    symbol: str
    slug: str
    title: str
    strike: float
    spot_price: float
    theoretical_prob: float
    polymarket_price: float
    time_to_maturity: float
    implied_volatility: float
    net_spread: float
    funding_rate: float
    orderbook_depth_ok: bool


@dataclass
class StrategySignal:
    """Normalized signal emitted by a strategy."""

    should_trade: bool
    strategy_name: str
    direction: str
    reason: str
    target_delta: float
    metadata: Dict[str, Any]


class BaseStrategy(ABC):
    """Abstract base class for all strategy implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> StrategySignal:
        """Create a trade signal from validated market context."""

    @abstractmethod
    def min_required_edge(self) -> float:
        """Return minimum edge threshold required by this strategy."""
