"""Delta-neutral strategy implementation.

Logic:
- Long/short Polymarket when model edge exists.
- Hedge delta on Binance perpetuals using opposite direction.
- Reject opportunities if funding is extreme, liquidity is weak,
  or slippage uncertainty invalidates expected edge.
"""

from __future__ import annotations

import math
from typing import Dict

from .base_strategy import BaseStrategy, StrategyContext, StrategySignal

try:
    from ..utils.config import Config
    from ..utils.logger import Logger
except ImportError:  # pragma: no cover
    from utils.config import Config
    from utils.logger import Logger

logger = Logger().logger


class DeltaNeutralStrategy(BaseStrategy):
    """Cross-venue delta-neutral arbitrage strategy."""

    @property
    def name(self) -> str:
        return "delta_neutral"

    def min_required_edge(self) -> float:
        return Config.DELTA_NEUTRAL_MIN_EDGE

    @staticmethod
    def _is_invalid_number(value: float) -> bool:
        return value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value)

    def _validate_context(self, context: StrategyContext) -> Dict[str, float]:
        """Validate and sanitize numeric context values."""
        spot = float(context.spot_price)
        theo = float(context.theoretical_prob)
        market = float(context.polymarket_price)
        funding = float(context.funding_rate)

        if any(self._is_invalid_number(v) for v in [spot, theo, market, funding]):
            raise ValueError("context contains invalid numeric values")

        theo = min(max(theo, 1e-4), 1 - 1e-4)
        market = min(max(market, 1e-4), 1 - 1e-4)

        return {
            "spot": spot,
            "theoretical_prob": theo,
            "market_price": market,
            "funding_rate": funding,
        }

    def generate_signal(self, context: StrategyContext) -> StrategySignal:
        try:
            values = self._validate_context(context)
        except Exception as exc:
            logger.warning("%s invalid context slug=%s reason=%s", self.name, context.slug, exc)
            return StrategySignal(False, self.name, "hold", "invalid_context", 0.0, {})

        if not context.orderbook_depth_ok:
            logger.info("%s hold slug=%s reason=insufficient_liquidity", self.name, context.slug)
            return StrategySignal(False, self.name, "hold", "insufficient_liquidity", 0.0, {})

        if abs(values["funding_rate"]) > Config.MAX_FUNDING_RATE_ABS:
            logger.info(
                "%s hold slug=%s reason=funding_too_expensive funding=%.6f",
                self.name,
                context.slug,
                values["funding_rate"],
            )
            return StrategySignal(
                False,
                self.name,
                "hold",
                "funding_too_expensive",
                0.0,
                {"funding_rate": values["funding_rate"]},
            )

        edge = values["theoretical_prob"] - values["market_price"]
        if abs(edge) < self.min_required_edge():
            return StrategySignal(False, self.name, "hold", "edge_below_threshold", 0.0, {"edge": edge})

        if edge > 0:
            direction = "buy_yes_hedge_short_binance"
            hedge_side = "sell"
            target_delta = -1.0
        else:
            direction = "buy_no_hedge_long_binance"
            hedge_side = "buy"
            target_delta = 1.0

        logger.info(
            "%s signal slug=%s direction=%s edge=%.6f funding=%.6f",
            self.name,
            context.slug,
            direction,
            edge,
            values["funding_rate"],
        )

        return StrategySignal(
            True,
            self.name,
            direction,
            "delta_neutral_edge",
            target_delta,
            {
                "edge": edge,
                "hedge_side": hedge_side,
                "funding_rate": values["funding_rate"],
                "spot": values["spot"],
                "slug": context.slug,
            },
        )
