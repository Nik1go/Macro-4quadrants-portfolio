"""Pure Polymarket directional mispricing strategy.

Trades only on Polymarket and avoids hedging costs. Includes additional
liquidity checks to avoid markets with insufficient executable depth.
"""

from __future__ import annotations

import math

from .base_strategy import BaseStrategy, StrategyContext, StrategySignal

try:
    from ..utils.config import Config
    from ..utils.logger import Logger
except ImportError:  # pragma: no cover
    from utils.config import Config
    from utils.logger import Logger

logger = Logger().logger


class PurePolymarketStrategy(BaseStrategy):
    """Directional strategy that trades only on Polymarket mispricing."""

    @property
    def name(self) -> str:
        return "pure_polymarket"

    def min_required_edge(self) -> float:
        return Config.PURE_POLY_MIN_EDGE

    @staticmethod
    def _valid(value: float) -> bool:
        return value is not None and isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value)

    def generate_signal(self, context: StrategyContext) -> StrategySignal:
        if not context.orderbook_depth_ok:
            logger.info("%s hold slug=%s reason=insufficient_liquidity", self.name, context.slug)
            return StrategySignal(False, self.name, "hold", "insufficient_liquidity", 0.0, {})

        if not all(
            [
                self._valid(context.theoretical_prob),
                self._valid(context.polymarket_price),
                self._valid(context.spot_price),
            ]
        ):
            logger.warning("%s hold slug=%s reason=invalid_context", self.name, context.slug)
            return StrategySignal(False, self.name, "hold", "invalid_context", 0.0, {})

        theoretical_prob = float(min(max(context.theoretical_prob, 1e-4), 1 - 1e-4))
        market_price = float(min(max(context.polymarket_price, 1e-4), 1 - 1e-4))
        edge = theoretical_prob - market_price

        if abs(edge) < self.min_required_edge():
            return StrategySignal(False, self.name, "hold", "edge_below_threshold", 0.0, {"edge": edge})

        direction = "buy_yes" if edge > 0 else "buy_no"

        logger.info(
            "%s signal slug=%s direction=%s edge=%.6f",
            self.name,
            context.slug,
            direction,
            edge,
        )

        return StrategySignal(
            True,
            self.name,
            direction,
            "pure_polymarket_mispricing",
            0.0,
            {"edge": edge, "spot": context.spot_price, "slug": context.slug},
        )
