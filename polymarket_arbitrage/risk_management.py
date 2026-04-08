"""Risk management module for multi-strategy Polymarket arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .utils.config import Config
    from .utils.logger import Logger
except ImportError:  # pragma: no cover
    from utils.config import Config
    from utils.logger import Logger

logger = Logger().logger


@dataclass
class Position:
    """Represent an open position tracked by risk manager."""

    position_id: str
    asset: str
    direction: str
    strategy: str
    size_usd: float
    entry_price: float
    stop_loss_price: float
    opened_at: datetime
    target_delta: float = 0.0
    ends_at: str = "?"
    entry_exchange_px: float = 0.0


class PositionSizer:
    """Position sizing using Kelly fraction and volatility scaling."""

    def __init__(
        self,
        max_position_size: float = Config.MAX_POSITION_SIZE,
        max_kelly_fraction: float = 0.25,
        target_volatility: float = 0.60,
        min_position_size: float = Config.MIN_POSITION_SIZE,
    ) -> None:
        self.max_position_size = max(float(max_position_size), 0.0)
        self.max_kelly_fraction = float(np.clip(max_kelly_fraction, 0.0, 1.0))
        self.target_volatility = max(float(target_volatility), 1e-6)
        self.min_position_size = max(float(min_position_size), 0.0)

    @staticmethod
    def _sanitize_probability(value: float, default: float = 0.5) -> float:
        try:
            clean = float(value)
            if np.isnan(clean) or np.isinf(clean):
                return default
            return float(np.clip(clean, 1e-4, 1 - 1e-4))
        except Exception:
            return default

    def calculate_kelly_fraction(self, win_probability: float, market_price: float) -> float:
        p = self._sanitize_probability(win_probability)
        q = self._sanitize_probability(market_price)

        b = (1.0 - q) / max(q, 1e-9)
        if b <= 0:
            return 0.0

        kelly_raw = (b * p - (1.0 - p)) / b
        if np.isnan(kelly_raw) or np.isinf(kelly_raw):
            return 0.0
        return float(np.clip(kelly_raw, 0.0, self.max_kelly_fraction))

    def adjust_position_for_volatility(self, base_size: float, volatility: float) -> float:
        base = max(float(base_size), 0.0)
        vol = float(volatility) if volatility is not None else np.nan
        if np.isnan(vol) or np.isinf(vol) or vol <= 0:
            vol = self.target_volatility

        scale = self.target_volatility / max(vol, 1e-6)
        scale = float(np.clip(scale, 0.25, 2.0))
        adjusted = base * scale
        return float(np.clip(adjusted, 0.0, self.max_position_size))

    def calculate_position_size(
        self,
        capital_usd: float,
        win_probability: float,
        market_price: float,
        volatility: float,
    ) -> float:
        capital = max(float(capital_usd), 0.0)
        if capital <= 0:
            return 0.0

        kelly_fraction = self.calculate_kelly_fraction(win_probability, market_price)
        if kelly_fraction <= 0:
            return 0.0

        base = capital * kelly_fraction
        adjusted = self.adjust_position_for_volatility(base, volatility)
        if adjusted < self.min_position_size:
            return 0.0

        return min(adjusted, capital, self.max_position_size)


class DrawdownManager:
    """Track drawdown and enforce global trading circuit breaker."""

    def __init__(self, initial_capital: float, max_drawdown_pct: float) -> None:
        self.initial_capital = max(float(initial_capital), 0.0)
        self.max_drawdown_pct = float(np.clip(max_drawdown_pct, 0.01, 0.95))
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital
        self.history = pd.DataFrame(columns=["timestamp", "equity", "peak_equity", "drawdown"])

    def update_drawdown(self, equity: float) -> Dict[str, float]:
        current = float(equity)
        if np.isnan(current) or np.isinf(current):
            current = self.current_equity

        self.current_equity = max(current, 0.0)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)

        row = pd.DataFrame(
            [{
                "timestamp": datetime.now(timezone.utc),
                "equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "drawdown": drawdown,
            }]
        )
        self.history = pd.concat([self.history, row], ignore_index=True)

        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": float(drawdown),
            "max_drawdown_limit": self.max_drawdown_pct,
            "breached": bool(drawdown >= self.max_drawdown_pct),
        }


class StopLossManager:
    """Stop-loss manager for binary position directions."""

    def __init__(self, stop_loss_pct: float) -> None:
        self.stop_loss_pct = float(np.clip(stop_loss_pct, 0.01, 0.95))

    def calculate_stop_loss_price(self, entry_price: float, direction: str) -> float:
        entry = float(np.clip(entry_price, 1e-4, 1 - 1e-4))
        side = direction.lower().strip()

        if "buy_yes" in side or side in {"long_yes", "yes"}:
            return float(np.clip(entry * (1.0 - self.stop_loss_pct), 1e-4, 1 - 1e-4))

        if "buy_no" in side or side in {"long_no", "no"}:
            return float(np.clip(entry * (1.0 + self.stop_loss_pct), 1e-4, 1 - 1e-4))

        return entry

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        price = float(current_price)
        if np.isnan(price) or np.isinf(price):
            return False

        side = position.direction.lower().strip()
        if "buy_yes" in side or side in {"long_yes", "yes"}:
            return price <= position.stop_loss_price
        if "buy_no" in side or side in {"long_no", "no"}:
            return price >= position.stop_loss_price
        return False


class RiskManager:
    """High-level risk orchestrator with strategy-specific controls."""

    def __init__(
        self,
        gas_fee_matic: float = Config.GAS_FEE_MATIC,
        exchange_taker_fee: float = Config.EXCHANGE_TAKER_FEE,
        polymarket_fee: float = Config.POLYMARKET_FEE,
        initial_capital: float = Config.INITIAL_CAPITAL,
        max_drawdown_pct: float = Config.MAX_DRAWDOWN_PCT,
        stop_loss_pct: float = Config.STOP_LOSS_PCT,
        max_position_size: float = Config.MAX_POSITION_SIZE,
    ) -> None:
        self.gas_fee_matic = max(float(gas_fee_matic), 0.0)
        self.exchange_taker_fee = max(float(exchange_taker_fee), 0.0)
        self.polymarket_fee = max(float(polymarket_fee), 0.0)

        self.position_sizer = PositionSizer(max_position_size=max_position_size)
        self.drawdown_manager = DrawdownManager(initial_capital=initial_capital, max_drawdown_pct=max_drawdown_pct)
        self.stop_loss_manager = StopLossManager(stop_loss_pct=stop_loss_pct)

    def estimate_total_fee_pct(self, position_size_usd: float, strategy_name: str = "delta_neutral", matic_price_usd: float = 1.0) -> float:
        """Estimate total fee percentage including gas, exchange fee and Polymarket fee."""
        size = max(float(position_size_usd), 1e-9)
        matic_price = max(float(matic_price_usd), 0.0)
        gas_cost_pct = (self.gas_fee_matic * matic_price) / size
        
        # Only add exchange (Binance) fee for delta-neutral
        venue_fees = self.polymarket_fee
        if strategy_name == "delta_neutral":
            venue_fees += self.exchange_taker_fee
            
        return float(venue_fees + gas_cost_pct)

    def calculate_net_spread(self, raw_spread: float, position_size_usd: float, strategy_name: str = "delta_neutral", matic_price_usd: float = 1.0) -> float:
        spread = float(raw_spread)
        total_fees = self.estimate_total_fee_pct(
            position_size_usd=position_size_usd, 
            strategy_name=strategy_name,
            matic_price_usd=matic_price_usd
        )
        net_spread = spread - total_fees
        if np.isnan(net_spread) or np.isinf(net_spread):
            return 0.0
        return float(net_spread)

    def check_liquidity_and_slippage(
        self,
        orderbook: Dict,
        target_size: float,
        side: str = "buy",
        max_slippage: float = 0.02,
    ) -> Tuple[float, bool]:
        """Estimate weighted average fill price and enforce slippage cap.

        Accepts two level formats:
        - list-of-lists : [[price, size], ...]        (CLOB style)
        - list-of-dicts : [{"price": p, "size": s}]  (Polymarket Gamma style)
        """
        if not isinstance(orderbook, dict) or target_size <= 0:
            return 0.0, False

        levels = orderbook.get("asks", []) if side.lower() == "buy" else orderbook.get("bids", [])
        if not levels:
            return 0.0, False

        def _extract(level) -> Tuple[float, float]:
            """Return (price, size) regardless of level format."""
            if isinstance(level, dict):
                return float(level.get("price", 0.0)), float(level.get("size", 0.0))
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                return float(level[0]), float(level[1])
            return 0.0, 0.0

        top_price, _ = _extract(levels[0])
        if top_price <= 0:
            return 0.0, False

        remaining = float(target_size)
        total_cost = 0.0
        total_available = 0.0

        for level in levels:
            price, size = _extract(level)
            if price <= 0 or size <= 0 or np.isnan(price) or np.isnan(size):
                continue
            total_available += size
            fill = min(remaining, size)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 1e-12:
                break

        if total_available < Config.MIN_POLY_LIQUIDITY_USD:
            return 0.0, False

        if remaining > 1e-12:
            return 0.0, False

        avg_price = total_cost / max(target_size, 1e-9)
        slippage = abs(avg_price - top_price) / max(top_price, 1e-9)
        return float(avg_price), bool(slippage <= max_slippage)

    def reject_trade_for_excess_slippage(self, expected_edge: float, estimated_slippage: float) -> bool:
        """Reject trade when slippage is too high relative to expected edge."""
        edge = abs(float(expected_edge))
        slip = abs(float(estimated_slippage))
        if edge <= 1e-9:
            return True
        return slip > (Config.MAX_SLIPPAGE_TO_EDGE_RATIO * edge)

    def validate_strategy_risk(
        self,
        strategy_name: str,
        net_spread: float,
        funding_rate: float,
        current_delta_exposure: float,
        liquidity_ok: bool,
    ) -> Tuple[bool, str]:
        """Strategy-specific pre-trade risk checks."""
        if not liquidity_ok:
            return False, "liquidity_check_failed"

        if strategy_name == "delta_neutral":
            if abs(funding_rate) > Config.MAX_FUNDING_RATE_ABS:
                return False, "funding_rate_limit"
            if abs(current_delta_exposure) > (1.0 + Config.DELTA_REBALANCE_THRESHOLD):
                return False, "delta_exposure_limit"
            if net_spread < Config.DELTA_NEUTRAL_MIN_EDGE:
                return False, "edge_below_delta_neutral_threshold"

        if strategy_name == "pure_polymarket" and net_spread < Config.PURE_POLY_MIN_EDGE:
            return False, "edge_below_pure_polymarket_threshold"

        return True, "ok"

    def compute_strategy_position_size(
        self,
        strategy_name: str,
        capital_usd: float,
        theoretical_prob: float,
        market_price: float,
        volatility: float,
        net_spread: float,
    ) -> float:
        """Compute strategy-aware trade size with concentration caps."""
        if net_spread <= 0:
            return 0.0

        base = self.position_sizer.calculate_position_size(
            capital_usd=capital_usd,
            win_probability=theoretical_prob,
            market_price=market_price,
            volatility=volatility,
        )
        if base <= 0:
            return 0.0

        concentration_cap = float(capital_usd) * Config.MAX_POSITION_PER_ASSET_PCT
        size = min(base, concentration_cap, Config.MAX_POSITION_SIZE)

        if strategy_name == "pure_polymarket":
            size *= 0.9
        return float(max(size, 0.0))

    def generate_position(
        self,
        position_id: str,
        asset: str,
        direction: str,
        strategy: str,
        size_usd: float,
        entry_price: float,
        target_delta: float = 0.0,
        ends_at: str = "?",
        entry_exchange_px: float = 0.0,
    ) -> Position:
        """Create a position with computed stop-loss trigger."""
        stop_loss_price = self.stop_loss_manager.calculate_stop_loss_price(entry_price, direction)
        return Position(
            position_id=position_id,
            asset=asset,
            direction=direction,
            strategy=strategy,
            size_usd=max(float(size_usd), 0.0),
            entry_price=float(np.clip(entry_price, 1e-4, 1 - 1e-4)),
            stop_loss_price=stop_loss_price,
            opened_at=datetime.now(timezone.utc),
            target_delta=float(target_delta),
            ends_at=ends_at,
            entry_exchange_px=entry_exchange_px,
        )

    def monitor_delta_neutrality(self, positions: Dict[str, Position]) -> float:
        """Compute aggregate target delta for delta-neutral strategy."""
        delta = 0.0
        for pos in positions.values():
            if pos.strategy == "delta_neutral":
                delta += float(pos.target_delta)
        return float(delta)

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop-loss is hit for a position."""
        return self.stop_loss_manager.check_stop_loss(position, current_price)

    def update_drawdown(self, equity: float) -> Dict[str, float]:
        """Update drawdown state and log breach events."""
        status = self.drawdown_manager.update_drawdown(equity)
        if status["breached"]:
            logger.warning(
                "Drawdown breached %.2f%% >= %.2f%%",
                status["drawdown"] * 100,
                status["max_drawdown_limit"] * 100,
            )
        return status

    def compute_trade_size(
        self,
        capital_usd: float,
        theoretical_prob: float,
        market_price: float,
        volatility: float,
        net_spread: float,
    ) -> float:
        """Backward-compatible generic size wrapper."""
        return self.compute_strategy_position_size(
            strategy_name="pure_polymarket",
            capital_usd=capital_usd,
            theoretical_prob=theoretical_prob,
            market_price=market_price,
            volatility=volatility,
            net_spread=net_spread,
        )
