"""Execution engine for multi-strategy Polymarket arbitrage.

Supports:
- Paper execution mode (deterministic fills)
- Live execution interfaces for Polymarket (Web3) and Binance (ccxt)
- Exponential retry with jitter for network-sensitive operations
- Durable persistence to CSV + SQLite
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

import ccxt.async_support as ccxt

try:
    from .output_manager import OutputManager
    from .storage import StorageManager
    from .utils.config import Config
    from .utils.logger import Logger
except ImportError:  # pragma: no cover
    from output_manager import OutputManager
    from storage import StorageManager
    from utils.config import Config
    from utils.logger import Logger

logger = Logger().logger


@dataclass
class ExecutionResult:
    """Normalized execution result payload."""

    status: str
    mode: str
    reason: str
    executed_spread: float
    realized_pnl: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "status": self.status,
            "mode": self.mode,
            "reason": self.reason,
            "executed_spread": self.executed_spread,
            "realized_pnl": self.realized_pnl,
            "timestamp": self.timestamp,
        }


class PolymarketExecutor:
    """Execution adapter for Polymarket.

    Current implementation provides the interface and validates payloads.
    Actual on-chain order placement can be plugged in safely.
    """

    async def place_order(self, side: str, price: float, size: float, slug: str) -> Dict[str, Any]:
        """Place a Polymarket order (stub for now)."""
        await asyncio.sleep(0.05)
        if size <= 0 or price <= 0:
            return {"status": "REJECTED", "reason": "invalid_order_params"}

        return {
            "status": "FILLED",
            "venue": "polymarket",
            "side": side,
            "price": price,
            "size": size,
            "slug": slug,
        }


class BinanceExecutor:
    """Execution adapter for Binance perpetual hedge leg."""

    def __init__(self) -> None:
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "apiKey": Config.BINANCE_API_KEY,
                "secret": Config.BINANCE_SECRET,
                "options": {"defaultType": Config.BINANCE_DEFAULT_TYPE},
            }
        )

    async def place_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """Place market order on Binance or simulate if credentials missing."""
        if amount <= 0:
            return {"status": "REJECTED", "reason": "invalid_amount"}

        if not Config.BINANCE_API_KEY or not Config.BINANCE_SECRET:
            await asyncio.sleep(0.05)
            return {
                "status": "FILLED",
                "venue": "binance",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "simulated": True,
            }

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
            )
            return {
                "status": "FILLED",
                "venue": "binance",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "order_id": order.get("id"),
                "simulated": False,
            }
        except Exception as exc:
            logger.error("Binance order failed: %s", exc)
            return {"status": "FAILED", "reason": str(exc), "venue": "binance"}

    async def close(self) -> None:
        """Close underlying ccxt client."""
        await self.exchange.close()


class ExecutionEngine:
    """Orchestrate execution and persistence for strategy signals."""

    def __init__(
        self,
        is_paper_trade: bool = Config.PAPER_TRADE,
        output_manager: Optional[OutputManager] = None,
        storage_manager: Optional[StorageManager] = None,
        execution_retry_attempts: int = Config.RETRY_ATTEMPTS,
    ) -> None:
        self.is_paper_trade = bool(is_paper_trade)
        self.output_manager = output_manager or OutputManager()
        self.storage_manager = storage_manager or StorageManager(db_path=Config.DB_PATH)
        self.execution_retry_attempts = max(int(execution_retry_attempts), 1)

        self.output_manager.initialize_output_files()

        self.polymarket_executor = PolymarketExecutor()
        self.binance_executor = BinanceExecutor()

    async def _with_exponential_retry(
        self,
        op: Callable[[], Awaitable[Dict[str, Any]]],
        label: str,
    ) -> Dict[str, Any]:
        """Retry asynchronous operation with exponential backoff and jitter."""
        last_error: Optional[str] = None

        for attempt in range(1, self.execution_retry_attempts + 1):
            try:
                result = await op()
                status = str(result.get("status", "")).upper()
                if status in {"FILLED", "REJECTED"}:
                    return result

                last_error = str(result.get("reason", "unknown_error"))
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Execution op '%s' failed attempt %d/%d: %s",
                    label,
                    attempt,
                    self.execution_retry_attempts,
                    exc,
                )

                if attempt < self.execution_retry_attempts:
                    base_delay = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    jitter = random.uniform(0.0, Config.RETRY_JITTER)
                    delay = min(base_delay + jitter, Config.RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)

        return {
            "status": "FAILED",
            "reason": f"{label}_failed_after_retries:{last_error}",
            "mode": "paper" if self.is_paper_trade else "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def execute_arbitrage(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute signal and persist trade logs on success."""
        result = await self._execute_with_retry(signal_data)
        if result.get("status") == "FILLED":
            self._persist_trade(signal_data=signal_data, execution_result=result)
            self.output_manager.update_metrics()
        return result

    async def _execute_with_retry(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retry wrapper for full execution."""
        if self.is_paper_trade:
            return await self._with_exponential_retry(
                lambda: self._execute_paper_trade(signal_data),
                label="paper_execution",
            )

        return await self._with_exponential_retry(
            lambda: self._execute_live_trade(signal_data),
            label="live_execution",
        )

    async def _execute_paper_trade(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate multi-leg execution in paper mode."""
        await asyncio.sleep(0.08)

        spread = float(signal_data.get("net_spread", signal_data.get("spread", 0.0)))
        expected_profit = float(signal_data.get("expected_profit", 0.0))
        strategy = str(signal_data.get("strategy", "unknown"))

        return ExecutionResult(
            status="FILLED",
            mode="paper",
            reason=f"paper_fill_{strategy}",
            executed_spread=spread,
            realized_pnl=expected_profit,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()

    async def _execute_live_trade(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live trades depending on strategy and direction."""
        direction = str(signal_data.get("direction", "")).lower()
        size = float(signal_data.get("size", 0.0))
        poly_price = float(signal_data.get("polymarket_price", signal_data.get("entry_price", 0.5)))
        slug = str(signal_data.get("slug", ""))

        if size <= 0:
            return ExecutionResult(
                status="REJECTED",
                mode="live",
                reason="non_positive_size",
                executed_spread=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict()

        if poly_price <= 0 or poly_price >= 1:
            return ExecutionResult(
                status="REJECTED",
                mode="live",
                reason="invalid_polymarket_price",
                executed_spread=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict()

        if "buy_no" in direction:
            poly_side = "buy_no"
        else:
            poly_side = "buy_yes"

        leg_a = await self._with_exponential_retry(
            lambda: self.polymarket_executor.place_order(
                side=poly_side,
                price=poly_price,
                size=size,
                slug=slug,
            ),
            label="polymarket_leg",
        )
        if leg_a.get("status") != "FILLED":
            return ExecutionResult(
                status="FAILED",
                mode="live",
                reason="polymarket_leg_failed",
                executed_spread=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict()

        if "hedge" in direction:
            hedge_side = "sell" if "short_binance" in direction else "buy"
            symbol = str(signal_data.get("symbol", "BTC/USDT"))
            approx_amount = max(size / max(float(signal_data.get("spot_price", 1.0)), 1e-9), 1e-6)

            leg_b = await self._with_exponential_retry(
                lambda: self.binance_executor.place_order(
                    symbol=symbol,
                    side=hedge_side,
                    amount=approx_amount,
                ),
                label="binance_leg",
            )
            if leg_b.get("status") != "FILLED":
                return ExecutionResult(
                    status="FAILED",
                    mode="live",
                    reason="binance_leg_failed",
                    executed_spread=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ).to_dict()

        spread = float(signal_data.get("net_spread", signal_data.get("spread", 0.0)))
        expected_profit = float(signal_data.get("expected_profit", 0.0))

        return ExecutionResult(
            status="FILLED",
            mode="live",
            reason="live_multi_leg_filled",
            executed_spread=spread,
            realized_pnl=expected_profit,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()

    def _persist_trade(self, signal_data: Dict[str, Any], execution_result: Dict[str, Any]) -> None:
        """Persist normalized trade in CSV and SQLite."""
        trade_payload = {
            "timestamp": execution_result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "asset": signal_data.get("asset", "UNKNOWN"),
            "strike": signal_data.get("strike", 0.0),
            "direction": signal_data.get("direction", "hold"),
            "size": signal_data.get("size", 0.0),
            "entry_price": signal_data.get("polymarket_price", signal_data.get("entry_price", 0.5)),
            "theoretical_prob": signal_data.get("theoretical_prob", 0.5),
            "polymarket_price": signal_data.get("polymarket_price", 0.5),
            "spread": signal_data.get("net_spread", signal_data.get("spread", 0.0)),
            "pnl": execution_result.get("realized_pnl", 0.0),
        }

        self.output_manager.append_trade(trade_payload)

        self.storage_manager.save_trade(
            {
                "timestamp": trade_payload["timestamp"],
                "asset_pair": signal_data.get("symbol", "UNKNOWN"),
                "side": trade_payload["direction"],
                "size": trade_payload["size"],
                "poly_price": trade_payload["polymarket_price"],
                "exchange_price": signal_data.get("spot_price", 0.0),
                "expected_profit": signal_data.get("expected_profit", 0.0),
                "status": execution_result.get("status", "UNKNOWN"),
                "trade_type": "PAPER" if self.is_paper_trade else "LIVE",
                "strategy": signal_data.get("strategy", "unknown"),
                "realized_pnl": execution_result.get("realized_pnl", 0.0),
                "fees_paid": signal_data.get("fees_paid", 0.0),
                "metadata": {
                    "slug": signal_data.get("slug", ""),
                    "signal_reason": signal_data.get("signal_reason", ""),
                    "signal_meta": signal_data.get("signal_meta", {}),
                },
            }
        )

    async def finalize(self) -> Dict[str, Any]:
        """Finalize resources and refresh metrics."""
        await self.binance_executor.close()
        metrics = self.output_manager.update_metrics()
        return {"metrics": metrics}
