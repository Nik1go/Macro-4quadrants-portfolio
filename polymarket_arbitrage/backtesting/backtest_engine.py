"""Backtesting engine for Polymarket arbitrage strategies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from ..risk_management import RiskManager
    from ..storage import StorageManager
    from ..strategies import DeltaNeutralStrategy, PurePolymarketStrategy, StrategyContext
    from ..utils.config import Config
except ImportError:  # pragma: no cover
    from risk_management import RiskManager
    from storage import StorageManager
    from strategies import DeltaNeutralStrategy, PurePolymarketStrategy, StrategyContext
    from utils.config import Config


@dataclass
class SimPosition:
    """Simple simulated position model for backtesting."""

    slug: str
    strategy: str
    direction: str
    size: float
    entry_price: float
    entry_ts: pd.Timestamp


class BacktestEngine:
    """Run deterministic backtests over historical spread snapshots."""

    def __init__(
        self,
        initial_capital: float = Config.BACKTEST_INITIAL_CAPITAL,
        storage_manager: Optional[StorageManager] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.storage = storage_manager or StorageManager(db_path=Config.DB_PATH)
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_drawdown_pct=Config.BACKTEST_MAX_DRAWDOWN,
            stop_loss_pct=Config.BACKTEST_STOP_LOSS,
            max_position_size=Config.MAX_POSITION_SIZE,
        )

        self.positions: Dict[str, SimPosition] = {}
        self.results: List[Dict[str, Any]] = []

    def _select_strategy(self, strategy_name: str):
        if strategy_name == "delta_neutral":
            return DeltaNeutralStrategy()
        return PurePolymarketStrategy()

    def _record_step(
        self,
        timestamp: pd.Timestamp,
        equity: float,
        pnl: float,
        drawdown: float,
        trade_executed: bool,
        size: float,
        entry_price: float,
        exit_price: float,
    ) -> None:
        self.results.append(
            {
                "timestamp": timestamp,
                "equity": float(equity),
                "pnl": float(pnl),
                "drawdown": float(drawdown),
                "trade_executed": bool(trade_executed),
                "size": float(size),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
            }
        )

    def _mark_to_market(self, current_prices: Dict[str, float]) -> float:
        unrealized = 0.0
        for slug, pos in self.positions.items():
            current = float(current_prices.get(slug, pos.entry_price))
            if "buy_yes" in pos.direction:
                unrealized += (current - pos.entry_price) * pos.size
            elif "buy_no" in pos.direction:
                unrealized += (pos.entry_price - current) * pos.size
        return float(unrealized)

    def _close_position(self, slug: str, current_price: float) -> float:
        pos = self.positions.pop(slug)
        if "buy_yes" in pos.direction:
            pnl = (current_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - current_price) * pos.size

        self.capital += pos.size + pnl
        return float(pnl)

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        min_edge: float,
        max_position_size: float,
    ) -> pd.DataFrame:
        """Execute a backtest over historical records."""
        if data.empty:
            return pd.DataFrame()

        df = data.copy().sort_values("timestamp").reset_index(drop=True)
        min_edge = float(max(min_edge, 0.0))
        max_position_size = float(max(max_position_size, 0.001))

        strategy = self._select_strategy(strategy_name)
        grouped = df.groupby("timestamp", sort=True)

        for ts, slice_df in grouped:
            current_prices = {
                str(row["slug"]): float(row["polymarket_price"])
                for _, row in slice_df.iterrows()
            }

            step_pnl = 0.0
            trade_executed = False
            record_size = 0.0
            entry_price = 0.0
            exit_price = 0.0

            to_close: List[str] = []
            for slug, position in self.positions.items():
                current_price = current_prices.get(slug, position.entry_price)
                simulated_position = self.risk_manager.generate_position(
                    position_id=slug,
                    asset="N/A",
                    direction=position.direction,
                    strategy=position.strategy,
                    size_usd=position.size,
                    entry_price=position.entry_price,
                )
                if self.risk_manager.check_stop_loss(simulated_position, current_price):
                    to_close.append(slug)

            for slug in to_close:
                current_price = float(current_prices.get(slug, self.positions[slug].entry_price))
                pnl = self._close_position(slug, current_price)
                step_pnl += pnl
                trade_executed = True
                exit_price = current_price

            drawdown_state = self.risk_manager.update_drawdown(self.capital)
            if drawdown_state["breached"]:
                equity = self.capital + self._mark_to_market(current_prices)
                self._record_step(ts, equity, step_pnl, drawdown_state["drawdown"], trade_executed, 0.0, 0.0, 0.0)
                continue

            for _, row in slice_df.iterrows():
                slug = str(row["slug"])
                if slug in self.positions:
                    continue

                context = StrategyContext(
                    asset="N/A",
                    symbol=str(row.get("asset_pair", "UNKNOWN")),
                    slug=slug,
                    title="",
                    strike=0.0,
                    spot_price=float(row["spot_price"]),
                    theoretical_prob=float(row["theoretical_prob"]),
                    polymarket_price=float(row["polymarket_price"]),
                    time_to_maturity=0.1,
                    implied_volatility=float(row["implied_vol"]),
                    net_spread=float(row["net_spread"]),
                    funding_rate=0.0,
                    orderbook_depth_ok=True,
                )

                signal = strategy.generate_signal(context)
                if not signal.should_trade:
                    continue

                if abs(float(row["net_spread"])) < min_edge:
                    continue

                allowed_position = self.capital * max_position_size
                size = self.risk_manager.compute_strategy_position_size(
                    strategy_name=strategy_name,
                    capital_usd=self.capital,
                    theoretical_prob=float(row["theoretical_prob"]),
                    market_price=float(row["polymarket_price"]),
                    volatility=float(row["implied_vol"]),
                    net_spread=float(row["net_spread"]),
                )
                size = min(size, allowed_position)
                if size <= 0:
                    continue

                entry = float(row["polymarket_price"])
                self.positions[slug] = SimPosition(
                    slug=slug,
                    strategy=strategy_name,
                    direction=signal.direction,
                    size=size,
                    entry_price=entry,
                    entry_ts=ts,
                )
                self.capital -= size
                trade_executed = True
                record_size = size
                entry_price = entry
                break

            equity = self.capital + self._mark_to_market(current_prices)
            drawdown = self.risk_manager.update_drawdown(equity)["drawdown"]
            self._record_step(ts, equity, step_pnl, drawdown, trade_executed, record_size, entry_price, exit_price)

        if self.positions:
            final_prices = {
                str(row["slug"]): float(row["polymarket_price"])
                for _, row in df.groupby("slug", as_index=False).tail(1).iterrows()
            }
            final_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
            final_pnl = 0.0
            for slug in list(self.positions.keys()):
                final_pnl += self._close_position(slug, float(final_prices.get(slug, self.positions[slug].entry_price)))
            equity = self.capital
            drawdown = self.risk_manager.update_drawdown(equity)["drawdown"]
            self._record_step(final_ts, equity, final_pnl, drawdown, True, 0.0, 0.0, 0.0)

        result_df = pd.DataFrame(self.results)
        return result_df


def run_backtest(
    data: pd.DataFrame,
    strategy_name: str,
    min_edge: float,
    max_position_size: float,
) -> pd.DataFrame:
    """Convenience wrapper for Streamlit integration.

    Persists result CSV/JSON under `backtests/` and inserts run metadata in SQLite.
    """
    engine = BacktestEngine(initial_capital=Config.BACKTEST_INITIAL_CAPITAL)
    result = engine.run_backtest(
        data=data,
        strategy_name=strategy_name,
        min_edge=min_edge,
        max_position_size=max_position_size,
    )

    run_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    backtests_dir = Path(Config.BACKTEST_DIR)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    csv_path = backtests_dir / f"{run_id}.csv"
    json_path = backtests_dir / f"{run_id}.json"

    if not result.empty:
        result.to_csv(csv_path, index=False)

    summary = {
        "run_id": run_id,
        "strategy": strategy_name,
        "rows": int(len(result)),
        "initial_capital": Config.BACKTEST_INITIAL_CAPITAL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    engine.storage.save_backtest_run(
        run_id=run_id,
        strategy=strategy_name,
        params={
            "min_edge": min_edge,
            "max_position_size": max_position_size,
        },
        summary=summary,
        points=result,
    )

    return result
