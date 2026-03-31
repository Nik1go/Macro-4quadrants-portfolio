from polymarket_arbitrage.storage import StorageManager
"""Unit tests for output manager and execution engine."""

import asyncio

from polymarket_arbitrage.execution_engine import ExecutionEngine
from polymarket_arbitrage.output_manager import OutputManager


def test_output_manager_initialize_and_append(tmp_path) -> None:
    trades_path = tmp_path / "trades.csv"
    metrics_path = tmp_path / "metrics.json"

    manager = OutputManager(str(trades_path), str(metrics_path))
    manager.initialize_output_files()

    ok = manager.append_trade(
        {
            "asset": "Bitcoin",
            "strike": 100000,
            "direction": "buy_yes",
            "size": 10,
            "entry_price": 0.5,
            "theoretical_prob": 0.6,
            "polymarket_price": 0.5,
            "spread": 0.1,
            "pnl": 1.0,
        }
    )
    assert ok is True

    metrics = manager.update_metrics()
    assert metrics["num_trades"] == 1


def test_execution_engine_paper_trade(tmp_path) -> None:
    output = OutputManager(str(tmp_path / "trades.csv"), str(tmp_path / "metrics.json"))
    storage = StorageManager(db_path=str(tmp_path / "test.db"))
    engine = ExecutionEngine(is_paper_trade=True, output_manager=output, storage_manager=storage)

    async def _run() -> None:
        result = await engine.execute_arbitrage(
            {
                "asset": "Bitcoin",
                "strike": 100000,
                "direction": "buy_yes",
                "size": 10,
                "polymarket_price": 0.5,
                "theoretical_prob": 0.6,
                "spread": 0.1,
                "net_spread": 0.08,
                "expected_profit": 0.8,
            }
        )
        assert result["status"] == "FILLED"
        summary = await engine.finalize()
        assert "metrics" in summary

    asyncio.run(_run())
