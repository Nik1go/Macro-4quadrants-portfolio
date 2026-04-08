"""SQLite persistence layer for arbitrage runtime, metrics and backtests."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


class StorageManager:
    """Manage SQLite persistence for market data, trades, states and metrics."""

    def __init__(self, db_path: str = "data/arbitrage.db") -> None:
        self.db_path = db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = self._table_columns(conn, table_name)
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def init_db(self) -> None:
        """Initialize all required tables with forward-compatible schema."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spreads (
                timestamp TEXT,
                asset_pair TEXT,
                spot_price REAL,
                implied_vol REAL,
                polymarket_price REAL,
                theoretical_prob REAL,
                rfr REAL,
                net_spread REAL,
                is_opportunity INTEGER,
                strategy TEXT,
                slug TEXT,
                signal_type TEXT
            )
            """
        )

        # Forward-compatible migrations
        self._ensure_column(conn, "spreads", "signal_type", "TEXT")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TEXT,
                asset_pair TEXT,
                side TEXT,
                size REAL,
                poly_price REAL,
                exchange_price REAL,
                expected_profit REAL,
                status TEXT,
                trade_type TEXT,
                strategy TEXT,
                metadata_json TEXT,
                exit_price REAL,
                exit_timestamp TEXT,
                realized_pnl REAL,
                fees_paid REAL
            )
            """
        )

        # Forward-compatible migration for older trades schema.
        self._ensure_column(conn, "trades", "metadata_json", "TEXT")
        self._ensure_column(conn, "trades", "exit_price", "REAL")
        self._ensure_column(conn, "trades", "exit_timestamp", "TEXT")
        self._ensure_column(conn, "trades", "realized_pnl", "REAL")
        self._ensure_column(conn, "trades", "fees_paid", "REAL")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                timestamp TEXT,
                strategy TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                strategy TEXT,
                params_json TEXT,
                summary_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_points (
                run_id TEXT,
                timestamp TEXT,
                equity REAL,
                pnl REAL,
                drawdown REAL,
                trade_executed INTEGER,
                size REAL,
                entry_price REAL,
                exit_price REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_state (
                state_key TEXT PRIMARY KEY,
                state_value_json TEXT,
                updated_at TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS config_snapshots (
                timestamp TEXT,
                strategy TEXT,
                config_json TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    def save_spread(self, data: Dict[str, Any]) -> None:
        """Persist spread snapshot with sanitization."""
        payload = dict(data)
        payload["timestamp"] = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        pd.DataFrame([payload]).to_sql("spreads", conn, if_exists="append", index=False)
        conn.close()

    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Persist executed trade row."""
        payload = dict(trade_data)
        payload["timestamp"] = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

        if "metadata_json" not in payload:
            payload["metadata_json"] = json.dumps(payload.pop("metadata", {}), ensure_ascii=False)

        payload.setdefault("exit_price", None)
        payload.setdefault("exit_timestamp", None)
        payload.setdefault("realized_pnl", 0.0)
        payload.setdefault("fees_paid", 0.0)

        conn = self._connect()
        pd.DataFrame([payload]).to_sql("trades", conn, if_exists="append", index=False)
        conn.close()

    def update_trade_settlement(self, slug: str, exit_price: float, realized_pnl: float) -> None:
        """Update an open trade with its final settlement price and pnl."""
        conn = self._connect()
        cursor = conn.cursor()
        now_ts = datetime.now(timezone.utc).isoformat()
        
        # On cherche le trade le plus récent pour ce slug qui n'est pas encore clôturé
        cursor.execute(
            """
            UPDATE trades 
            SET exit_price = ?, exit_timestamp = ?, realized_pnl = ?, status = 'SETTLED'
            WHERE rowid = (
                SELECT rowid FROM trades 
                WHERE json_extract(metadata_json, '$.slug') = ? 
                AND (exit_timestamp IS NULL OR exit_timestamp = '')
                ORDER BY timestamp DESC LIMIT 1
            )
            """,
            (exit_price, now_ts, realized_pnl, slug)
        )
        conn.commit()
        conn.close()

    def save_strategy_metric(
        self,
        strategy: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a strategy metric point."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        }

        conn = self._connect()
        pd.DataFrame([row]).to_sql("strategy_metrics", conn, if_exists="append", index=False)
        conn.close()

    def save_backtest_run(
        self,
        run_id: str,
        strategy: str,
        params: Dict[str, Any],
        summary: Dict[str, Any],
        points: pd.DataFrame,
    ) -> None:
        """Persist a full backtest run (header + timeseries)."""
        run_row = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "params_json": json.dumps(params, ensure_ascii=False),
            "summary_json": json.dumps(summary, ensure_ascii=False),
        }

        conn = self._connect()
        pd.DataFrame([run_row]).to_sql("backtest_runs", conn, if_exists="append", index=False)

        if not points.empty:
            clean_points = points.copy()
            clean_points["run_id"] = run_id
            clean_points.to_sql("backtest_points", conn, if_exists="append", index=False)

        conn.close()

    def save_bot_state(self, state_key: str, state_value: Dict[str, Any]) -> None:
        """Upsert bot state for crash-safe restart behavior."""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO bot_state(state_key, state_value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET state_value_json=excluded.state_value_json, updated_at=excluded.updated_at
            """,
            (
                state_key,
                json.dumps(state_value, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def load_bot_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """Load bot state payload if available."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT state_value_json FROM bot_state WHERE state_key = ?", (state_key,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def save_config_snapshot(self, strategy: str, config_data: Dict[str, Any]) -> None:
        """Persist runtime config snapshot for observability."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "config_json": json.dumps(config_data, ensure_ascii=False),
        }
        conn = self._connect()
        pd.DataFrame([row]).to_sql("config_snapshots", conn, if_exists="append", index=False)
        conn.close()

    def get_recent_spreads(self, limit: int = 1000, asset_pair: Optional[str] = None) -> pd.DataFrame:
        """Retrieve recent spread history."""
        conn = self._connect()
        query = "SELECT * FROM spreads"
        params: List[Any] = []
        if asset_pair:
            query += " WHERE asset_pair = ?"
            params.append(asset_pair)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent trades for monitoring UI."""
        conn = self._connect()
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", conn, params=[int(limit)])
        conn.close()
        return df

    def get_open_positions(self) -> pd.DataFrame:
        """Return inferred open positions from trades table.

        Positions are inferred as rows where `status` is FILLED and
        `exit_timestamp` is null or empty.
        """
        conn = self._connect()
        query = """
            SELECT
                timestamp,
                asset_pair,
                side,
                size,
                poly_price AS entry_price,
                exchange_price,
                strategy,
                expected_profit,
                realized_pnl,
                fees_paid,
                exit_price,
                exit_timestamp
            FROM trades
            WHERE upper(status) = 'FILLED'
              AND (exit_timestamp IS NULL OR trim(exit_timestamp) = '')
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return df

        df["size_usd"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.5)
        df["current_price"] = df["entry_price"]
        df["unrealized_pnl"] = 0.0
        return df

    def get_trades_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return complete trade history including realized fields."""
        conn = self._connect()
        query = """
            SELECT
                timestamp,
                asset_pair,
                side,
                size,
                poly_price AS entry_price,
                exit_price,
                exit_timestamp,
                strategy,
                expected_profit,
                realized_pnl,
                fees_paid,
                status,
                trade_type,
                metadata_json
            FROM trades
            ORDER BY timestamp DESC
        """
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.5)
            df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
            df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
            df["fees_paid"] = pd.to_numeric(df["fees_paid"], errors="coerce").fillna(0.0)

        return df
