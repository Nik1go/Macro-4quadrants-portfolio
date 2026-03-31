"""Output manager for trade and performance artifacts.

This module centralizes write access to:
- ``trades.csv`` (append-only trade ledger)
- ``metrics.json`` (aggregated strategy metrics)

It includes robust handling of missing files, malformed data and fallback metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .utils.config import Config
    from .utils.logger import Logger
except ImportError:  # pragma: no cover
    from utils.config import Config
    from utils.logger import Logger

logger = Logger().logger


class OutputManager:
    """Orchestrate output files for trades and metrics."""

    TRADE_COLUMNS: List[str] = [
        "timestamp",
        "asset",
        "strike",
        "direction",
        "size",
        "entry_price",
        "theoretical_prob",
        "polymarket_price",
        "spread",
        "pnl",
    ]

    def __init__(
        self,
        trades_csv_path: str = Config.TRADES_CSV_PATH,
        metrics_json_path: str = Config.METRICS_JSON_PATH,
    ) -> None:
        self.trades_csv_path = trades_csv_path
        self.metrics_json_path = metrics_json_path
        self._last_valid_metrics: Dict[str, Any] = {
            "sharpe_ratio": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def initialize_output_files(self) -> None:
        """Create or repair output files with valid initial structure."""
        self._ensure_parent_directory(self.trades_csv_path)
        self._ensure_parent_directory(self.metrics_json_path)

        if not os.path.exists(self.trades_csv_path):
            pd.DataFrame(columns=self.TRADE_COLUMNS).to_csv(self.trades_csv_path, index=False)
            logger.info("Initialized trades.csv at %s", self.trades_csv_path)
        else:
            self._repair_trades_csv_if_needed()

        if not os.path.exists(self.metrics_json_path):
            self._write_metrics(self._last_valid_metrics)
            logger.info("Initialized metrics.json at %s", self.metrics_json_path)
        else:
            loaded = self._load_metrics()
            if loaded:
                self._last_valid_metrics = loaded

    def append_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Append a trade row to ``trades.csv``."""
        self.initialize_output_files()

        normalized = self._normalize_trade(trade_data)
        if normalized is None:
            return False

        try:
            new_row = pd.DataFrame([normalized], columns=self.TRADE_COLUMNS)
            new_row.to_csv(self.trades_csv_path, mode="a", index=False, header=False)
            return True
        except Exception as exc:
            logger.error("Failed to append trade to CSV: %s", exc)
            return False

    def update_metrics(self) -> Dict[str, Any]:
        """Recompute and persist strategy metrics from current trade ledger."""
        self.initialize_output_files()

        try:
            df = pd.read_csv(self.trades_csv_path)
            if df.empty:
                metrics = {
                    "sharpe_ratio": 0.0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "num_trades": 0,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
                self._write_metrics(metrics)
                self._last_valid_metrics = metrics
                return metrics

            df = self._sanitize_trade_dataframe(df)

            pnl_series = df["pnl"].astype(float)
            total_pnl = float(pnl_series.sum())
            num_trades = int(len(df))
            win_rate = float((pnl_series > 0).mean()) if num_trades > 0 else 0.0

            returns = pnl_series.copy()
            returns_std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
            sharpe = float(returns.mean() / returns_std * np.sqrt(len(returns))) if returns_std > 1e-12 else 0.0

            equity_curve = returns.cumsum()
            running_peak = equity_curve.cummax()
            drawdowns = running_peak - equity_curve
            max_drawdown = float(drawdowns.max()) if not drawdowns.empty else 0.0

            metrics = {
                "sharpe_ratio": round(sharpe, 6),
                "total_pnl": round(total_pnl, 6),
                "win_rate": round(win_rate, 6),
                "max_drawdown": round(max_drawdown, 6),
                "num_trades": num_trades,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            self._write_metrics(metrics)
            self._last_valid_metrics = metrics
            return metrics

        except Exception as exc:
            logger.error("Failed to compute metrics; reusing last valid metrics: %s", exc)
            fallback = dict(self._last_valid_metrics)
            fallback["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._write_metrics(fallback)
            return fallback

    def _normalize_trade(self, trade_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize and validate trade payload for persistence."""
        if not isinstance(trade_data, dict):
            logger.warning("Trade data must be a dict. Received: %s", type(trade_data))
            return None

        timestamp = trade_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        try:
            normalized_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).isoformat()
        except Exception:
            normalized_timestamp = datetime.now(timezone.utc).isoformat()

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                f_val = float(value)
                if np.isnan(f_val) or np.isinf(f_val):
                    return default
                return f_val
            except (TypeError, ValueError):
                return default

        normalized = {
            "timestamp": normalized_timestamp,
            "asset": str(trade_data.get("asset", "UNKNOWN")),
            "strike": _to_float(trade_data.get("strike", 0.0), 0.0),
            "direction": str(trade_data.get("direction", "buy_yes")),
            "size": _to_float(trade_data.get("size", 0.0), 0.0),
            "entry_price": float(np.clip(_to_float(trade_data.get("entry_price", 0.5), 0.5), 1e-4, 1 - 1e-4)),
            "theoretical_prob": float(
                np.clip(_to_float(trade_data.get("theoretical_prob", 0.5), 0.5), 1e-4, 1 - 1e-4)
            ),
            "polymarket_price": float(
                np.clip(_to_float(trade_data.get("polymarket_price", 0.5), 0.5), 1e-4, 1 - 1e-4)
            ),
            "spread": _to_float(trade_data.get("spread", 0.0), 0.0),
            "pnl": _to_float(trade_data.get("pnl", 0.0), 0.0),
        }

        if normalized["size"] <= 0:
            logger.warning("Ignoring trade with non-positive size: %s", normalized)
            return None

        return normalized

    def _sanitize_trade_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and are clean numeric values."""
        clean_df = df.copy()

        for column in self.TRADE_COLUMNS:
            if column not in clean_df.columns:
                clean_df[column] = np.nan

        numeric_columns = ["strike", "size", "entry_price", "theoretical_prob", "polymarket_price", "spread", "pnl"]
        for col in numeric_columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        clean_df["size"] = clean_df["size"].fillna(0.0).clip(lower=0.0)
        clean_df["entry_price"] = clean_df["entry_price"].fillna(0.5).clip(lower=1e-4, upper=1 - 1e-4)
        clean_df["theoretical_prob"] = clean_df["theoretical_prob"].fillna(0.5).clip(lower=1e-4, upper=1 - 1e-4)
        clean_df["polymarket_price"] = clean_df["polymarket_price"].fillna(0.5).clip(lower=1e-4, upper=1 - 1e-4)
        clean_df["spread"] = clean_df["spread"].fillna(0.0)
        clean_df["pnl"] = clean_df["pnl"].fillna(0.0)

        return clean_df[clean_df["size"] > 0].copy()

    def _repair_trades_csv_if_needed(self) -> None:
        """Repair malformed trade file by enforcing expected schema."""
        try:
            df = pd.read_csv(self.trades_csv_path)
            if set(self.TRADE_COLUMNS).issubset(df.columns):
                return

            repaired = self._sanitize_trade_dataframe(df)
            repaired = repaired[self.TRADE_COLUMNS]
            repaired.to_csv(self.trades_csv_path, index=False)
            logger.warning("Repaired malformed trades CSV schema at %s", self.trades_csv_path)
        except Exception as exc:
            logger.error("Failed to repair trades CSV. Reinitializing file: %s", exc)
            pd.DataFrame(columns=self.TRADE_COLUMNS).to_csv(self.trades_csv_path, index=False)

    @staticmethod
    def _ensure_parent_directory(path: str) -> None:
        """Ensure parent directory exists."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
        """Write metrics JSON atomically."""
        self._ensure_parent_directory(self.metrics_json_path)
        tmp_path = f"{self.metrics_json_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.metrics_json_path)

    def _load_metrics(self) -> Optional[Dict[str, Any]]:
        """Load metrics JSON if valid."""
        try:
            with open(self.metrics_json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Could not load existing metrics file: %s", exc)
        return None
