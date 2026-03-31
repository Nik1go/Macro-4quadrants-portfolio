"""Centralized logging with size-based rotation for production deployment."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import Config


class Logger:
    """Singleton-like logger factory for the arbitrage service."""

    _logger: Optional[logging.Logger] = None

    def __init__(self, name: str = "polymarket_arbitrage") -> None:
        if Logger._logger is None:
            Logger._logger = self._build_logger(name=name)
        self.logger = Logger._logger

    @staticmethod
    def _build_logger(name: str) -> logging.Logger:
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        file_path = Path(Config.LOG_DIR) / "arbitrage.log"
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger
