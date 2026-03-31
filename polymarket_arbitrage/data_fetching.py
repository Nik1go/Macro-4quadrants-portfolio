"""Data fetching layer for Polymarket arbitrage.

Features:
- Async market data retrieval (Polymarket + Binance)
- Exponential backoff with jitter
- Lightweight circuit breaker protection
- Outlier filtering (z-score + IQR)
- Local fallback cache for last valid prices
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import re
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

from utils.config import Config
from utils.logger import Logger

logger = Logger().logger


class CircuitBreaker:
    """Simple circuit breaker to protect unstable upstream dependencies."""

    def __init__(self, threshold: int, cooldown_seconds: int) -> None:
        self.threshold = max(int(threshold), 1)
        self.cooldown_seconds = max(int(cooldown_seconds), 1)
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        return time.time() >= self.open_until

    def success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.time() + self.cooldown_seconds


async def _with_retry(coro_factory, label: str) -> Any:
    """Run async callable with exponential backoff and jitter."""
    last_error: Optional[Exception] = None

    for attempt in range(1, Config.RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            delay = min(
                Config.RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25),
                Config.RETRY_MAX_DELAY,
            )
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt, Config.RETRY_ATTEMPTS, exc)
            if attempt < Config.RETRY_ATTEMPTS:
                await asyncio.sleep(delay)

    raise RuntimeError(f"{label} failed after retries: {last_error}")


@dataclass
class PriceSnapshot:
    """Local fallback cache payload."""

    timestamp: float
    value: float


class OutlierFilter:
    """Robust outlier detector using rolling z-score and IQR checks."""

    def __init__(self, maxlen: int = 256) -> None:
        self.history: Deque[float] = deque(maxlen=maxlen)

    def is_valid(self, value: float) -> bool:
        clean = float(value)
        if np.isnan(clean) or np.isinf(clean) or clean <= 0:
            return False

        if len(self.history) < 20:
            self.history.append(clean)
            return True

        series = pd.Series(list(self.history), dtype=float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = max(q3 - q1, 1e-12)

        z = abs((clean - mean) / max(std, 1e-12))
        low = q1 - Config.OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + Config.OUTLIER_IQR_MULTIPLIER * iqr

        is_outlier = (z > Config.OUTLIER_ZSCORE_THRESHOLD) or (clean < low) or (clean > high)
        if not is_outlier:
            self.history.append(clean)
        return not is_outlier


class CryptoFetcher:
    """Fetch spot, orderbook, volatility and funding from Binance via ccxt."""

    def __init__(self, exchange_id: str = Config.EXCHANGE_ID) -> None:
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
                "options": {"defaultType": Config.BINANCE_DEFAULT_TYPE},
                "apiKey": Config.BINANCE_API_KEY,
                "secret": Config.BINANCE_SECRET,
            }
        )
        self.breaker = CircuitBreaker(Config.CIRCUIT_BREAKER_THRESHOLD, Config.CIRCUIT_BREAKER_COOLDOWN)
        self.price_filters: Dict[str, OutlierFilter] = {}
        self.last_prices: Dict[str, PriceSnapshot] = {}

    def _get_filter(self, symbol: str) -> OutlierFilter:
        if symbol not in self.price_filters:
            self.price_filters[symbol] = OutlierFilter()
        return self.price_filters[symbol]

    async def fetch_spot_price(self, symbol: str = "BTC/USDT") -> Optional[float]:
        """Fetch last trade price with outlier and cache fallback checks."""
        if not self.breaker.allow():
            cached = self.last_prices.get(symbol)
            return cached.value if cached else None

        async def _call() -> float:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker.get("last", np.nan))

        try:
            price = await _with_retry(_call, f"fetch_spot_price:{symbol}")
            if not self._get_filter(symbol).is_valid(price):
                logger.warning("Outlier spot price ignored for %s: %.6f", symbol, price)
                cached = self.last_prices.get(symbol)
                return cached.value if cached else None

            self.last_prices[symbol] = PriceSnapshot(timestamp=time.time(), value=price)
            self.breaker.success()
            return price
        except Exception as exc:
            self.breaker.failure()
            logger.error("Spot price fetch failed for %s: %s", symbol, exc)
            cached = self.last_prices.get(symbol)
            return cached.value if cached else None

    async def fetch_historical_volatility(self, symbol: str, timeframe: str = "1d", lookback: int = 30) -> float:
        """Return annualized volatility computed from log-returns."""

        async def _call() -> float:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=max(lookback, 10))
            closes = np.array([float(item[4]) for item in ohlcv if item and len(item) >= 5], dtype=float)
            closes = closes[np.isfinite(closes) & (closes > 0)]
            if len(closes) < 10:
                return 0.60
            returns = np.diff(np.log(closes))
            returns = returns[np.isfinite(returns)]
            if len(returns) < 5:
                return 0.60
            return float(np.std(returns, ddof=0) * np.sqrt(365.25))

        try:
            value = await _with_retry(_call, f"fetch_historical_volatility:{symbol}")
            return float(np.clip(value, 0.05, 3.0))
        except Exception as exc:
            logger.error("Volatility fetch failed for %s: %s", symbol, exc)
            return 0.60

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Fetch Binance orderbook depth."""

        async def _call() -> Dict[str, Any]:
            return await self.exchange.fetch_order_book(symbol=symbol, limit=max(limit, 5))

        try:
            return await _with_retry(_call, f"fetch_orderbook:{symbol}")
        except Exception as exc:
            logger.error("Orderbook fetch failed for %s: %s", symbol, exc)
            return None

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch perpetual funding rate, defaulting to 0 on unsupported exchanges."""

        async def _call() -> float:
            if hasattr(self.exchange, "fetch_funding_rate"):
                data = await self.exchange.fetch_funding_rate(symbol)
                return float(data.get("fundingRate", 0.0))
            return 0.0

        try:
            value = await _with_retry(_call, f"fetch_funding_rate:{symbol}")
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(np.clip(value, -0.05, 0.05))
        except Exception as exc:
            logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)
            return 0.0

    async def close(self) -> None:
        """Close ccxt exchange session."""
        await self.exchange.close()


class RiskFreeRateFetcher:
    """Fetch risk-free rate from macro CSV."""

    @staticmethod
    def get_risk_free_rate() -> float:
        """Read the latest rate from configured CSV path."""
        path = Path(Config.MACRO_DATA_PATH)
        if not path.exists():
            logger.warning("Risk-free CSV missing at %s. Using fallback=5%%.", path)
            return 0.05

        try:
            df = pd.read_csv(path)
            if "value" not in df.columns or df.empty:
                return 0.05
            rate = float(df["value"].dropna().iloc[-1]) / 100.0
            if np.isnan(rate) or np.isinf(rate):
                return 0.05
            return float(np.clip(rate, -0.05, 0.20))
        except Exception as exc:
            logger.error("Risk-free rate loading failed: %s", exc)
            return 0.05


class PolymarketFetcher:
    """Fetch market list and orderbooks from Polymarket Gamma endpoints."""

    def __init__(self) -> None:
        self.base_url = "https://gamma-api.polymarket.com"
        self.breaker = CircuitBreaker(Config.CIRCUIT_BREAKER_THRESHOLD, Config.CIRCUIT_BREAKER_COOLDOWN)

    async def discover_price_markets(self, asset_name: str) -> List[Dict[str, Any]]:
        """Discover active markets using targeted date-based queries (J to J+7)."""
        if not self.breaker.allow():
            return []

        # Target the next 7 days specifically to avoid search noise (e.g., "Bitcoin above on March 30")
        base_date = datetime.now()
        queries = []
        for i in range(8):
            target = base_date + timedelta(days=i)
            # Format: "March 30" (Polymarket standard)
            date_str = target.strftime("%B %d").replace(" 0", " ")
            queries.append(f"{asset_name} above on {date_str}")

        raw_markets: List[Dict[str, Any]] = []
        
        async def _fetch_query(q: str) -> List[Dict[str, Any]]:
            params = {"active": "true", "closed": "false", "query": q, "limit": 15}
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/markets", params=params)
                response.raise_for_status()
                return response.json()

        try:
            # We fetch all queries. 
            # Note: For production with many assets, we might want to parallelize this.
            for query in queries:
                try:
                    batch = await _with_retry(lambda: _fetch_query(query), f"query:{query}")
                    raw_markets.extend(batch)
                except Exception as exc:
                    logger.warning("Query failed for %s: %s", query, exc)
            
            self.breaker.success()
            
            # Deduplicate by slug
            seen_slugs = set()
            discovered: List[Dict[str, Any]] = []
            
            for market in raw_markets:
                slug = market.get("slug")
                if not slug or slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                
                try:
                    raw_ids = market.get("clobTokenIds")
                    clob_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else (raw_ids or [])
                except Exception:
                    clob_ids = []

                if len(clob_ids) < 2:
                    continue

                discovered.append(
                    {
                        "asset": asset_name,
                        "slug": str(slug),
                        "title": str(market.get("question") or market.get("title") or ""),
                        "token_id_yes": clob_ids[0],
                        "token_id_no": clob_ids[1],
                        "ends_at": market.get("endDate") or market.get("end_date") or "",
                    }
                )

            return discovered[: Config.MAX_MARKETS_PER_ASSET * 2]  # Allow more since we have multiple dates
        except Exception as exc:
            self.breaker.failure()
            logger.error("Targeted market discovery failed for %s: %s", asset_name, exc)
            return []
        except Exception as exc:
            self.breaker.failure()
            logger.error("Market discovery failed for %s: %s", asset_name, exc)
            return []

    async def fetch_orderbook_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch a simplified orderbook for a Polymarket slug."""
        if not self.breaker.allow():
            return None

        async def _call() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/markets", params={"slug": slug})
                response.raise_for_status()
                payload = response.json()
                if not payload:
                    raise ValueError("No market payload returned")
                return payload[0]

        try:
            market = await _with_retry(_call, f"fetch_poly_orderbook:{slug}")
            self.breaker.success()
            raw_prices = market.get("outcomePrices", ["0.5", "0.5"])
            prices = ["0.5", "0.5"]
            try:
                if isinstance(raw_prices, str):
                    prices = json.loads(raw_prices)
                elif isinstance(raw_prices, list):
                    prices = raw_prices
                
                # Double decoding check (sometimes APIs wrap JSON in strings twice)
                if isinstance(prices, str):
                    prices = json.loads(prices)
            except Exception:
                prices = ["0.5", "0.5"]

            # Final safety check before float conversion
            def _extract_price(v: Any) -> float:
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).strip()
                if s.startswith('[') or s.startswith('{'):
                    try:
                        inner = json.loads(s)
                        if isinstance(inner, list) and inner:
                            return _extract_price(inner[0])
                        if isinstance(inner, dict):
                            return 0.5
                    except Exception:
                        pass
                # Regex extract first number
                match = re.search(r"(\d+\.\d+|\d+)", s)
                if match:
                    return float(match.group(1))
                return 0.5

            try:
                p_yes = _extract_price(prices[0]) if isinstance(prices, list) and prices else 0.5
                p_no = _extract_price(prices[1]) if isinstance(prices, list) and len(prices) > 1 else (1.0 - p_yes)
                
                # Clip to avoid exact 0 or 1
                p_yes = float(np.clip(p_yes, 1e-4, 1 - 1e-4))
                p_no = float(np.clip(p_no, 1e-4, 1 - 1e-4))

                # Approximate depth
                return {
                    "yes": p_yes,
                    "no": p_no,
                    "bids": [{"price": p_yes * 0.99, "size": 500}],
                    "asks": [{"price": p_yes * 1.01, "size": 500}],
                }
            except Exception as exc:
                logger.error("Deep parsing logic error for %s: %s", slug, exc)
                return {"yes": 0.5, "no": 0.5, "bids": [], "asks": []}
        except Exception as exc:
            self.breaker.failure()
            logger.error("Gamma market fetch failed for %s: %s", slug, exc)
            return None
