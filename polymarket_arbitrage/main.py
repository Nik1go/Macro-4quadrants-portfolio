"""Main orchestrator for the Polymarket multi-strategy arbitrage bot."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
import datetime
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from data_fetching import CryptoFetcher, PolymarketFetcher, RiskFreeRateFetcher
from execution_engine import ExecutionEngine
from pricing_model import PricingModel
from risk_management import Position, RiskManager
from storage import StorageManager
from strategies import DeltaNeutralStrategy, PurePolymarketStrategy, StrategyContext
from utils.config import Config
from utils.health_check import BotHealthState, start_health_server
from utils.logger import Logger
from utils.notifier import Notifier

logger = Logger().logger


@dataclass
class OpenTrade:
    """Track minimal state for open positions."""

    slug: str
    position: Position
    strike: float
    theoretical_prob: float


class FileLock:
    """Simple lock-file guard to prevent multi-instance execution."""

    def __init__(self, lock_path: str) -> None:
        self.lock_path = Path(lock_path)

    def acquire(self) -> None:
        if self.lock_path.exists():
            raise RuntimeError(f"Lock file exists: {self.lock_path}")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(str(Path.cwd()), encoding="utf-8")

    def release(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()


def is_crypto_price_market(title: str, slug: str = "") -> bool:
    """Filter out noise (GTA, movies, etc.) using title and slug to keep only pure price action."""
    if not isinstance(title, str):
        return False
    t = (title + " " + slug).lower()
    # Forbidden keywords for pure price arbitrage
    noise = ["gta", "movie", "trailer", "release", "released", "announced", "game", "official", "rockstar"]
    if any(n in t for n in noise):
        return False
    # Must contain price-related keywords
    must_have = ["price", "above", "below", "hit", "reach", "level", "value", "$"]
    return any(m in t for m in must_have)

def extract_strike_from_title(title: str) -> Optional[float]:
    """Extract strike from market title strings."""
    if not isinstance(title, str) or not title.strip():
        return None

    clean = title.lower().replace(",", "")
    match = re.search(r"\$?(\d+(?:\.\d+)?)\s*k\b|\$?(\d+(?:\.\d+)?)", clean)
    if not match:
        return None
    if match.group(1):
        return float(match.group(1)) * 1000.0
    if match.group(2):
        return float(match.group(2))
    return None


def compute_binary_pnl(direction: str, entry_price: float, current_price: float, size: float) -> float:
    """Mark-to-market pnl for binary yes/no position."""
    entry = float(np.clip(entry_price, 1e-4, 1 - 1e-4))
    current = float(np.clip(current_price, 1e-4, 1 - 1e-4))
    notional = max(float(size), 0.0)
    side = direction.lower().strip()

    if "buy_yes" in side or side in {"yes", "long_yes"}:
        return (current - entry) * notional
    if "buy_no" in side or side in {"no", "long_no"}:
        return (entry - current) * notional
    return 0.0


def load_bot_state(storage: StorageManager) -> Dict[str, OpenTrade]:
    """Rebuild open positions from persisted bot state."""
    state = storage.load_bot_state("runtime") or {}
    open_positions: Dict[str, OpenTrade] = {}

    raw_positions = state.get("open_positions", {}) if isinstance(state, dict) else {}
    for slug, payload in raw_positions.items():
        try:
            position = Position(
                position_id=str(payload["position_id"]),
                asset=str(payload["asset"]),
                direction=str(payload["direction"]),
                strategy=str(payload.get("strategy", "pure_polymarket")),
                size_usd=float(payload["size_usd"]),
                entry_price=float(payload["entry_price"]),
                stop_loss_price=float(payload["stop_loss_price"]),
                opened_at=np.datetime64(payload["opened_at"]).astype("datetime64[ms]").astype(object),
                target_delta=float(payload.get("target_delta", 0.0)),
            )
            open_positions[str(slug)] = OpenTrade(
                slug=str(slug),
                position=position,
                strike=float(payload.get("strike", 0.0)),
                theoretical_prob=float(payload.get("theoretical_prob", 0.5)),
            )
        except Exception:
            continue

    return open_positions


def persist_bot_state(storage: StorageManager, capital: float, open_positions: Dict[str, OpenTrade]) -> None:
    """Persist runtime state for crash-safe restart."""
    payload = {
        "capital": float(capital),
        "open_positions": {
            slug: {
                "position_id": item.position.position_id,
                "asset": item.position.asset,
                "direction": item.position.direction,
                "strategy": item.position.strategy,
                "size_usd": item.position.size_usd,
                "entry_price": item.position.entry_price,
                "stop_loss_price": item.position.stop_loss_price,
                "opened_at": item.position.opened_at.isoformat(),
                "target_delta": item.position.target_delta,
                "strike": item.strike,
                "theoretical_prob": item.theoretical_prob,
            }
            for slug, item in open_positions.items()
        },
    }
    storage.save_bot_state("runtime", payload)


async def close_position(
    slug: str,
    open_trade: OpenTrade,
    current_price: float,
    execution: ExecutionEngine,
    reason: str,
) -> float:
    """Close existing position and return realized pnl."""
    pnl = compute_binary_pnl(
        direction=open_trade.position.direction,
        entry_price=open_trade.position.entry_price,
        current_price=current_price,
        size=open_trade.position.size_usd,
    )

    signal = {
        "slug": slug,
        "asset": open_trade.position.asset,
        "strategy": open_trade.position.strategy,
        "direction": "close",
        "size": open_trade.position.size_usd,
        "entry_price": current_price,
        "polymarket_price": current_price,
        "theoretical_prob": open_trade.theoretical_prob,
        "spread": 0.0,
        "net_spread": 0.0,
        "expected_profit": pnl,
        "realized_pnl": pnl,
    }
    result = await execution.execute_arbitrage(signal)
    if result.get("status") == "FILLED":
        logger.info("Closed position %s reason=%s pnl=%.4f", slug, reason, pnl)
    return pnl


async def main_loop() -> None:
    """Run the main multi-strategy scan and execution loop."""
    Config.validate()

    lock = FileLock(Config.LOCK_FILE_PATH)
    health_state = BotHealthState()
    notifier = Notifier(enabled=Config.ENABLE_NOTIFIER)

    lock.acquire()
    logger.info("Starting health server on %s:%s", Config.HEALTH_HOST, Config.HEALTH_PORT)
    start_health_server(health_state)

    logger.info("Starting strategy=%s paper=%s", Config.STRATEGY_NAME, Config.PAPER_TRADE)

    storage = StorageManager(db_path=Config.DB_PATH)
    storage.save_config_snapshot(
        strategy=Config.STRATEGY_NAME,
        config_data={
            "scan_interval": Config.SCAN_INTERVAL,
            "min_edge": Config.MIN_SPREAD_THRESHOLD,
            "strategy": Config.STRATEGY_NAME,
        },
    )

    if Config.STRATEGY_NAME == "delta_neutral":
        strategy = DeltaNeutralStrategy()
    else:
        strategy = PurePolymarketStrategy()

    crypto_fetcher = CryptoFetcher(exchange_id=Config.EXCHANGE_ID)
    polymarket_fetcher = PolymarketFetcher()
    pricing_model = PricingModel()
    risk_manager = RiskManager(
        initial_capital=Config.INITIAL_CAPITAL,
        max_drawdown_pct=Config.MAX_DRAWDOWN_PCT,
        stop_loss_pct=Config.STOP_LOSS_PCT,
        max_position_size=Config.MAX_POSITION_SIZE,
    )
    execution_engine = ExecutionEngine(is_paper_trade=Config.PAPER_TRADE, storage_manager=storage)

    recovered_positions = load_bot_state(storage)
    state = storage.load_bot_state("runtime") or {}
    available_capital = float(state.get("capital", Config.INITIAL_CAPITAL))
    open_positions: Dict[str, OpenTrade] = dict(recovered_positions)

    try:
        while True:
            health_state.mark_alive()
            risk_free_rate = RiskFreeRateFetcher.get_risk_free_rate()

            drawdown_state = risk_manager.update_drawdown(available_capital)
            if drawdown_state["breached"]:
                health_state.ready = False
                await notifier.send("⚠️ Drawdown breached. Trading paused.")
                await asyncio.sleep(Config.SCAN_INTERVAL)
                continue

            health_state.ready = True
            
            # Reset counters for this cycle
            total_scanned = 0
            eligible_markets = 0
            max_spread = 0.0

            for asset in Config.TARGET_ASSETS:
                symbol = Config.ASSET_TO_SYMBOL.get(asset)
                if not symbol:
                    continue

                # Broad search using the tactical 'above' query defined in data_fetching
                all_markets = await polymarket_fetcher.discover_price_markets(asset_name=asset)
                
                if not all_markets:
                    logger.info("Scanner: 0 markets for %s", asset)
                    continue

                logger.info("Scanner: %d raw markets for %s", len(all_markets), asset)

                spot = await crypto_fetcher.fetch_spot_price(symbol)
                vol = await crypto_fetcher.fetch_historical_volatility(symbol)
                funding = await crypto_fetcher.fetch_funding_rate(symbol)
                hedge_book = await crypto_fetcher.fetch_orderbook(symbol)

                if spot is None or spot <= 0:
                    continue

                for market in all_markets:
                    title = str(market.get("question", market.get("title", "")))
                    slug = str(market.get("slug", ""))
                    if not is_crypto_price_market(title, slug):
                        logger.debug("Skipping %s: NOT a crypto price market", slug or title[:30])
                        continue
                    if not slug:
                        continue

                    strike = extract_strike_from_title(str(market.get("title", "")))
                    if strike is None or strike <= 0:
                        logger.debug("Skipping %s: could not extract strike from title: %s", slug, title)
                        continue

                    ttm = pricing_model.calculate_time_to_maturity(market.get("ends_at"))
                    ttm_days = ttm * 365.25
                    if ttm_days < Config.MIN_TTM_DAYS or ttm_days > Config.MAX_TTM_DAYS:
                        logger.debug("Skipping %s: TTM range failed (%.2f days vs config [%d, %d])", 
                                     slug, ttm_days, Config.MIN_TTM_DAYS, Config.MAX_TTM_DAYS)
                        continue

                    book = await polymarket_fetcher.fetch_orderbook_by_slug(slug)
                    if not book:
                        continue
                    
                    total_scanned += 1

                    poly_price, liquid = risk_manager.check_liquidity_and_slippage(book, target_size=1.0, side="buy")
                    if not liquid:
                        continue

                    if hedge_book is None and strategy.name == "delta_neutral":
                        continue

                    poly_price = float(np.clip(poly_price, 1e-4, 1 - 1e-4))
                    theo = pricing_model.binary_call_option_price(
                        s=spot,
                        k=strike,
                        t=ttm,
                        r=risk_free_rate,
                        sigma=vol,
                    )
                    raw_spread = pricing_model.calculate_mispricing(theoretical_prob=theo, polymarket_price=poly_price)
                    net_spread = risk_manager.calculate_net_spread(
                        raw_spread=raw_spread,
                        position_size_usd=max(Config.MIN_POSITION_SIZE, 1.0),
                    )

                    context = StrategyContext(
                        asset=asset,
                        symbol=symbol,
                        slug=slug,
                        title=str(market.get("title", "")),
                        strike=strike,
                        spot_price=float(spot),
                        theoretical_prob=float(theo),
                        polymarket_price=float(poly_price),
                        time_to_maturity=float(ttm),
                        implied_volatility=float(vol),
                        net_spread=float(net_spread),
                        funding_rate=float(funding),
                        orderbook_depth_ok=bool(liquid),
                    )

                    signal = strategy.generate_signal(context)

                    storage.save_spread(
                        {
                            "timestamp": np.datetime_as_string(np.datetime64("now"), unit="s"),
                            "asset_pair": symbol,
                            "spot_price": spot,
                            "implied_vol": vol,
                            "polymarket_price": poly_price,
                            "theoretical_prob": theo,
                            "rfr": risk_free_rate,
                            "net_spread": net_spread,
                            "is_opportunity": int(signal.should_trade),
                            "strategy": strategy.name,
                            "slug": slug,
                        }
                    )
                    
                    eligible_markets += 1
                    max_spread = max(max_spread, float(net_spread))

                    if slug in open_positions:
                        tracked = open_positions[slug]
                        if risk_manager.check_stop_loss(tracked.position, poly_price):
                            realized = await close_position(
                                slug=slug,
                                open_trade=tracked,
                                current_price=poly_price,
                                execution=execution_engine,
                                reason="stop_loss",
                            )
                            available_capital += realized
                            del open_positions[slug]
                        continue

                    if not signal.should_trade:
                        continue

                    current_delta = risk_manager.monitor_delta_neutrality(
                        {k: v.position for k, v in open_positions.items()}
                    )
                    is_risk_ok, risk_reason = risk_manager.validate_strategy_risk(
                        strategy_name=strategy.name,
                        net_spread=net_spread,
                        funding_rate=funding,
                        current_delta_exposure=current_delta,
                        liquidity_ok=liquid,
                    )
                    if not is_risk_ok:
                        continue

                    size = risk_manager.compute_strategy_position_size(
                        strategy_name=strategy.name,
                        capital_usd=available_capital,
                        theoretical_prob=theo,
                        market_price=poly_price,
                        volatility=vol,
                        net_spread=net_spread,
                    )
                    if size <= 0:
                        continue

                    expected_profit = size * net_spread
                    payload = {
                        "slug": slug,
                        "asset": asset,
                        "symbol": symbol,
                        "spot_price": spot,
                        "strike": strike,
                        "strategy": strategy.name,
                        "direction": signal.direction,
                        "size": size,
                        "entry_price": poly_price,
                        "polymarket_price": poly_price,
                        "theoretical_prob": theo,
                        "spread": raw_spread,
                        "net_spread": net_spread,
                        "expected_profit": expected_profit,
                        "signal_reason": signal.reason,
                        "signal_meta": signal.metadata,
                        "risk_reason": risk_reason,
                    }

                    result = await execution_engine.execute_arbitrage(payload)
                    if result.get("status") != "FILLED":
                        continue

                    available_capital -= size
                    pos = risk_manager.generate_position(
                        position_id=slug,
                        asset=asset,
                        direction=signal.direction,
                        strategy=strategy.name,
                        size_usd=size,
                        entry_price=poly_price,
                        target_delta=signal.target_delta,
                    )
                    open_positions[slug] = OpenTrade(
                        slug=slug,
                        position=pos,
                        strike=strike,
                        theoretical_prob=theo,
                    )

                    storage.save_strategy_metric(strategy.name, "executed_net_spread", net_spread)

            persist_bot_state(storage, available_capital, open_positions)
            logger.info(
                "Cycle completed: %d markets scanned, %d eligible, Max Net Spread: %.2f%%",
                total_scanned,
                eligible_markets,
                max_spread * 100.0,
            )
            await asyncio.sleep(Config.SCAN_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as exc:  # pragma: no cover
        logger.exception("Fatal error in main loop: %s", exc)
        await notifier.send(f"❌ Fatal bot error: {exc}")
    finally:
        for slug, trade in list(open_positions.items()):
            realized = await close_position(
                slug=slug,
                open_trade=trade,
                current_price=trade.position.entry_price,
                execution=execution_engine,
                reason="shutdown",
            )
            available_capital += realized
            open_positions.pop(slug, None)

        persist_bot_state(storage, available_capital, open_positions)
        await execution_engine.finalize()
        await crypto_fetcher.close()
        lock.release()


if __name__ == "__main__":
    asyncio.run(main_loop())
