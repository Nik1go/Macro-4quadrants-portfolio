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
import pytz

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
    """Strictly filter for crypto price markets: [Asset] above [Strike] on [Date]."""
    if not isinstance(title, str) or not isinstance(slug, str):
        return False
        
    t = title.lower()
    s = slug.lower()
    
    # 1. Asset check (exact whitelist)
    allowed_assets = ["bitcoin", "ethereum", "xrp", "btc", "eth"]
    if not any(a in t for a in allowed_assets) and not any(a in s for a in allowed_assets):
        return False

    # 2. Pattern check
    # We want markets like "Bitcoin above 70000 on April 05" or slug "bitcoin-above-70000-on-april-05"
    has_keywords = "above" in s or "above" in t
    has_date_indicator = "on" in s or "on" in t
    
    # Noise rejection (Hard exclusion)
    noise = ["gta", "movie", "album", "ceasefire", "trump", "election", "war", "announced", "trailer"]
    if any(n in s for n in noise) or any(n in t for n in noise):
        return False

    # 3. Final validation: must look like a price bet
    # Slug usually: bitcoin-above-70000-on-april-05-2024
    # Title usually: Will Bitcoin be above $70,000 on April 05?
    match_regex = re.search(r"above|below|hit|reach", s + " " + t)
    return bool(match_regex and has_date_indicator)

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


def compute_binary_pnl(
    direction: str, 
    entry_price: float, 
    current_price: float, 
    size: float,
    entry_exchange_px: float = 0.0,
    exit_exchange_px: float = 0.0
) -> float:
    """Compute combined realized pnl (Polymarket + Binance leg)."""
    entry = float(np.clip(entry_price, 1e-4, 1 - 1e-4))
    current = float(np.clip(current_price, 1e-4, 1 - 1e-4))
    notional = max(float(size), 0.0)
    side = direction.lower().strip()

    # Leg A: Polymarket (Binary Outcome)
    poly_pnl = 0.0
    if "buy_yes" in side or side in {"yes", "long_yes"}:
        poly_pnl = (current - entry) * notional
    elif "buy_no" in side or side in {"no", "long_no"}:
        poly_pnl = (entry - current) * notional
    
    # Leg B: Binance (Hedge)
    # Si on a des prix Binance, on calcule Gain = (Entry - Exit) pour un Short (buy_yes)
    # ou Gain = (Exit - Entry) pour un Long (buy_no)
    binance_pnl = 0.0
    if entry_exchange_px > 0 and exit_exchange_px > 0:
        qty_crypto = notional / max(entry_exchange_px, 1e-9)
        if "buy_yes" in side or side in {"yes", "long_yes"}:
            # On a Shorté sur Binance pour protéger un Long Yes Poly
            binance_pnl = (entry_exchange_px - exit_exchange_px) * qty_crypto
        else:
            # On a Longé sur Binance pour protéger un Long No Poly
            binance_pnl = (exit_exchange_px - entry_exchange_px) * qty_crypto
            
    return poly_pnl + binance_pnl
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
                ends_at=str(payload.get("ends_at", "?")),
                entry_exchange_px=float(payload.get("entry_exchange_px", 0.0)),
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
                "ends_at": item.position.ends_at,
                "entry_exchange_px": item.position.entry_exchange_px,
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
    final_settlement: bool = False,
    exit_exchange_px: float = 0.0,
) -> float:
    """Close existing position and return realized pnl."""
    # En cas de maturité, le prix de sortie Poly est forcé à 1.0 ou 0.0
    exit_price = current_price
    if final_settlement:
        # Heuristique : si le prix est > 0.5 à l'échéance, c'est gagné ($1), sinon perdu ($0)
        exit_price = 1.0 if current_price > 0.5 else 0.0
        logger.info("[SETTLEMENT] Slug %s settled at %.2f (Final Poly Value: $%.2f)", slug, current_price, exit_price)

    pnl = compute_binary_pnl(
        direction=open_trade.position.direction,
        entry_price=open_trade.position.entry_price,
        current_price=exit_price,
        size=open_trade.position.size_usd,
        entry_exchange_px=open_trade.position.entry_exchange_px,
        exit_exchange_px=exit_exchange_px,
    )

    signal = {
        "slug": slug,
        "asset": open_trade.position.asset,
        "strategy": open_trade.position.strategy,
        "direction": "close",
        "size": open_trade.position.size_usd,
        "entry_price": exit_price,
        "polymarket_price": exit_price,
        "theoretical_prob": open_trade.theoretical_prob,
        "spread": 0.0,
        "net_spread": 0.0,
        "expected_profit": pnl,
        "realized_pnl": pnl,
        "signal_meta": {
            "hedge_side": "buy" if "buy_yes" in open_trade.position.direction else "sell",
            "close_reason": reason,
        }
    }
    result = await execution.execute_arbitrage(signal)
    if result.get("status") == "FILLED":
        logger.info("Closed position %s reason=%s pnl=%.4f", slug, reason, pnl)
    return pnl

async def check_open_maturities(
    open_positions: Dict[str, OpenTrade], 
    available_capital: float, 
    execution_engine: ExecutionEngine,
    polymarket_fetcher: PolymarketFetcher,
    crypto_fetcher: CryptoFetcher
) -> float:
    """Triple-Check for maturity at 12 PM NY time."""
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)
    realized_this_cycle = 0.0
    
    to_close = []
    for slug, trade in open_positions.items():
        # Utilisation du nouvel attribut ends_at de la classe Position
        ends_at_str = str(trade.position.ends_at)
        
        if not ends_at_str or ends_at_str == "?":
            continue
            
        try:
            # Parse ISO date
            expiry_dt = datetime.fromisoformat(ends_at_str.replace("Z", "+00:00")).astimezone(ny_tz)
            
            # Condition 1 & 2 : Date correspondante et >= 12h00 NY
            is_expiry_time = (now_ny >= expiry_dt) or (now_ny.date() == expiry_dt.date() and now_ny.hour >= 12)
            
            if is_expiry_time:
                # Condition 3 : Vérification du prix settle sur Polymarket
                book = await polymarket_fetcher.fetch_orderbook_by_slug(slug)
                poly_price = book["yes"] if book else 0.5
                
                # Fetch current Binance price for final PnL
                symbol = Config.ASSET_TO_SYMBOL.get(trade.position.asset, "BTC/USDT")
                exit_px = await crypto_fetcher.fetch_spot_price(symbol) or 0.0
                
                # On ferme si le prix est quasi settled (1 ou 0) OU si on a passé midi de 5 min
                is_settled = poly_price > 0.95 or poly_price < 0.05
                time_buffer_ok = (now_ny - expiry_dt).total_seconds() > 300 # 5 min de sécu
                
                if is_settled or time_buffer_ok:
                    logger.info("[MATURITY] Triple-check OK for %s. Closing now.", slug)
                    to_close.append((slug, trade, poly_price))
                    
        except Exception as e:
            logger.error("Error checking maturity for %s: %s", slug, e)
            continue
            
    for slug, trade, poly_price, exit_px in to_close:
        pnl = await close_position(
            slug=slug,
            open_trade=trade,
            current_price=poly_price,
            execution=execution_engine,
            reason="maturity_triple_check",
            final_settlement=True,
            exit_exchange_px=exit_px
        )
        realized_this_cycle += pnl
        del open_positions[slug]
        
    return realized_this_cycle


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

            # [TRIPLE CHECK] Gestion des maturités (Midi NY)
            capital_recovered = await check_open_maturities(
                open_positions=open_positions,
                available_capital=available_capital,
                execution_engine=execution_engine,
                polymarket_fetcher=polymarket_fetcher,
                crypto_fetcher=crypto_fetcher
            )
            available_capital += capital_recovered
            persist_bot_state(storage, available_capital, open_positions)

            # [CORRECTION] Calcul sur l'Equity Totale (Cash + Investi) et non juste le Cash
            invested_value = sum(p.position.size_usd for p in open_positions.values())
            total_equity = available_capital + invested_value
            
            drawdown_state = risk_manager.update_drawdown(total_equity)
            if drawdown_state["breached"]:
                health_state.ready = False
                logger.warning("Bot in safety pause: Drawdown exceeded (Equity: $%.2f)", total_equity)
                await asyncio.sleep(Config.SCAN_INTERVAL)
                continue

            health_state.ready = True
            
            # Reset counters and candidates for this cycle
            total_scanned = 0
            eligible_markets = 0
            max_spread = 0.0
            candidates = []

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
                    slug  = str(market.get("slug", ""))

                    if not slug:
                        continue

                    group_strike_raw = market.get("group_strike")
                    if group_strike_raw is None and not is_crypto_price_market(title, slug):
                        continue

                    # Strike extraction
                    if group_strike_raw is not None:
                        try:
                            strike = float(str(group_strike_raw).replace(",", "").replace(" ", ""))
                        except (ValueError, TypeError):
                            strike = None
                    else:
                        strike = extract_strike_from_title(title)

                    if strike is None or strike <= 0:
                        continue

                    ttm = pricing_model.calculate_time_to_maturity(market.get("ends_at"))
                    ttm_days = ttm * 365.25
                    if ttm_days < Config.MIN_TTM_DAYS or ttm_days > Config.MAX_TTM_DAYS:
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
                        strategy_name=strategy.name,
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

                    # Persistence massive pour historique UI
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
                            "is_opportunity": int(signal.should_trade and abs(net_spread) >= Config.MIN_BATCH_EDGE),
                            "strategy": strategy.name,
                            "slug": slug,
                            "signal_type": str(signal.direction).replace("_", " ").upper(),
                        }
                    )
                    
                    eligible_markets += 1
                    max_spread = max(max_spread, abs(float(net_spread)))

                    # Closing check (happens inside the round scan)
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

                    # Potential entry candidate
                    if signal.should_trade and abs(net_spread) >= Config.MIN_BATCH_EDGE:
                        # Extra risk checks before adding to candidates
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
                            logger.info("%s candidate rejected slug=%s reason=%s", strategy.name, slug, risk_reason)
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
                            logger.info("%s candidate rejected slug=%s reason=zero_position_size (cap=%.2f, edge=%.4f)", 
                                        strategy.name, slug, available_capital, net_spread)
                            continue

                        candidates.append({
                            "payload": {
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
                                "expected_profit": size * net_spread,
                                "signal_reason": signal.reason,
                                "signal_meta": signal.metadata,
                                "risk_reason": risk_reason,
                                "ends_at": market.get("ends_at"),
                            },
                            "signal": signal,
                        })

            # ── Fin du tour complet : Tri et Exécution ──────────────────────────
            if candidates:
                # Sort by net_spread DESC
                candidates.sort(key=lambda x: x["payload"]["net_spread"], reverse=True)
                
                # Take top N
                top_targets = candidates[:Config.MAX_TRADES_PER_ROUND]
                logger.info("Round completed: Found %d potential trades. Selecting top %d.", 
                            len(candidates), len(top_targets))

                for item in top_targets:
                    trade_payload = item["payload"]
                    trade_signal = item["signal"]
                    slug = trade_payload["slug"]

                    # Safety: re-check if already in open_positions (paranoia)
                    if slug in open_positions:
                        continue

                    # Re-check available capital (it might have decreased during this batch execution)
                    if available_capital < trade_payload["size"]:
                        logger.warning("Skipping %s: insufficient capital ($%.2f vs target $%.2f)", 
                                       slug, available_capital, trade_payload["size"])
                        continue

                    result = await execution_engine.execute_arbitrage(trade_payload)
                    if result.get("status") == "FILLED":
                        available_capital -= trade_payload["size"]
                        pos = risk_manager.generate_position(
                            position_id=slug,
                            asset=trade_payload["asset"],
                            direction=trade_signal.direction,
                            strategy=strategy.name,
                            size_usd=trade_payload["size"],
                            entry_price=trade_payload["entry_price"],
                            target_delta=trade_signal.target_delta,
                            ends_at=trade_payload.get("ends_at", "?"),
                            entry_exchange_px=trade_payload.get("spot_price", 0.0),
                        )
                        open_positions[slug] = OpenTrade(
                            slug=slug,
                            position=pos,
                            strike=trade_payload["strike"],
                            theoretical_prob=trade_payload["theoretical_prob"],
                        )
                        storage.save_strategy_metric(strategy.name, "executed_net_spread", trade_payload["net_spread"])

            persist_bot_state(storage, available_capital, open_positions)
            logger.info(
                "Cycle completed: %d scanned, %d eligible, %d candidates, Round Max Net Spread: %.2f%%",
                total_scanned,
                eligible_markets,
                len(candidates),
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
