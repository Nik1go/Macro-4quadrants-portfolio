"""
debug_fetch.py – Script de debug Polymarket
============================================
Objectif : Récupérer UNIQUEMENT les probabilités et prix des marchés
           BTC / ETH / XRP sur J+1 à J+7, sans bruit.

Stratégie : Utiliser l'endpoint /events?slug={asset}-above-on-{date}
            qui retourne en UN seul appel TOUS les strikes du jour
            pour l'actif demandé, proprement groupés.

Usage :
    cd polymarket_arbitrage
    python debug_fetch.py

    # Filtrer sur un seul asset :
    python debug_fetch.py --assets BTC

    # Changer la fenêtre de jours :
    python debug_fetch.py --days 1 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 15.0  # secondes

# Mapping asset → slug-prefix utilisé par Polymarket
ASSET_SLUG_PREFIX: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "XRP": "xrp",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def format_date_for_slug(dt: datetime) -> str:
    """
    Convertit une date en format slug Polymarket.
    Ex: datetime(2026, 4, 5) → 'april-5'  (sans zéro devant le jour)
    """
    month = dt.strftime("%B").lower()   # 'april'
    day   = str(dt.day)                 # '5' (pas '05')
    return f"{month}-{day}"


def parse_outcome_prices(raw: Any) -> tuple[float, float]:
    """
    Extrait (p_yes, p_no) depuis le champ outcomePrices qui peut être :
    - une string JSON :  '["0.85", "0.15"]'
    - une liste        : ["0.85", "0.15"]
    """
    try:
        if isinstance(raw, str):
            prices = json.loads(raw)
        elif isinstance(raw, list):
            prices = raw
        else:
            return 0.5, 0.5

        p_yes = float(prices[0]) if len(prices) > 0 else 0.5
        p_no  = float(prices[1]) if len(prices) > 1 else (1.0 - p_yes)
        return round(p_yes, 4), round(p_no, 4)
    except Exception:
        return 0.5, 0.5


def parse_strike_from_group_title(group_title: str) -> Optional[float]:
    """
    Extrait le strike depuis groupItemTitle (ex: '64,000' → 64000.0).
    Ce champ est beaucoup plus fiable que le titre textuel.
    """
    try:
        return float(group_title.replace(",", "").replace(" ", ""))
    except (ValueError, AttributeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FETCHER
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_event(client: httpx.AsyncClient, asset: str, target_date: datetime) -> Optional[Dict]:
    """
    Appelle GET /events?slug={prefix}-above-on-{date}
    Retourne le premier événement trouvé ou None.
    """
    prefix   = ASSET_SLUG_PREFIX[asset]
    date_str = format_date_for_slug(target_date)
    slug     = f"{prefix}-above-on-{date_str}"
    url      = f"{GAMMA_BASE}/events"
    params   = {"slug": slug}

    try:
        resp = await client.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            print(f"  ⚠  Aucun événement trouvé pour slug={slug}")
            return None
        return data[0]  # On prend le premier (unique slug = unique event)
    except httpx.HTTPStatusError as exc:
        print(f"  ✗  HTTP {exc.response.status_code} pour {slug}")
        return None
    except Exception as exc:
        print(f"  ✗  Erreur pour {slug}: {exc}")
        return None


def extract_markets_from_event(event: Dict, asset: str, target_date: datetime) -> List[Dict]:
    """
    Extrait la liste des marchés (1 par strike) depuis l'événement.
    Retourne une liste de dict propres avec uniquement les infos utiles.
    """
    raw_markets: List[Dict] = event.get("markets", [])
    results = []

    for m in raw_markets:
        # Vérification que le marché est actif et non fermé
        if not m.get("active", False) or m.get("closed", False):
            continue

        # Strike depuis groupItemTitle (fiable) ou slug en fallback
        strike = parse_strike_from_group_title(m.get("groupItemTitle", ""))

        # Probabilités
        p_yes, p_no = parse_outcome_prices(m.get("outcomePrices"))

        # Best bid / ask depuis le CLOB (données temps réel)
        best_bid = m.get("bestBid")
        best_ask = m.get("bestAsk")
        last_trade = m.get("lastTradePrice")

        # Liquidité
        liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0.0)

        # Volume 24h
        vol_24h = float(m.get("volume24hr") or 0.0)

        # Spread CLOB
        spread = m.get("spread")

        results.append({
            "asset"       : asset,
            "date"        : target_date.strftime("%Y-%m-%d"),
            "slug"        : m.get("slug", ""),
            "question"    : m.get("question", ""),
            "strike"      : strike,
            "p_yes"       : p_yes,
            "p_no"        : p_no,
            "best_bid"    : best_bid,
            "best_ask"    : best_ask,
            "last_trade"  : last_trade,
            "spread"      : spread,
            "liquidity"   : round(liquidity, 2),
            "volume_24h"  : round(vol_24h, 2),
            "end_date"    : m.get("endDateIso") or m.get("endDate", ""),
        })

    # Trie par strike croissant pour lisibilité
    results.sort(key=lambda x: x["strike"] or 0.0)
    return results


def print_markets(markets: List[Dict], asset: str, date_label: str) -> None:
    """Affiche un tableau propre dans le terminal."""
    if not markets:
        print(f"    → Aucun marché actif trouvé.")
        return

    print(f"    {'STRIKE':>10}  {'P_YES':>8}  {'P_NO':>8}  {'BID':>8}  {'ASK':>8}  {'SPREAD':>8}  {'LIQ ($)':>12}  {'VOL24H ($)':>12}  SLUG")
    print(f"    {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  ----")
    for m in markets:
        strike  = f"${m['strike']:,.0f}"    if m['strike']  is not None else "N/A"
        p_yes   = f"{m['p_yes']*100:.2f}%" if m['p_yes']   is not None else "N/A"
        p_no    = f"{m['p_no']*100:.2f}%"  if m['p_no']    is not None else "N/A"
        bid     = f"{m['best_bid']:.4f}"   if m['best_bid']  is not None else " N/A"
        ask     = f"{m['best_ask']:.4f}"   if m['best_ask']  is not None else " N/A"
        spread  = f"{m['spread']:.4f}"     if m['spread']    is not None else " N/A"
        liq     = f"${m['liquidity']:>10,.0f}"
        vol24h  = f"${m['volume_24h']:>10,.0f}"
        slug    = m['slug']
        print(f"    {strike:>10}  {p_yes:>8}  {p_no:>8}  {bid:>8}  {ask:>8}  {spread:>8}  {liq:>12}  {vol24h:>12}  {slug}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

async def run(assets: List[str], day_range: tuple[int, int]) -> None:
    """Point d'entrée principal."""
    now      = datetime.now(tz=timezone.utc)
    min_day, max_day = day_range

    print("=" * 80)
    print(f"  Polymarket Debug Fetch  –  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Assets  : {', '.join(assets)}")
    print(f"  Fenêtre : J+{min_day} → J+{max_day}")
    print("=" * 80)

    # Tous les (asset, date) à fetcher
    tasks = [
        (asset, now + timedelta(days=d))
        for asset in assets
        for d in range(min_day, max_day + 1)
    ]

    async with httpx.AsyncClient() as client:
        for asset, target_date in tasks:
            date_label = target_date.strftime("%A %d %B %Y")
            slug_date  = format_date_for_slug(target_date)
            print(f"\n  [{asset}]  {date_label}  (slug-date: {slug_date})")
            print(f"  {'-'*60}")

            event = await fetch_event(client, asset, target_date)
            if event is None:
                continue

            # Méta de l'événement
            total_liq = float(event.get("liquidity") or 0.0)
            total_vol = float(event.get("volume24hr") or 0.0)
            nb_markets = len(event.get("markets", []))
            print(f"  Event : {event.get('title', 'N/A')}")
            print(f"  Total : {nb_markets} marchés  |  Liquidité: ${total_liq:,.0f}  |  Vol 24h: ${total_vol:,.0f}")
            print()

            markets = extract_markets_from_event(event, asset, target_date)
            print_markets(markets, asset, date_label)

    print("\n" + "=" * 80)
    print("  Fetch terminé.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug fetch Polymarket – BTC/ETH/XRP J+1 à J+7")
    parser.add_argument(
        "--assets", nargs="+",
        choices=list(ASSET_SLUG_PREFIX.keys()),
        default=list(ASSET_SLUG_PREFIX.keys()),
        help="Assets à fetcher (défaut: BTC ETH XRP)"
    )
    parser.add_argument(
        "--days", nargs=2, type=int, metavar=("MIN", "MAX"),
        default=[1, 7],
        help="Fenêtre de jours J+MIN à J+MAX (défaut: 1 7)"
    )
    args = parser.parse_args()

    min_day, max_day = args.days
    if min_day < 0 or max_day < min_day:
        print("Erreur : --days MIN MAX avec 0 <= MIN <= MAX requis.")
        sys.exit(1)

    asyncio.run(run(assets=args.assets, day_range=(min_day, max_day)))


if __name__ == "__main__":
    main()
