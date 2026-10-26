"""Sélection de titres par colonie de fourmis, sans solveur de pondération.

Le même fichier alimente Streamlit et le notebook autonome généré par build_notebook.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from io import StringIO
import base64
import hashlib
import json
import math

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

START = "2022-01-01"
BENCHMARKS = {"^FCHI": "CAC 40", "^CM100": "CAC Mid 60"}
MID60_PAGE = "https://live.euronext.com/en/product/indices/QS0010989117-XPAR"
MID60_CHART = "https://live.euronext.com/intraday_chart/getChartData/QS0010989117-XPAR/max"
DATA_QUALITY_VERSION = "2026-10-26"
EXCLUDED_ASSETS = {
    "ATO.PA": "Exclu : échelles Yahoo incohérentes pendant la restructuration 2024 et le regroupement 2025 ; historique non reconstruit.",
    "EMEIS.PA": "Exclu : échelles Yahoo incohérentes pendant la restructuration 2023 ; historique non reconstruit.",
}
LARGE_MOVE_NOTES = {
    "ETL.PA": "Mars 2025 : mouvements conservés ; attribution précise à Starlink non confirmée par une source primaire.",
    "WLN.PA": "25/10/2023 : objectifs annuels revus lors des résultats T3 ; mouvement conservé.",
    "VU.PA": "23/06/2023 : après le rapport Gotham du 22/06, contesté par SES-imagotag ; mouvement conservé.",
    "NANO.PA": "05/05/2023 : annonce de négociations finales pour une licence NBTXR3 ; mouvement conservé.",
    "IVA.PA": "14/10/2024 : annonce d'un financement conditionnel jusqu'à 348 M€ ; mouvement conservé.",
    "DBV.PA": "09/01/2025 : après les résultats positifs EPITOPE publiés le 08/01 ; mouvement conservé.",
    "ATO.PA": "19/06/2024 : contexte de restructuration documenté, cause exacte de la hausse non confirmée ; série exclue pour incohérences ultérieures.",
}


def paris_today() -> str:
    """Borne exclusive : seules les séances antérieures au jour courant sont utilisées."""
    return datetime.now(ZoneInfo("Europe/Paris")).date().isoformat()


# %% Données et contrôle qualité
def decode_public_euronext(payload: dict, public_format_key: str):
    """Lire le format du graphique public, comme le JavaScript fourni sur la page.

    Aucune connexion ni donnée privée : la clé de format est publiée dans
    drupalSettings.ajax_secure.kye. Le format peut changer côté Euronext.
    """
    salt = bytes.fromhex(payload["s"])
    derived, previous = b"", b""
    while len(derived) < 48:
        previous = hashlib.md5(previous + public_format_key.encode() + salt).digest()
        derived += previous
    decoder = Cipher(algorithms.AES(derived[:32]), modes.CBC(bytes.fromhex(payload["iv"]))).decryptor()
    plain = decoder.update(base64.b64decode(payload["ct"])) + decoder.finalize()
    return json.loads(plain[:-plain[-1]])


def fetch_euronext_chart(product: str, category: str, start: str, end: str) -> pd.Series:
    """Graphique public de l'instrument exact ; ce n'est pas un export de clôtures certifiées."""
    with requests.Session() as session:
        page = session.get(f"https://live.euronext.com/en/product/{category}/{product}", timeout=40)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        node = soup.select_one('script[data-drupal-selector="drupal-settings-json"]')
        if node is None:
            raise ValueError("Format Euronext modifié : importer un CSV Date,Close de l'indice exact.")
        settings = json.loads(node.string)
        response = session.get(f"https://live.euronext.com/intraday_chart/getChartData/{product}/max", timeout=40)
        response.raise_for_status()
        payload = response.json()
        rows = decode_public_euronext(payload, settings["ajax_secure"]["kye"]) if isinstance(payload, dict) and "ct" in payload else payload
    frame = pd.DataFrame(rows)
    dates = pd.to_datetime(frame["time"]).dt.normalize()
    series = pd.Series(pd.to_numeric(frame["price"]).to_numpy(), index=dates)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.loc[(series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))]


def fetch_official_mid60(start: str, end: str) -> pd.Series:
    """Indice prix exact ; source graphique public Euronext, sans ETF proxy."""
    return fetch_euronext_chart("QS0010989117-XPAR", "indices", start, end).rename("^CM100")


def correct_known_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Correction bornée au point erroné identifié, sans filtrage des mouvements extrêmes.

    Idempotente : une clôture différente de la valeur erronée n'est jamais remplacée.
    L'interpolation est une estimation explicitement tracée, pas une clôture officielle.
    """
    corrected = prices.copy()
    changes = []
    day = pd.Timestamp("2023-11-28")
    if "^CM100" in corrected and day in corrected.index:
        original = corrected.at[day, "^CM100"]
        if np.isfinite(original) and np.isclose(original, 14749.16, rtol=0, atol=.005):
            adjacent = corrected["^CM100"].reindex(pd.to_datetime(["2023-11-27", "2023-11-29"]))
            if adjacent.isna().any() or not np.isfinite(adjacent).all() or (adjacent <= 0).any():
                raise ValueError("Correction CAC Mid 60 impossible sans les deux séances adjacentes valides.")
            replacement = float(adjacent.mean())
            corrected.at[day, "^CM100"] = replacement
            changes.append({"ticker": "^CM100", "date": day.date().isoformat(),
                            "original": float(original), "replacement": replacement,
                            "method": "Estimation : interpolation linéaire entre les séances du 27 et du 29/11/2023 ; non certifiée."})
    return corrected, changes


def validate_euronext_replacement(yahoo: pd.Series, replacement: pd.Series,
                                  calendar: pd.DatetimeIndex, overlap_start: str | None = None):
    """Accepter une source complète seulement après contrôle de son échelle sur le chevauchement."""
    replacement = replacement.reindex(calendar)
    if replacement.isna().any() or not np.isfinite(replacement).all() or (replacement <= 0).any():
        raise ValueError("Historique Euronext incomplet ou non positif.")
    overlap = pd.concat([yahoo.reindex(calendar), replacement], axis=1).dropna()
    if overlap_start:
        overlap = overlap.loc[overlap_start:]
    if len(overlap) < 20 or not np.allclose(overlap.iloc[:, 0], overlap.iloc[:, 1], rtol=1e-5, atol=1e-5):
        raise ValueError("Raccord Euronext/Yahoo refusé : chevauchement insuffisant ou échelles différentes.")
    return replacement


def read_mid60_csv(content: str) -> pd.Series:
    """CSV simple Date,Close, ou export Euronext avec son préambule."""
    lines = content.lstrip("\ufeff").splitlines()
    header = next((i for i, line in enumerate(lines) if line.lower().startswith("date,") or line.lower().startswith("date;")), None)
    if header is None:
        raise ValueError("Le CSV doit contenir les colonnes Date et Close.")
    separator = ";" if ";" in lines[header] else ","
    columns = lines[header].replace('"', '').split(separator)
    # Certains exports Euronext ont un séparateur final absent de l'en-tête.
    rows = [line.split(separator)[:len(columns)] for line in lines[header+1:] if line.strip()]
    frame = pd.DataFrame(rows, columns=columns)
    frame.columns = frame.columns.str.strip().str.lower()
    if "close" not in frame:
        raise ValueError("Colonne Close manquante.")
    dates = pd.to_datetime(frame["date"], format="mixed", dayfirst=separator == ";", errors="raise")
    values = pd.to_numeric(frame["close"].str.replace('"', '').str.replace(" ", "").str.replace(",", "."), errors="raise")
    if dates.duplicated().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Dates uniques et cours strictement positifs requis.")
    return pd.Series(values.to_numpy(), index=dates, name="^CM100").sort_index()


def fetch_market_data(universe: pd.DataFrame, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Télécharger Yahoo, puis compléter l'indice Mid 60 exact si nécessaire."""
    end = end or paris_today()
    tickers = universe["ticker"].tolist() + list(BENCHMARKS)
    chunks = []
    for offset in range(0, len(tickers), 20):
        raw = yf.download(tickers[offset:offset+20], start=START, end=end,
                          auto_adjust=False, actions=False, progress=False, threads=4, timeout=25)
        if raw is not None and not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[offset])
            chunks.append(close)
    if not chunks:
        raise ValueError("Yahoo n'a renvoyé aucun historique. Réessayer plus tard.")
    prices = pd.concat(chunks, axis=1).reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index < end)]
    if prices["^FCHI"].notna().sum() < 252:
        raise ValueError("Historique CAC 40 Yahoo insuffisant.")
    calendar = prices["^FCHI"].dropna().index
    notes = []
    source = "Yahoo Finance (^CM100)"
    if prices["^CM100"].reindex(calendar).isna().any():
        try:
            official = fetch_official_mid60(START, end)
            # Source unique : ne pas raccorder une cotation Yahoo intraday à une clôture officielle.
            prices["^CM100"] = official.reindex(prices.index)
            source = "Euronext · graphique public de l'indice prix CAC Mid 60"
            notes.append("Yahoo ^CM100 incomplet : historique exact complété par Euronext, sans proxy.")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            notes.append(f"Complément Euronext indisponible ({type(exc).__name__}) : importer Date,Close.")
    stock_sources = {}
    # Vivendi : la série Yahoo mélange les échelles les 6–9/12/2024.
    # Une source Euronext unique, ajustée de la scission, évite de neutraliser un rendement réel.
    if "VIV.PA" in universe["ticker"].values:
        vivendi = fetch_euronext_chart("FR0000127771-XPAR", "equities", START, end)
        replacement = validate_euronext_replacement(prices["VIV.PA"], vivendi, calendar, "2024-12-16")
        prices["VIV.PA"] = replacement.reindex(prices.index)
        stock_sources["VIV.PA"] = "Euronext · graphique public FR0000127771-XPAR, historique ajusté de la scission du 16/12/2024."
    # Ancienne cotation Amsterdam : Yahoo URW.AS n'est plus téléchargeable.
    # Utiliser l'instrument Paris de même ISIN, vérifié sur tout le chevauchement.
    if "URW.PA" in universe["ticker"].values and prices["URW.PA"].reindex(calendar).isna().any():
        try:
            urw = fetch_euronext_chart("FR0013326246-XPAR", "equities", START, end)
            replacement = validate_euronext_replacement(prices["URW.PA"], urw, calendar)
            prices["URW.PA"] = replacement.reindex(prices.index)
            stock_sources["URW.PA"] = "Euronext · graphique public FR0013326246-XPAR ; échelle validée sur le chevauchement Yahoo."
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            notes.append(f"URW : complément Euronext indisponible ({type(exc).__name__}) ; exclusion si incomplet.")
    prices, corrections = correct_known_prices(prices)
    metadata = {"requested_start": START, "end_exclusive": end,
                "fetched_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                "price_basis": "Close Yahoo sans dividendes réinvestis ; indices prix",
                "mid60_source": source, "notes": notes, "stock_sources": stock_sources,
                "data_quality_version": DATA_QUALITY_VERSION, "data_corrections": corrections}
    return prices, metadata


def prepare_data(prices: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Même calendrier ; exclusions qualité explicites et correction ponctuelle tracée."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates or not prices.columns.is_unique:
        raise ValueError("Dates et colonnes doivent être uniques.")
    prices, _ = correct_known_prices(prices.sort_index())
    if any(symbol not in prices for symbol in BENCHMARKS):
        raise ValueError("Les deux indices exacts sont obligatoires.")
    reference = prices["^FCHI"].dropna()
    if len(reference) < 252 or reference.index[0] > pd.Timestamp(START) + pd.Timedelta(days=7):
        raise ValueError("Le CAC 40 doit couvrir la période demandée depuis janvier 2022.")
    calendar = reference.index
    benchmarks = prices.reindex(calendar)[list(BENCHMARKS)]
    if benchmarks.isna().any().any() or not np.isfinite(benchmarks).all().all() or (benchmarks <= 0).any().any():
        raise ValueError("Historique CAC Mid 60 incomplet sur la période depuis 2022 : actualiser ou importer le CSV exact. Aucune comparaison partielle n'est présentée comme complète.")
    eligible, audit = [], []
    for row in universe.to_dict("records"):
        series = prices.reindex(calendar).get(row["ticker"], pd.Series(index=calendar, dtype=float))
        missing = int(series.isna().sum())
        reason = EXCLUDED_ASSETS.get(row["ticker"])
        # Une ancienne archive/import ne doit pas réintroduire le décrochage Yahoo de Vivendi.
        if row["ticker"] == "VIV.PA":
            discontinuity = series.pct_change(fill_method=None).get(pd.Timestamp("2024-12-09"), 0.)
            if discontinuity < -.5:
                reason = "Exclu : échelles Vivendi incohérentes avant la scission ; reconstruire depuis Euronext."
        valid = reason is None and missing == 0 and np.isfinite(series).all() and (series > 0).all()
        observed = series.dropna()
        audit.append({**row, "eligible": bool(valid), "missing_sessions": missing,
                      "first_quote": str(observed.index.min().date()) if len(observed) else "—",
                      "reason": reason or ("Historique complet" if valid else "Cours absent, non positif ou historique incomplet"),
                      "large_moves": int((series.pct_change(fill_method=None).abs() > .5).sum()),
                      "large_moves_note": LARGE_MOVE_NOTES.get(row["ticker"], "")})
        if valid:
            eligible.append(row["ticker"])
    if len(eligible) < 10:
        raise ValueError(f"Seulement {len(eligible)} actions ont un historique complet ; au moins 10 sont nécessaires.")
    return prices.loc[calendar, eligible], benchmarks, pd.DataFrame(audit)


# %% Mesures financières
def return_metrics(returns: pd.Series, years: float) -> dict:
    """CAGR calendaire, volatilité annualisée et baisse maximale (capital initial inclus)."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or years <= 0 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Rendements finis > -100 %, durée positive et au moins deux observations requis.")
    wealth = np.r_[1., np.cumprod(1. + values)]
    return {"cagr": float(np.expm1(np.log1p(values).sum() / years)),
            "volatility": float(values.std(ddof=1) * np.sqrt(252)),
            "total_return": float(wealth[-1] - 1),
            "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min())}


def calendar_years(index: pd.DatetimeIndex) -> float:
    return (index[-1] - index[0]).days / 365.25


def benchmark_target(benchmarks: pd.DataFrame) -> tuple[float, str, pd.DataFrame]:
    years = calendar_years(benchmarks.index)
    stats = pd.DataFrame({BENCHMARKS.get(symbol, symbol): return_metrics(series.pct_change(fill_method=None).dropna(), years)
                          for symbol, series in benchmarks.items()}).T
    winner = stats["cagr"].idxmax()
    return float(stats.loc[winner, "cagr"]), winner, stats


# %% Paramètres et classement des solutions
@dataclass(frozen=True)
class ACOConfig:
    k: int = 10
    n_ants: int = 40
    n_iterations: int = 80
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = .20
    n_elite: int = 5
    q: float = 1.0
    tau_min: float = .05
    tau_max: float = 5.0
    seed: int = 42

    def validate(self, n_assets: int):
        if not 1 <= self.k <= n_assets or self.n_ants < 1 or self.n_iterations < 1:
            raise ValueError("Cardinalité et budget de recherche invalides.")
        if not 1 <= self.n_elite <= self.n_ants:
            raise ValueError("n_elite doit être compris entre 1 et n_ants.")
        if not 0 < self.rho <= 1 or self.alpha < 0 or self.beta < 0 or self.q <= 0:
            raise ValueError("rho ∈ ]0,1], alpha/beta ≥ 0 et Q > 0 requis.")
        if not 0 < self.tau_min <= 1 <= self.tau_max:
            raise ValueError("0 < tau_min ≤ tau_initial = 1 ≤ tau_max requis.")
        if not np.isfinite([self.alpha, self.beta, self.rho, self.q, self.tau_min, self.tau_max]).all():
            raise ValueError("Paramètres finis requis.")


def solution_rank(cagr: float, volatility: float, target: float) -> tuple:
    """Une solution admissible passe toujours avant une solution sous le seuil."""
    if cagr + 1e-12 >= target:
        return (0, volatility, -cagr)
    return (1, target - cagr, volatility)


# %% Algorithme des fourmis
def optimize_aco(returns: pd.DataFrame, target: float, years: float,
                 config: ACOConfig = ACOConfig(), progress=None) -> dict:
    """Chercher un sous-ensemble : les poids restent exactement égaux à 1/k.

    Chaque fourmi tire k titres sans remise. L'heuristique privilégie les titres
    moins volatils ; l'évaluation finale tient compte de leurs covariances.
    Le CAGR composé sert à la contrainte et la volatilité sert à l'objectif.
    """
    config.validate(returns.shape[1])
    matrix = returns.to_numpy(dtype=float)
    if len(matrix) < 2 or not np.isfinite(matrix).all() or (matrix <= -1).any() or not math.isfinite(target) or years <= 0:
        raise ValueError("Rendements, objectif ou durée invalides.")
    rng = np.random.default_rng(config.seed)
    n = matrix.shape[1]
    tau = np.ones(n)  # Une trace par action ; l'ordre des titres n'a pas d'importance.
    asset_vol = matrix.std(axis=0, ddof=1) * np.sqrt(252)
    eta = 1. / np.maximum(asset_vol, 1e-8)
    eta /= np.median(eta)  # Normalisation sans effet sur les probabilités.
    best, best_rank = None, None
    cache, history = {}, []
    for iteration in range(1, config.n_iterations + 1):
        log_attractiveness = config.alpha * np.log(tau) + config.beta * np.log(eta)
        attractiveness = np.exp(log_attractiveness - log_attractiveness.max())
        attractiveness = np.maximum(attractiveness, np.finfo(float).tiny)
        probabilities = attractiveness / attractiveness.sum()
        colony = []
        for _ in range(config.n_ants):
            selection = tuple(sorted(rng.choice(n, size=config.k, replace=False, p=probabilities).tolist()))
            if selection not in cache:
                daily = matrix[:, selection].mean(axis=1)  # 10 % par titre chaque jour.
                cache[selection] = return_metrics(pd.Series(daily), years)
            metrics = cache[selection]
            rank = solution_rank(metrics["cagr"], metrics["volatility"], target)
            colony.append((rank, selection, metrics))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = (selection, metrics.copy())
        colony.sort(key=lambda candidate: candidate[0])
        # Évaporation : rho = 0.20 signifie que 20 % de la trace disparaît.
        tau *= 1. - config.rho
        elites = colony[:config.n_elite]
        # Si une solution admissible existe, aucune solution non admissible ne dépose.
        if best_rank[0] == 0:
            elites = [entry for entry in elites if entry[0][0] == 0]
        # Le meilleur global renforce aussi la mémoire de la recherche.
        # Choix délibéré : s'il figure parmi les élites, il reçoit deux dépôts.
        deposits = [(best_rank, best[0], best[1])] + elites
        normalizer = len(deposits) * (len(deposits) + 1) / 2
        for rank_number, (_, selection, _) in enumerate(deposits):
            tau[list(selection)] += config.q * (len(deposits) - rank_number) / normalizer
        tau = np.clip(tau, config.tau_min, config.tau_max)
        history.append({"iteration": iteration, **best[1], "feasible": best_rank[0] == 0,
                        "shortfall": max(0., target - best[1]["cagr"]),
                        "feasible_ants": sum(entry[0][0] == 0 for entry in colony),
                        "unique_evaluated": len(cache)})
        if progress:
            progress(iteration / config.n_iterations)
    selection, metrics = best
    return {"tickers": returns.columns[list(selection)].tolist(), "metrics": metrics,
            "feasible": bool(best_rank[0] == 0), "target": target,
            "history": pd.DataFrame(history),
            "pheromones": pd.Series(tau, index=returns.columns).sort_values(ascending=False),
            "config": asdict(config), "unique_evaluated": len(cache)}


def wealth_curves(prices: pd.DataFrame, benchmarks: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    daily = prices[tickers].pct_change(fill_method=None).iloc[1:].mean(axis=1)
    portfolio = pd.concat([pd.Series([100.], index=prices.index[:1]), 100. * (1. + daily).cumprod()])
    curves = 100. * benchmarks / benchmarks.iloc[0]
    curves = curves.rename(columns=BENCHMARKS)
    curves.insert(0, "Portefeuille ACO · rétrospectif", portfolio)
    return curves
