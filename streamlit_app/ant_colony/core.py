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


def fetch_official_mid60(start: str, end: str) -> pd.Series:
    """Historique de l'indice prix CAC Mid 60 exact (aucun ETF de substitution)."""
    with requests.Session() as session:
        page = session.get(MID60_PAGE, timeout=40)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        node = soup.select_one('script[data-drupal-selector="drupal-settings-json"]')
        if node is None:
            raise ValueError("Format Euronext modifié : importer un CSV Date,Close de l'indice exact.")
        settings = json.loads(node.string)
        response = session.get(MID60_CHART, timeout=40)
        response.raise_for_status()
        payload = response.json()
        rows = decode_public_euronext(payload, settings["ajax_secure"]["kye"]) if isinstance(payload, dict) and "ct" in payload else payload
    frame = pd.DataFrame(rows)
    dates = pd.to_datetime(frame["time"]).dt.normalize()
    series = pd.Series(pd.to_numeric(frame["price"]).to_numpy(), index=dates, name="^CM100")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.loc[(series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))]


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
    metadata = {"requested_start": START, "end_exclusive": end,
                "fetched_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
                "price_basis": "Close Yahoo sans dividendes réinvestis ; indices prix",
                "mid60_source": source, "notes": notes}
    return prices, metadata


def prepare_data(prices: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Même calendrier et même période ; aucun remplissage des cours manquants."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates or not prices.columns.is_unique:
        raise ValueError("Dates et colonnes doivent être uniques.")
    prices = prices.sort_index()
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
        valid = missing == 0 and np.isfinite(series).all() and (series > 0).all()
        observed = series.dropna()
        audit.append({**row, "eligible": bool(valid), "missing_sessions": missing,
                      "first_quote": str(observed.index.min().date()) if len(observed) else "—",
                      "reason": "Historique complet" if valid else "Cours absent, non positif ou historique incomplet",
                      "large_moves": int((series.pct_change(fill_method=None).abs() > .5).sum())})
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
