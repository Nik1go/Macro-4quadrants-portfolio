"""Ratios annualisés sur les rendements quotidiens, taux explicite en décimal annuel."""
import numpy as np

RISK_FREE_RATE = 0.0
SESSIONS_PER_YEAR = 252


def portfolio_ratios(returns, risk_free_rate=RISK_FREE_RATE):
    """Excès quotidien r - rf/252 ; Sortino sur toutes les séances, cible = rf."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Au moins deux rendements quotidiens finis sont requis.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("Le taux sans risque doit être fini, en décimal annuel.")
    excess = values - risk_free_rate / SESSIONS_PER_YEAR
    annual_mean = float(excess.mean() * SESSIONS_PER_YEAR)
    volatility = float(excess.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR))
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.) ** 2) * SESSIONS_PER_YEAR))
    return {
        "sharpe": annual_mean / volatility if volatility > 0 else float("nan"),
        "sortino": annual_mean / downside if downside > 0 else float("nan"),
    }
