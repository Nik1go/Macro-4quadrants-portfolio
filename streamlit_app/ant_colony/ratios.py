"""Ratios annualisés sur les rendements quotidiens, taux et cible fixés à 0 %."""
import numpy as np


def portfolio_ratios(returns):
    """Sortino : semi-déviation calculée sur toutes les séances, pertes seules au numérateur quadratique."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Au moins deux rendements quotidiens finis sont requis.")
    annual_mean = float(values.mean() * 252)
    volatility = float(values.std(ddof=1) * np.sqrt(252))
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.) ** 2) * 252))
    return {
        "sharpe": annual_mean - 0.02 / volatility if volatility > 0 else float("nan"),
        "sortino": annual_mean - 0.02 / downside if downside > 0 else float("nan"),
    }
