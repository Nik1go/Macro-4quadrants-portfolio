"""Pricing model module for Polymarket binary markets.

This module provides a production-ready wrapper around the Black-Scholes
cash-or-nothing digital option model with robust handling of edge cases,
date parsing utilities, and implied volatility inversion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from .models.black_scholes import BlackScholesDigitalOption
    from .utils.logger import Logger
except ImportError:  # pragma: no cover
    from models.black_scholes import BlackScholesDigitalOption
    from utils.logger import Logger

logger = Logger().logger


class PricingModel:
    """Pricing model for binary options using Black-Scholes digital option logic."""

    MIN_PROBABILITY: float = 1e-4
    MAX_PROBABILITY: float = 1 - 1e-4
    MIN_SIGMA: float = 1e-6
    MAX_SIGMA: float = 5.0

    def __init__(self) -> None:
        """Initialize the pricing model with the digital option engine."""
        self.model = BlackScholesDigitalOption()

    @staticmethod
    def _to_float(value: Union[int, float, np.floating], default: float = 0.0) -> float:
        """Convert a value to float while handling invalid values safely."""
        try:
            converted = float(value)
            if np.isnan(converted) or np.isinf(converted):
                return default
            return converted
        except (TypeError, ValueError):
            return default

    def _sanitize_probability(self, prob: float) -> float:
        """Clamp probabilities to a safe interval to avoid numerical instabilities."""
        clean_prob = self._to_float(prob, default=0.5)
        return float(np.clip(clean_prob, self.MIN_PROBABILITY, self.MAX_PROBABILITY))

    def _terminal_probability(self, s: float, k: float) -> float:
        """Return deterministic terminal probability when uncertainty collapses."""
        if s > k:
            return self.MAX_PROBABILITY
        if s < k:
            return self.MIN_PROBABILITY
        return 0.5

    def binary_call_option_price(self, s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate the theoretical probability of a digital call option."""
        s_val = max(self._to_float(s, default=0.0), 0.0)
        k_val = max(self._to_float(k, default=0.0), 0.0)
        t_val = self._to_float(t, default=0.0)
        r_val = self._to_float(r, default=0.0)
        sigma_val = max(self._to_float(sigma, default=0.0), 0.0)

        if s_val <= 0 or k_val <= 0:
            logger.warning("Invalid spot/strike encountered (s=%s, k=%s). Returning neutral probability.", s, k)
            return 0.5

        if t_val <= 0 or sigma_val <= 0:
            return self._terminal_probability(s_val, k_val)

        try:
            prob = self.model.digital_call_price(s_val, k_val, t_val, r_val, sigma_val)
            return self._sanitize_probability(prob)
        except Exception as exc:  # pragma: no cover
            logger.error("Call probability calculation failed: %s", exc)
            return self._terminal_probability(s_val, k_val)

    def binary_put_option_price(self, s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate the theoretical probability of a digital put option."""
        call_prob = self.binary_call_option_price(s=s, k=k, t=t, r=r, sigma=sigma)
        return self._sanitize_probability(1.0 - call_prob)

    def calculate_mispricing(self, theoretical_prob: float, polymarket_price: float) -> float:
        """Compute mispricing spread between model probability and market price."""
        theo = self._sanitize_probability(theoretical_prob)
        market = self._sanitize_probability(polymarket_price)
        return theo - market

    def calculate_time_to_maturity(
        self,
        maturity_date: Union[str, datetime],
        current_date: Optional[datetime] = None,
    ) -> float:
        """Calculate time-to-maturity in years with robust date handling."""
        now = current_date or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        try:
            if isinstance(maturity_date, str):
                end_dt = datetime.fromisoformat(maturity_date.replace("Z", "+00:00"))
            elif isinstance(maturity_date, datetime):
                end_dt = maturity_date
            else:
                logger.warning("Unsupported maturity_date type: %s", type(maturity_date))
                return 0.0

            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            delta_seconds = (end_dt - now).total_seconds()
            return max(delta_seconds, 0.0) / (365.25 * 24 * 3600)
        except Exception as exc:
            logger.error("Failed to calculate time to maturity for %s: %s", maturity_date, exc)
            return 0.0

    def calculate_implied_volatility(
        self,
        market_price: float,
        s: float,
        k: float,
        t: float,
        r: float,
        option_type: str = "call",
    ) -> Optional[float]:
        """Infer implied volatility from market price using bounded optimization."""
        target = self._sanitize_probability(market_price)
        if t <= 0:
            logger.warning("Implied volatility requested with non-positive maturity (t=%s).", t)
            return None

        side = option_type.lower().strip()
        if side not in {"call", "put"}:
            logger.warning("Unsupported option_type=%s. Expected 'call' or 'put'.", option_type)
            return None

        def objective(vol: float) -> float:
            sigma = max(self.MIN_SIGMA, float(vol))
            if side == "call":
                model_price = self.binary_call_option_price(s=s, k=k, t=t, r=r, sigma=sigma)
            else:
                model_price = self.binary_put_option_price(s=s, k=k, t=t, r=r, sigma=sigma)
            return (model_price - target) ** 2

        try:
            result = minimize_scalar(
                objective,
                bounds=(self.MIN_SIGMA, self.MAX_SIGMA),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 200},
            )
            if not result.success:
                logger.warning("Implied volatility optimization failed: %s", result.message)
                return None
            return float(np.clip(result.x, self.MIN_SIGMA, self.MAX_SIGMA))
        except Exception as exc:
            logger.error("Implied volatility calculation failed: %s", exc)
            return None
