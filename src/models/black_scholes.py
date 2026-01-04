"""Black-Scholes pricing and implied volatility solver."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_call_price(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = _d1(S0, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    return float(
        S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    )


def bs_put_price(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    d1 = _d1(S0, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    return float(
        K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)
    )


def implied_volatility(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
    bracket: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Recover implied volatility using Brent's method."""
    if T <= 0:
        return 0.0

    # Basic no-arbitrage bounds to avoid root finder failures on extreme inputs
    if option_type == "call":
        lower_bound = max(0.0, S0 * math.exp(-q * T) - K * math.exp(-r * T))
        upper_bound = S0 * math.exp(-q * T)
    else:
        lower_bound = max(0.0, K * math.exp(-r * T) - S0 * math.exp(-q * T))
        upper_bound = K * math.exp(-r * T)

    if price < lower_bound - 1e-8:
        return float("nan")
    if price > upper_bound + 1e-8:
        return float("nan")

    def price_diff(sigma: float) -> float:
        if option_type == "call":
            return bs_call_price(S0, K, T, r, q, sigma) - price
        return bs_put_price(S0, K, T, r, q, sigma) - price

    try:
        implied = brentq(price_diff, *bracket, xtol=tol, maxiter=max_iter)
    except ValueError:
        # fall back to near intrinsic value
        intrinsic = max(0.0, S0 - K) if option_type == "call" else max(0.0, K - S0)
        if price <= intrinsic + 1e-8:
            return 0.0
        return float("nan")
    return float(implied)
