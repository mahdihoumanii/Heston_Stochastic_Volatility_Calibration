"""Utility helpers for reproducible Heston experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from math import exp

from src.models.black_scholes import implied_volatility

def set_seed(seed: int | None = None) -> None:
    """Seed NumPy's RNG for deterministic runs."""
    if seed is None:
        return
    np.random.seed(seed)


def year_fraction(days: float) -> float:
    """Convert a day count to year fraction using ACT/365."""
    return days / 365.0


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def mape(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    denom = np.maximum(np.abs(x_arr), eps)
    return float(np.mean(np.abs(x_arr - y_arr) / denom))


def feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """Return True if the Feller condition 2*kappa*theta >= sigma^2 holds."""
    return 2 * kappa * theta >= sigma**2


@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "v0": self.v0,
        }

    def validate(self) -> Tuple[bool, str | None]:
        """Validate parameter constraints."""
        if self.kappa <= 0:
            return False, "kappa must be > 0"
        if self.theta <= 0:
            return False, "theta must be > 0"
        if self.sigma <= 0:
            return False, "sigma must be > 0"
        if self.v0 <= 0:
            return False, "v0 must be > 0"
        if not (-1 < self.rho < 1):
            return False, "rho must be in (-1, 1)"
        return True, None


def parameter_bounds():
    """Standard calibration bounds."""
    lower = [0.1, 1e-4, 0.05, -0.99, 1e-4]
    upper = [10.0, 1.0, 2.0, 0.99, 1.0]
    return lower, upper


def format_params(params: Dict[str, float]) -> str:
    """Pretty-print parameters in a stable order."""
    keys = ["kappa", "theta", "sigma", "rho", "v0"]
    formatted = []
    for k in keys:
        if k in params:
            formatted.append(f"{k}={params[k]:.4f}")
    return ", ".join(formatted)


def ensure_array(x: Iterable[float]) -> np.ndarray:
    """Convert iterable to a float64 numpy array."""
    return np.asarray(list(x), dtype=float)


def safe_implied_volatility(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str = "call",
    eps: float = 1e-12,
) -> float:
    """
    Robust implied volatility inversion with basic no-arbitrage clipping.

    Heston-derived prices can very rarely drift outside BS no-arb bounds due to
    numerical integration noise. We clip into the open interval (lower, upper)
    before calling the standard implied_volatility solver. Failures yield NaN.
    """
    if not np.isfinite(price):
        return float("nan")

    disc_q = exp(-q * T)
    disc_r = exp(-r * T)
    if option_type == "call":
        lower = max(0.0, S0 * disc_q - K * disc_r)
        upper = S0 * disc_q
    else:
        lower = max(0.0, K * disc_r - S0 * disc_q)
        upper = K * disc_r

    clipped = min(max(price, lower + eps), upper - eps) if upper - eps > lower + eps else price

    try:
        return float(implied_volatility(clipped, S0, K, T, r, q, option_type))
    except Exception:
        return float("nan")


def diagnose_atm_price(params: HestonParams, S0: float = 100.0, r: float = 0.02, q: float = 0.0, T: float = 1.0) -> dict:
    """Diagnostic for Heston vs BS ATM pricing and implied vol."""
    from src.models.heston import heston_call_price, heston_probabilities
    from src.models.black_scholes import bs_call_price, implied_volatility

    K = S0
    h_price = heston_call_price(S0, K, T, r, q, params)
    bs_price = bs_call_price(S0, K, T, r, q, np.sqrt(params.theta))
    P1, P2 = heston_probabilities(S0, K, T, r, q, params)
    try:
        h_iv = implied_volatility(h_price, S0, K, T, r, q, "call")
    except Exception:
        h_iv = float("nan")
    diag = {"heston_price": h_price, "bs_price": bs_price, "heston_iv": h_iv, "P1": P1, "P2": P2}
    print(diag)
    return diag


def check_call_no_arbitrage(prices: np.ndarray, strikes: np.ndarray, T: float, S0: float, r: float, q: float, tol: float = 1e-8) -> None:
    """Ensure call prices respect basic no-arbitrage bounds for a given maturity."""
    lower = np.maximum(0.0, S0 * np.exp(-q * T) - strikes * np.exp(-r * T))
    upper = S0 * np.exp(-q * T)
    if np.any(prices < lower - tol) or np.any(prices > upper + tol):
        raise ValueError("Call prices violate no-arbitrage bounds.")
