"""Heston stochastic volatility model pricing utilities."""

from __future__ import annotations

import cmath
import warnings
from typing import Iterable

import numpy as np
from scipy import integrate

from src.utils.helpers import HestonParams, feller_condition


def _validate_params(params: HestonParams) -> None:
    ok, msg = params.validate()
    if not ok:
        raise ValueError(f"Invalid parameters: {msg}")


def _char_function(u: complex, params: HestonParams, T: float, r: float, q: float, S0: float) -> complex:
    """
    Gatheral's Little Heston Trap characteristic function for log(S_T).
    """
    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    x0 = np.log(S0)
    a = kappa * theta
    b = kappa
    d = cmath.sqrt((rho * sigma * 1j * u - b) ** 2 + sigma**2 * (1j * u + u * u))
    g = (b - rho * sigma * 1j * u - d) / (b - rho * sigma * 1j * u + d)
    exp_neg_dT = cmath.exp(-d * T)
    one_minus_g = 1 - g
    G = 1 - g * exp_neg_dT
    # Clamp to avoid division by very small complex numbers when sigma -> 0
    if abs(one_minus_g) < 1e-14:
        one_minus_g = 1e-14 + 0j
    if abs(G) < 1e-14:
        G = 1e-14 + 0j

    C = (r - q) * 1j * u * T + (a / sigma**2) * ((b - rho * sigma * 1j * u - d) * T - 2 * np.log(G / one_minus_g))
    D = ((b - rho * sigma * 1j * u - d) / sigma**2) * ((1 - exp_neg_dT) / G)
    return np.exp(C + D * v0 + 1j * u * x0)


def _probabilities(
    S0: float, K: float, T: float, r: float, q: float, params: HestonParams, integration_limit: float
) -> tuple[float, float]:
    """Risk-neutral probabilities P1 and P2 using the original Heston formulation."""
    _validate_params(params)
    lnK = np.log(K)
    phi_minus_i = _char_function(-1j, params, T, r, q, S0)
    if abs(phi_minus_i) < 1e-14:
        phi_minus_i = 1e-14 + 0j

    def integrand_p1(u: float) -> float:
        u_complex = u - 1e-10
        phi_shift = _char_function(u_complex - 1j, params, T, r, q, S0)
        numerator = np.exp(-1j * u_complex * lnK) * phi_shift
        return np.real(numerator / (1j * u_complex * phi_minus_i))

    def integrand_p2(u: float) -> float:
        u_complex = u - 1e-10
        phi_val = _char_function(u_complex, params, T, r, q, S0)
        numerator = np.exp(-1j * u_complex * lnK) * phi_val
        return np.real(numerator / (1j * u_complex))

    integral_p1, _ = integrate.quad(integrand_p1, 0.0, integration_limit, limit=500)
    integral_p2, _ = integrate.quad(integrand_p2, 0.0, integration_limit, limit=500)
    P1 = 0.5 + integral_p1 / np.pi
    P2 = 0.5 + integral_p2 / np.pi
    return P1, P2


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    integration_limit: float = 150.0,
) -> float:
    """Price a European call option under Heston."""
    _validate_params(params)
    if T <= 0:
        return max(S0 - K, 0.0)
    # When volatility of volatility is extremely small and v0 ~ theta, fall back to BS
    if params.sigma < 1e-3 and abs(params.v0 - params.theta) < 1e-6:
        from src.models.black_scholes import bs_call_price

        return float(bs_call_price(S0, K, T, r, q, np.sqrt(params.theta)))
    if not feller_condition(params.kappa, params.theta, params.sigma):
        warnings.warn("Feller condition violated: 2*kappa*theta < sigma^2", RuntimeWarning)

    P1, P2 = _probabilities(S0, K, T, r, q, params, integration_limit=integration_limit)
    return float(S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)


def heston_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    integration_limit: float = 150.0,
) -> float:
    """Price a European put via put-call parity."""
    call_price = heston_call_price(S0, K, T, r, q, params, integration_limit)
    return float(call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T))


def heston_price_vectorized(
    S0: float,
    strikes: Iterable[float],
    maturities: Iterable[float],
    r: float,
    q: float,
    params: HestonParams,
    option_type: str = "call",
) -> np.ndarray:
    """Vectorized pricing for a grid of strikes and maturities."""
    strikes_arr = np.asarray(list(strikes), dtype=float)
    maturities_arr = np.asarray(list(maturities), dtype=float)
    prices = np.zeros((len(maturities_arr), len(strikes_arr)))
    pricer = heston_call_price if option_type.lower() == "call" else heston_put_price
    for i, T in enumerate(maturities_arr):
        for j, K in enumerate(strikes_arr):
            prices[i, j] = pricer(S0, K, T, r, q, params)
    return prices


def heston_probabilities(
    S0: float, K: float, T: float, r: float, q: float, params: HestonParams, integration_limit: float = 150.0
) -> tuple[float, float]:
    """Public wrapper to inspect P1/P2 probabilities."""
    return _probabilities(S0, K, T, r, q, params, integration_limit)
