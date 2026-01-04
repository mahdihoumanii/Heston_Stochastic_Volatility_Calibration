"""Heston path simulation using Euler full truncation."""

from __future__ import annotations

import numpy as np

from src.utils.helpers import HestonParams, set_seed


def simulate_heston_paths(
    S0: float,
    params: HestonParams,
    r: float,
    q: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston price and variance paths."""
    set_seed(seed)
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0

    v = np.full((n_paths, steps + 1), v0, dtype=float)
    S = np.full((n_paths, steps + 1), S0, dtype=float)
    z1 = np.random.randn(n_paths, steps)
    z2 = np.random.randn(n_paths, steps)
    w1 = z1
    w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    for t in range(steps):
        vt = v[:, t]
        vt_pos = np.maximum(vt, 0.0)
        drift_v = kappa * (theta - vt_pos) * dt
        diff_v = sigma * np.sqrt(vt_pos) * sqrt_dt * w2[:, t]
        v[:, t + 1] = np.maximum(vt + drift_v + diff_v, 0.0)

        vol_step = np.sqrt(np.maximum(v[:, t], 0.0))
        drift_S = (r - q - 0.5 * vol_step**2) * dt
        diff_S = vol_step * sqrt_dt * w1[:, t]
        S[:, t + 1] = S[:, t] * np.exp(drift_S + diff_S)

    return S, v


def variance_diagnostics(v_paths: np.ndarray) -> dict:
    """Basic diagnostics on variance trajectories."""
    return {
        "min": float(np.min(v_paths)),
        "max": float(np.max(v_paths)),
        "mean_terminal": float(np.mean(v_paths[:, -1])),
    }
