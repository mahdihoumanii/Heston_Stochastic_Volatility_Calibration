"""Objective functions for calibrating the Heston model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.black_scholes import implied_volatility
from src.utils.helpers import safe_implied_volatility
from src.models.heston import heston_call_price, heston_put_price
from src.utils.helpers import HestonParams


def array_to_params(x: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(x[0]),
        theta=float(x[1]),
        sigma=float(x[2]),
        rho=float(x[3]),
        v0=float(x[4]),
    )


def _model_price(row, params: HestonParams, S0: float, r: float, q: float) -> float:
    if row.option_type == "call":
        return heston_call_price(S0, row.strike, row.maturity, r, q, params)
    return heston_put_price(S0, row.strike, row.maturity, r, q, params)


def price_residuals(
    x: np.ndarray,
    market_df: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    x0: np.ndarray | None = None,
    reg_weight: float = 0.0,
) -> np.ndarray:
    """Residuals between model and market prices."""
    params = array_to_params(x)
    residuals = []
    for row in market_df.itertuples():
        model_price = _model_price(row, params, S0, r, q)
        scale = max(row.price, 1e-4)
        residuals.append((model_price - row.price) / scale)
    if reg_weight > 0 and x0 is not None:
        residuals.extend((np.sqrt(reg_weight) * (x - x0)).tolist())
    return np.asarray(residuals, dtype=float)


def implied_vol_residuals(
    x: np.ndarray,
    market_df: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    x0: np.ndarray | None = None,
    reg_weight: float = 0.0,
) -> np.ndarray:
    """Residuals between model and market implied volatilities."""
    params = array_to_params(x)
    residuals = []
    for row in market_df.itertuples():
        model_price = _model_price(row, params, S0, r, q)
        model_iv = safe_implied_volatility(model_price, S0, row.strike, row.maturity, r, q, row.option_type)
        market_iv = safe_implied_volatility(row.price, S0, row.strike, row.maturity, r, q, row.option_type)
        if not np.isfinite(model_iv) or not np.isfinite(market_iv):
            residuals.append(0.0)
        else:
            residuals.append(model_iv - market_iv)
    if reg_weight > 0 and x0 is not None:
        residuals.extend((np.sqrt(reg_weight) * (x - x0)).tolist())
    return np.asarray(residuals, dtype=float)
