"""Synthetic option chain generator and placeholder for real data loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.models.heston import heston_call_price, heston_put_price
from src.utils.helpers import HestonParams, ensure_array, set_seed


@dataclass
class OptionRecord:
    strike: float
    maturity: float
    option_type: str
    price: float


def synthetic_option_chain(
    strikes: Iterable[float],
    maturities: Iterable[float],
    params: HestonParams,
    S0: float = 100.0,
    r: float = 0.02,
    q: float = 0.0,
    noise: float = 0.005,
    seed: int | None = 42,
    include_puts: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic option chain using Heston prices plus small noise."""
    set_seed(seed)
    strikes_arr = ensure_array(strikes)
    maturities_arr = ensure_array(maturities)

    data: List[OptionRecord] = []
    for T in maturities_arr:
        for K in strikes_arr:
            call_price = heston_call_price(S0, K, T, r, q, params)
            noisy_call = max(call_price * (1 + noise * np.random.randn()), 1e-5)
            data.append(OptionRecord(strike=K, maturity=T, option_type="call", price=noisy_call))
            if include_puts:
                put_price = heston_put_price(S0, K, T, r, q, params)
                noisy_put = max(put_price * (1 + noise * np.random.randn()), 1e-5)
                data.append(OptionRecord(strike=K, maturity=T, option_type="put", price=noisy_put))

    df = pd.DataFrame([r.__dict__ for r in data])
    return df


def load_yahoo_chain(*args, **kwargs) -> pd.DataFrame:  # pragma: no cover - stub
    """Placeholder for real data loading using yfinance."""
    raise NotImplementedError("Real market data loading is not implemented in this template.")


def default_true_params() -> HestonParams:
    """Default true parameters used throughout the project."""
    return HestonParams(kappa=1.5, theta=0.04, sigma=0.4, rho=-0.7, v0=0.04)


def default_grid() -> tuple[np.ndarray, np.ndarray]:
    """Default strikes and maturities grid."""
    strikes = np.arange(60, 145, 5)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    return strikes, maturities
