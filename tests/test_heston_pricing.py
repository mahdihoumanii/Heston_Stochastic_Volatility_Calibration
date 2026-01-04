import numpy as np
import warnings

from src.models.black_scholes import bs_call_price
from src.models.heston import heston_call_price, heston_put_price
from src.utils.helpers import HestonParams


def test_heston_near_black_scholes_constant_vol():
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.01, 0.0
    theta = 0.04
    params = HestonParams(kappa=5.0, theta=theta, sigma=1e-4, rho=0.0, v0=theta)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
        heston_price = heston_call_price(S0, K, T, r, q, params, integration_limit=120)
    bs_price = bs_call_price(S0, K, T, r, q, np.sqrt(theta))
    assert np.isclose(heston_price, bs_price, atol=5e-3)


def test_put_call_parity():
    S0, K, T, r, q = 100.0, 110.0, 0.5, 0.02, 0.0
    params = HestonParams(kappa=2.0, theta=0.03, sigma=0.4, rho=-0.6, v0=0.03)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
        call = heston_call_price(S0, K, T, r, q, params, integration_limit=120)
        put = heston_put_price(S0, K, T, r, q, params, integration_limit=120)
    parity = call - put
    forward = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(parity, forward, atol=1e-3)


def test_heston_matches_bs_in_constant_vol_limit():
    S0, T, r, q = 100.0, 1.0, 0.01, 0.0
    params = HestonParams(kappa=10.0, theta=0.04, sigma=1e-6, rho=0.0, v0=0.04)
    ks = [80.0, 100.0, 120.0]
    for K in ks:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
            h_price = heston_call_price(S0, K, T, r, q, params, integration_limit=200)
        bs_price = bs_call_price(S0, K, T, r, q, np.sqrt(params.theta))
        assert np.isclose(h_price, bs_price, atol=0.1)


def test_heston_atm_reasonable_level():
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.4, rho=-0.7, v0=0.04)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
        price = heston_call_price(S0, K, T, r, q, params, integration_limit=200)
    assert 1.0 < price < 20.0


def test_heston_no_arbitrage_bounds():
    S0, r, q = 100.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.4, rho=-0.3, v0=0.04)
    strikes = [80.0, 100.0, 120.0]
    maturities = [0.5, 1.0]
    for T in maturities:
        for K in strikes:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
                price = heston_call_price(S0, K, T, r, q, params, integration_limit=200)
            lower = max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T))
            upper = S0 * np.exp(-q * T)
            assert lower - 1e-8 <= price <= upper + 1e-8
