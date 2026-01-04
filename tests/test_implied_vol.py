import numpy as np

from src.models.black_scholes import bs_call_price, implied_volatility


def test_implied_volatility_recovers_sigma():
    S0, K, T, r, q = 100.0, 105.0, 1.0, 0.01, 0.0
    sigma_true = 0.25
    price = bs_call_price(S0, K, T, r, q, sigma_true)
    sigma_est = implied_volatility(price, S0, K, T, r, q, option_type="call")
    assert np.isclose(sigma_est, sigma_true, atol=1e-4)
