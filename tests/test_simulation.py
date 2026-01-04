import numpy as np

from src.simulation.paths import simulate_heston_paths
from src.utils.helpers import HestonParams


def test_simulation_variance_non_negative():
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04)
    S, v = simulate_heston_paths(
        S0=100.0,
        params=params,
        r=0.02,
        q=0.0,
        T=1.0,
        steps=50,
        n_paths=200,
        seed=123,
    )
    assert np.all(v >= 0.0)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(v))
