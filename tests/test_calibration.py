import numpy as np
import warnings

from src.calibration.calibrate import calibrate
from src.data.option_chain import default_true_params, synthetic_option_chain
from src.calibration.calibrate import _save_output
from src.utils.helpers import HestonParams
import json
import tempfile


# Calibration on sparse price data is ill-posed; we target implied vols for better identifiability
# and only check economically meaningful ranges instead of exact recovery.
def test_calibration_recovers_parameters():
    true_params = default_true_params()
    strikes = [80, 100, 120]
    maturities = [0.5, 1.0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Feller condition violated:*", category=RuntimeWarning)
        market_df = synthetic_option_chain(
            strikes=strikes,
            maturities=maturities,
            params=true_params,
            noise=0.001,
            seed=7,
            include_puts=False,
        )
        result = calibrate(market_df, S0=100.0, r=0.02, q=0.0, target="iv", verbose=False)
    params_hat = HestonParams(**result["params"])
    theta_hat = params_hat.theta
    rho_hat = params_hat.rho
    v0_hat = params_hat.v0

    assert 0.01 <= theta_hat <= 0.1
    assert np.sign(rho_hat) == np.sign(true_params.rho)
    assert abs(rho_hat - true_params.rho) < 0.3
    assert abs(v0_hat - true_params.theta) < 0.02
    assert result["rmse"] < 0.05


def test_calibration_save_uses_fitted_params(tmp_path):
    true_params = default_true_params()
    strikes = [80, 100, 120]
    maturities = [0.5, 1.0]
    market_df = synthetic_option_chain(
        strikes=strikes,
        maturities=maturities,
        params=true_params,
        noise=0.01,
        seed=11,
        include_puts=False,
    )
    result = calibrate(market_df, S0=100.0, r=0.02, q=0.0, target="price", verbose=False)
    out_path = tmp_path / "calibration_result.json"
    meta = {"seed": 11, "target": "price", "noise": 0.01}
    _save_output(result, str(out_path), metadata=meta)
    with open(out_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    # Saved params match fitted params
    for k, v in result["params"].items():
        assert k in saved["params"]
        assert abs(saved["params"][k] - v) < 1e-8
    # With noise, at least one parameter should differ from the exact true value
    diffs = [abs(saved["params"][k] - true_params.as_dict()[k]) for k in saved["params"]]
    assert any(d > 1e-4 for d in diffs)
    assert saved["metadata"]["seed"] == 11
