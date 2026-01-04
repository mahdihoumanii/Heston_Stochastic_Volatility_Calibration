"""Calibration routines for the Heston model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.calibration.objective import array_to_params, implied_vol_residuals, price_residuals
from src.data.option_chain import default_grid, default_true_params, synthetic_option_chain
from src.utils import helpers


def calibrate(
    market_df: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    target: str = "price",
    x0: np.ndarray | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    verbose: bool = False,
    reg_weight: float = 0.1,
) -> Dict:
    """Calibrate Heston parameters to a market option chain."""
    residual_fn: Callable[[np.ndarray, pd.DataFrame, float, float, float, np.ndarray | None, float], np.ndarray]
    if target == "iv":
        residual_fn = implied_vol_residuals
    else:
        residual_fn = price_residuals

    if x0 is None:
        true_params = default_true_params()
        x0 = np.array([true_params.kappa, true_params.theta, true_params.sigma, true_params.rho, true_params.v0])

    if bounds is None:
        bounds = helpers.parameter_bounds()

    result = least_squares(
        residual_fn,
        x0,
        args=(market_df, S0, r, q, x0, reg_weight),
        bounds=bounds,
        xtol=1e-8,
        ftol=1e-8,
        loss="soft_l1",
    )
    params_hat = array_to_params(result.x)
    residuals = residual_fn(result.x, market_df, S0, r, q, x0, reg_weight)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    if verbose:
        print(f"Calibrated parameters: {helpers.format_params(params_hat.as_dict())}")
        print(f"Objective norm: {result.cost:.6f}")
        print(f"Feller condition satisfied: {helpers.feller_condition(params_hat.kappa, params_hat.theta, params_hat.sigma)}")

    return {
        "params": params_hat.as_dict(),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message,
        "feller": helpers.feller_condition(params_hat.kappa, params_hat.theta, params_hat.sigma),
        "nfev": result.nfev,
        "rmse": rmse,
    }


def _save_output(payload: Dict, path: str, metadata: Dict | None = None) -> None:
    """Persist calibration result with metadata and plain floats."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_clean = {k: float(v) for k, v in payload.get("params", {}).items()}
    output = {
        "params": params_clean,
        "cost": float(payload.get("cost", 0.0)),
        "success": bool(payload.get("success", False)),
        "message": payload.get("message"),
        "feller": bool(payload.get("feller", False)),
        "nfev": int(payload.get("nfev", 0)),
        "rmse": float(payload.get("rmse", 0.0)),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if metadata:
        output["metadata"] = metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the Heston model.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic option chain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--target", type=str, choices=["price", "iv"], default="price", help="Calibration target")
    parser.add_argument("--output", type=str, default="reports/calibration_result.json", help="Where to save calibration result")
    args = parser.parse_args()

    helpers.set_seed(args.seed)
    if args.synthetic:
        strikes, maturities = default_grid()
        noise = 0.005
        market_df = synthetic_option_chain(strikes, maturities, default_true_params(), noise=noise, seed=args.seed)
        S0, r, q = 100.0, 0.02, 0.0
    else:
        raise ValueError("Only synthetic data is supported in this template.")

    result = calibrate(market_df, S0, r, q, target=args.target, verbose=True)
    params_to_write = {k: float(v) for k, v in result["params"].items()}
    print("calibrate() returned:", result["params"])
    print("writing JSON params:", params_to_write)
    meta = {"seed": args.seed, "target": args.target, "synthetic": args.synthetic, "noise": noise}
    _save_output(result, args.output, metadata=meta)


if __name__ == "__main__":
    main()
