# Executive Summary

## Data
Synthetic option chain with strikes 60-140 (step 5) and maturities 0.25, 0.5, 1.0, 2.0 years. Prices generated from Heston with small Gaussian noise (0.5%).

## Model
Heston stochastic volatility priced via Little Heston Trap formulation. Black--Scholes used as baseline and for implied volatility inversion. Feller condition is reported for diagnostics.

## Calibration
Bounded least-squares using SciPy. Targets option prices by default; implied-vol targeting available. Default bounds: kappa [0.1,10], theta [1e-4,1], sigma [0.05,2], rho [-0.99,0.99], v0 [1e-4,1].

## Results
Synthetic calibration recovers parameters near ground truth on the provided grid. Figures are written to `reports/figures/` by the notebooks; calibration summary saved to `reports/calibration_result.json`.

## Limitations
Numerical integration stability and speed; identifiability of parameters from sparse/noisy chains; synthetic data only (real data loader stub); market frictions and dividends beyond a flat yield are ignored.
