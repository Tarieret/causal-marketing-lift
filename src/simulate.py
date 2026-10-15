"""Data-generating process (SYNTHETIC data with known ground truth).

One observed confounder X ~ N(0,1) drives BOTH exposure and conversion:

    exposure:   P(T=1 | X) = sigmoid(a + gamma * X)        gamma = confounding strength
    outcome:    P(Y0=1 | X) = sigmoid(b + beta * X)         baseline conversion
                P(Y1=1 | X) = P(Y0=1 | X) + tau             constant additive lift
    placebo:    P(Yp=1 | X) = sigmoid(b + beta * X)         NO treatment effect

Y0 and Y1 share one uniform draw (monotone coupling), so the true ATT equals
`tau` exactly in expectation. Intercepts a and b are solved so population
exposure is `exposure_rate` and the population baseline conversion is `baseline`.
Because X raises both exposure and conversion, the exposed have a HIGHER
baseline than the population: the naive estimator is biased upward.
"""
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

CONFOUNDING_LEVELS = {"none": 0.0, "moderate": 0.75, "strong": 1.5}

_Z = stats.norm.ppf((np.arange(4001) + 0.5) / 4001)  # fixed quantile grid for N(0,1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_intercept(slope, target_mean):
    """Intercept c such that E[sigmoid(c + slope * X)] = target_mean, X ~ N(0,1)."""
    return optimize.brentq(lambda c: sigmoid(c + slope * _Z).mean() - target_mean, -20, 20)


@dataclass
class SimData:
    x: np.ndarray
    t: np.ndarray       # exposure (1 = exposed)
    y: np.ndarray       # conversion
    y_placebo: np.ndarray  # negative-control outcome (true effect = 0)
    true_att: float


def simulate(n=50_000, baseline=0.05, tau=0.02, gamma=0.75, beta=0.75,
             exposure_rate=0.30, randomized=False, seed=0):
    """Draw one synthetic dataset. `randomized=True` assigns T by coin flip
    (the holdout/randomized benchmark): gamma is ignored."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if randomized:
        t = rng.random(n) < exposure_rate
    else:
        a = solve_intercept(gamma, exposure_rate)
        t = rng.random(n) < sigmoid(a + gamma * x)
    b = solve_intercept(beta, baseline)
    p0 = sigmoid(b + beta * x)
    u = rng.random(n)
    y0 = u < p0
    y1 = u < np.minimum(p0 + tau, 1.0)
    y = np.where(t, y1, y0).astype(float)
    y_placebo = (rng.random(n) < p0).astype(float)
    return SimData(x=x, t=t.astype(float), y=y, y_placebo=y_placebo, true_att=tau)
