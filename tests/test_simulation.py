import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src import estimators as est
from src.power_analysis import min_detectable_lift, power_at, required_n_per_arm
from src.simulate import simulate


def test_simulation_is_reproducible():
    a, b = simulate(n=1000, seed=5), simulate(n=1000, seed=5)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.t, b.t)


def test_population_baseline_and_exposure_calibrated():
    d = simulate(n=400_000, gamma=0.75, seed=1)
    assert abs(d.t.mean() - 0.30) < 0.01


def test_power_roundtrip():
    n = required_n_per_arm(0.05, 0.02)
    assert abs(power_at(0.05, 0.02, n) - 0.80) < 1e-6
    assert abs(min_detectable_lift(0.05, n) - 0.02) < 1e-6


def test_power_matches_statsmodels_reference():
    h = proportion_effectsize(0.07, 0.05)
    assert np.isclose(required_n_per_arm(0.05, 0.02), NormalIndPower().solve_power(h, power=0.8, alpha=0.05))


def _mean_est(f, **kw):
    return np.mean([f(simulate(n=50_000, seed=s, **kw)) for s in range(30)])


def test_naive_unbiased_without_confounding():
    m = _mean_est(lambda d: est.naive(d.y, d.t)[0], gamma=0.0)
    assert abs(m - 0.02) < 0.003


def test_naive_biased_up_under_confounding_but_ipw_psm_recover():
    g = dict(gamma=1.5)
    assert _mean_est(lambda d: est.naive(d.y, d.t)[0], **g) > 0.04
    ipw = _mean_est(lambda d: est.ipw_estimator(d.y, d.t, d.x), **g)
    psm = _mean_est(lambda d: est.psm_estimator(d.y, d.t, d.x), **g)
    assert abs(ipw - 0.02) < 0.006 and abs(psm - 0.02) < 0.006


def test_zero_true_effect_centres_on_zero():
    m = _mean_est(lambda d: est.ipw_estimator(d.y, d.t, d.x), gamma=0.75, tau=0.0)
    assert abs(m) < 0.004


def test_placebo_outcome_has_zero_effect_after_adjustment():
    m = _mean_est(lambda d: est.ipw_estimator(d.y_placebo, d.t, d.x), gamma=1.5)
    assert abs(m) < 0.004
