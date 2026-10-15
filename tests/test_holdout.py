import numpy as np
from statsmodels.stats.proportion import proportions_ztest

from src.holdout import bootstrap_diff, incrementality, two_prop_ztest


def test_ztest_matches_statsmodels():
    z = two_prop_ztest(14423, 564577, 420, 23524)
    z_sm, p_sm = proportions_ztest([14423, 420], [564577, 23524])
    assert np.isclose(z["z"], z_sm) and np.isclose(z["p_value"], p_sm)


def test_incrementality_known_case():
    out = incrementality(0.02, 0.01, 1000, (0.005, 0.015))
    assert np.isclose(out["attributed"], 20) and np.isclose(out["incremental"], 10)
    assert np.isclose(out["incremental_share"], 0.5)


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded():
    d1, ci1 = bootstrap_diff(200, 10_000, 100, 10_000, seed=1)
    _, ci2 = bootstrap_diff(200, 10_000, 100, 10_000, seed=1)
    assert ci1 == ci2 and ci1[0] < 0.01 < ci1[1]
