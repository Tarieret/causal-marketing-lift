"""Holdout analysis of ad vs PSA conversion (real data).

Estimand: effect of seeing the ad *relative to a PSA* (control was shown a
PSA, not nothing). Assumes random assignment, SUTVA, no differential attrition.
"""
import numpy as np
import pandas as pd
from scipy import stats


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Users, conversions, and conversion rate per test group."""
    out = df.groupby("test_group")["converted"].agg(n="size", conversions="sum", rate="mean")
    return out


def two_prop_ztest(x1: int, n1: int, x0: int, n0: int) -> dict:
    """Pooled two-proportion z-test (two-sided), computed by hand.

    Group 1 = treated (ad), group 0 = control (PSA).
    """
    p1, p0 = x1 / n1, x0 / n0
    p_pool = (x1 + x0) / (n1 + n0)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    # unpooled SE for a Wald CI on the difference
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    diff = p1 - p0
    return {
        "diff": diff,
        "z": z,
        "p_value": p_value,
        "wald_ci": (diff - 1.96 * se, diff + 1.96 * se),
    }


def bootstrap_diff(x1, n1, x0, n0, n_boot=10_000, seed=42):
    """Bootstrap the rate difference.

    For a binary outcome, resampling users with replacement is equivalent to
    drawing the conversion count from Binomial(n, p_hat), which is much faster.
    Returns (draws, (lo, hi)) with a 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    d = rng.binomial(n1, x1 / n1, n_boot) / n1 - rng.binomial(n0, x0 / n0, n_boot) / n0
    return d, tuple(np.percentile(d, [2.5, 97.5]))


def incrementality(rate_ad, rate_psa, n_ad, diff_ci):
    """Attributed vs incremental conversions in the ad group.

    Attributed  = all conversions in the ad group (exposure-based).
    Incremental = (ad rate - PSA rate) * n_ad.
    """
    attributed = rate_ad * n_ad
    incremental = (rate_ad - rate_psa) * n_ad
    return {
        "attributed": attributed,
        "incremental": incremental,
        "incremental_ci": (diff_ci[0] * n_ad, diff_ci[1] * n_ad),
        "incremental_share": incremental / attributed,
    }


def balance_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Within-group distribution of a categorical/discrete covariate."""
    return pd.crosstab(df[col], df["test_group"], normalize="columns")


def chi2_balance(df: pd.DataFrame, col: str) -> dict:
    """Chi-square test of independence between covariate and group."""
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df[col], df["test_group"]))
    return {"chi2": chi2, "p_value": p, "dof": dof}
