"""Power analysis for a two-proportion test (SYNTHETIC planning numbers).

Inputs: baseline rate, minimum detectable effect (MDE), alpha, power.
Fix any four of {n, effect, alpha, power} and solve for the fifth.
"""
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_power = NormalIndPower()


def required_n_per_arm(baseline, mde, alpha=0.05, power=0.80, ratio=1.0):
    """Users needed in the *control* arm to detect `baseline -> baseline + mde`.

    `ratio` = n_treated / n_control. Two-sided test, Cohen's h effect size.
    """
    h = proportion_effectsize(baseline + mde, baseline)
    return _power.solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio,
                              alternative="two-sided")


def min_detectable_lift(baseline, n_control, alpha=0.05, power=0.80, ratio=1.0):
    """Smallest absolute lift detectable with `n_control` control users.

    Solves for Cohen's h, then inverts h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p0)).
    """
    h = _power.solve_power(nobs1=n_control, alpha=alpha, power=power, ratio=ratio,
                           alternative="two-sided")
    p1 = np.sin(np.arcsin(np.sqrt(baseline)) + h / 2) ** 2
    return p1 - baseline


def power_at(baseline, mde, n_control, alpha=0.05, ratio=1.0):
    """Power to detect `mde` given `n_control` control users."""
    h = proportion_effectsize(baseline + mde, baseline)
    return _power.power(effect_size=h, nobs1=n_control, alpha=alpha, ratio=ratio,
                        alternative="two-sided")
