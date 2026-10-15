"""Balance and overlap diagnostics (SYNTHETIC data)."""
import numpy as np


def smd(x, t, w=None):
    """Standardised mean difference (exposed - control) / pooled raw SD.
    Optional weights `w` apply to the control group (IPW-ATT weights)."""
    x1, x0 = x[t == 1], x[t == 0]
    sd = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    m0 = np.average(x0, weights=None if w is None else w[t == 0])
    return (x1.mean() - m0) / sd


def smd_matched(x, t, treated_idx, control_idx):
    """SMD after matching (controls counted once per match, i.e. with replacement)."""
    sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
    return (x[treated_idx].mean() - x[control_idx].mean()) / sd
