"""Estimators of the ATT (effect on the exposed). All take arrays y, t (0/1), x."""
import numpy as np


def naive(y, t):
    """Exposed minus unexposed mean outcome. Biased under confounding."""
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return est, se


def fit_propensity(x, t, n_iter=25):
    """Logistic regression of t on x (intercept + slope) by Newton-Raphson.
    Fast single-covariate version; returns fitted P(T=1|x)."""
    b0 = b1 = 0.0
    x2 = x * x
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(b0 + b1 * x)))
        r, w = t - p, p * (1 - p)
        g0, g1 = r.sum(), (r * x).sum()
        h00, h01, h11 = w.sum(), (w * x).sum(), (w * x2).sum()
        det = h00 * h11 - h01 * h01
        s0, s1 = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        b0, b1 = b0 + s0, b1 + s1
        if max(abs(s0), abs(s1)) < 1e-8:
            break
    return 1 / (1 + np.exp(-(b0 + b1 * x)))


def ipw_att(y, t, ps):
    """IPW for the ATT: exposed mean minus control mean reweighted by ps/(1-ps)
    so controls resemble the exposed (Hajek-normalised weights)."""
    w = ps / (1 - ps)
    treated = y[t == 1].mean()
    control = np.sum(w[t == 0] * y[t == 0]) / np.sum(w[t == 0])
    return treated - control


def psm_att(y, t, ps, caliper_sd=0.2):
    """1:1 nearest-neighbour propensity matching WITH replacement on the logit,
    caliper = caliper_sd * sd(logit ps). Returns (ATT among matched treated,
    number of treated dropped, matched control index for each kept treated)."""
    logit = np.log(ps / (1 - ps))
    cal = caliper_sd * logit.std()
    ti, ci = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    order = np.argsort(logit[ci])
    cs = logit[ci][order]
    pos = np.clip(np.searchsorted(cs, logit[ti]), 1, len(cs) - 1)
    left, right = cs[pos - 1], cs[pos]
    pick = np.where(np.abs(logit[ti] - left) <= np.abs(right - logit[ti]), pos - 1, pos)
    match = ci[order][pick]
    keep = np.abs(logit[ti] - logit[match]) <= cal
    att = (y[ti][keep] - y[match][keep]).mean()
    return att, int((~keep).sum()), ti[keep], match[keep]


def bootstrap_ci(estimator, y, t, x, n_boot=100, seed=0, level=0.95):
    """Percentile bootstrap CI; refits the propensity model in every resample
    so uncertainty from estimating the propensity is included."""
    rng = np.random.default_rng(seed)
    n = len(y)
    est = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        est[b] = estimator(y[i], t[i], x[i])
    a = (1 - level) / 2
    return tuple(np.quantile(est, [a, 1 - a]))


def ipw_estimator(y, t, x):
    return ipw_att(y, t, fit_propensity(x, t))


def psm_estimator(y, t, x):
    return psm_att(y, t, fit_propensity(x, t))[0]
