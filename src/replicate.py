"""Monte-Carlo replications: bias, RMSE, CI coverage across confounding levels.
Everything here is SYNTHETIC. Seeds are spawned from one base seed => reproducible."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import estimators as est
from .simulate import CONFOUNDING_LEVELS, simulate

N, TAU, BASELINE = 50_000, 0.02, 0.05


def one_rep(args):
    level, seed, n, n_boot = args
    gamma = CONFOUNDING_LEVELS[level]
    d = simulate(n=n, tau=TAU, baseline=BASELINE, gamma=gamma, seed=seed)
    r = simulate(n=n, tau=TAU, baseline=BASELINE, randomized=True, seed=seed + 10**9)
    ps = est.fit_propensity(d.x, d.t)
    psm, dropped, ti, ci = est.psm_att(d.y, d.t, ps)
    w = ps / (1 - ps)
    row = {"level": level, "seed": seed,
           "naive": est.naive(d.y, d.t)[0],
           "randomized": est.naive(r.y, r.t)[0],
           "ipw": est.ipw_att(d.y, d.t, ps), "psm": psm, "psm_dropped": dropped,
           "smd_raw": dg.smd(d.x, d.t), "smd_ipw": dg.smd(d.x, d.t, w),
           "smd_psm": dg.smd_matched(d.x, d.t, ti, ci),
           # placebo outcome: true effect is 0
           "placebo_naive": est.naive(d.y_placebo, d.t)[0],
           "placebo_ipw": est.ipw_att(d.y_placebo, d.t, ps),
           "placebo_psm": est.psm_att(d.y_placebo, d.t, ps)[0]}
    for name, (e, se) in {"naive": est.naive(d.y, d.t), "randomized": est.naive(r.y, r.t)}.items():
        row[f"{name}_lo"], row[f"{name}_hi"] = e - 1.96 * se, e + 1.96 * se
    if n_boot:
        for name, f in {"ipw": est.ipw_estimator, "psm": est.psm_estimator}.items():
            row[f"{name}_lo"], row[f"{name}_hi"] = est.bootstrap_ci(f, d.y, d.t, d.x, n_boot, seed)
    return row


def run(n_reps=500, n_ci_reps=100, n_boot=100, n=N, base_seed=2026, processes=4):
    """Run replications for every confounding level. Bootstrap CIs (slow) are
    computed only for the first `n_ci_reps` replications of each level."""
    seeds = np.random.SeedSequence(base_seed).generate_state(n_reps)
    jobs = [(lvl, int(s), n, n_boot if i < n_ci_reps else 0)
            for lvl in CONFOUNDING_LEVELS for i, s in enumerate(seeds)]
    with Pool(processes) as p:
        return pd.DataFrame(p.map(one_rep, jobs, chunksize=10))


def summarise(df, tau=TAU):
    """Bias, RMSE and CI coverage per level x estimator."""
    rows = []
    for lvl, g in df.groupby("level", sort=False):
        for m in ["naive", "randomized", "ipw", "psm"]:
            ci = g.dropna(subset=[f"{m}_lo"]) if f"{m}_lo" in g else g.iloc[0:0]
            cover = ((ci[f"{m}_lo"] <= tau) & (tau <= ci[f"{m}_hi"])).mean() if len(ci) else np.nan
            rows.append({"level": lvl, "estimator": m, "mean_est": g[m].mean(),
                         "bias": g[m].mean() - tau, "rmse": np.sqrt(((g[m] - tau) ** 2).mean()),
                         "sd": g[m].std(), "coverage": cover, "n_ci_reps": len(ci)})
    return pd.DataFrame(rows)
