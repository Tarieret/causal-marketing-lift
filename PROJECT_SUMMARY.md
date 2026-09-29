# Project Summary — causal-marketing-lift

## Why this project exists
Portfolio project to close the "experimentation gap" flagged across multiple job
applications (marketing/growth analytics roles). Goal: be able to honestly claim
three things —
1. designed an experiment,
2. analyzed it with appropriate statistical methods,
3. distinguished correlation from causal lift.

This is a **learning project**. Claude's role is to explain concepts and guide,
not to write the analysis code — the user writes it themselves, with concepts
explained first and practice second (unlike the pure practice-first pandas drills).

## Central question
Did a marketing campaign cause incremental conversions, or would they have
happened anyway? Deliverable centers on a visual contrasting **attributed**
(exposure-based) conversions vs. the **causal/incremental** estimate.

## Scope (tiered, in priority order)

### Tier 2 — simulated data with known ground truth (primary/highest priority)
Per the reframed brief: simulation is what makes the project *rigorous* rather
than descriptive, and most directly answers "experimentation frameworks" /
"incrementality" language in JDs — because ground truth lets you *prove* a
method recovers the right answer, not just that it ran.
- Documented data-generating process: known true lift, a confounder affecting
  both exposure and conversion, non-random exposure (selection bias).
- Estimand: **ATT** (Average Treatment effect on the Treated) — matches "did
  the campaign work on the people it reached," natural target for PSM.
- Estimators compared against true effect: naive exposed-vs-unexposed,
  holdout/randomized benchmark, propensity-based (PSM and/or IPW).
- Many replications → bias, RMSE, CI coverage. Vary confounding strength.
- Diagnostics: standardized mean differences (pre/post adjustment), propensity
  overlap plot, placebo/negative-control outcome test.
- **Power analysis up front** (explicitly flagged as the piece most portfolio
  projects skip): given baseline rate + minimum detectable effect, what sample
  size is needed? Given a fixed n, what's the smallest lift detectable?
- All simulated data explicitly labeled synthetic everywhere (code, figures, README).

### Tier 1 — real data, holdout analysis (secondary, if time allows)
- Dataset: Kaggle "Marketing A/B Testing" (~588k users, ad vs. PSA groups,
  converted flag, total ads seen, most-ads day/hour). User places CSV in
  `data/raw/` — not downloaded by Claude.
- Checks: schema, missingness, duplicates, group-size ratio, covariate balance.
- Estimand stated explicitly: effect of ad **relative to PSA**, not ad vs. no ad
  (control group still received a PSA, not nothing).
- Two-proportion test + bootstrap CI for the rate difference.
- Attributed conversions = all conversions in ad group (exposure-based, not
  true last-click — no touchpoint paths in this data).
- Incremental conversions = (ad rate − PSA rate) × ad-group size, with CI.
- Optional iROAS only with clearly labeled assumed cost/value inputs.

### Tier 3 — optional, only after Tier 1 and 2 are solid
- Difference-in-differences on a simulated panel with a pre-period.
- Pre-trend / event-study plot, parallel-trends violation sensitivity check.

## Confirmed decisions
| Decision | Choice |
|---|---|
| Tier 2 second method | PSM **and** IPW (both, for cross-check), DiD pushed to optional Tier 3 |
| Tier 2 estimand | ATT |
| Plotting library | matplotlib + seaborn (static PNGs, simplest for README embed) |
| Sim defaults | baseline 5%, true lift +2pp (relative +40%), n=50,000, 500 replications, confounding swept over 3 levels (none/moderate/strong) |
| Project name | `causal-marketing-lift` |
| Repo location | `~/causal-marketing-lift` (local), GitHub remote via `gh` CLI |

## Working style for this project
- **Explain the concept first, then the user practices** (not silent
  practice-first like the pandas/Netflix drills — causal inference is newer
  territory).
- Claude does not write the analysis code. Claude explains, hints, reviews —
  the user writes it.
- No statistic in the README/figures may be asserted without being computed
  from the data or simulation in this repo. If something is ambiguous or
  unverifiable, say so rather than guess.
- Fully reproducible: fixed seeds, pinned `requirements.txt`, single command
  to rerun end to end.
- State assumptions per method (randomization, SUTVA, ignorability/overlap,
  parallel trends for DiD). Report uncertainty on every estimate. Report both
  statistical and practical significance.

## Repo structure (scaffolding only — content not yet written)
```
causal-marketing-lift/
├── README.md              # one-page decision memo (written last)
├── requirements.txt       # pinned versions
├── .gitignore
├── data/
│   ├── raw/                # user-placed Kaggle CSV (gitignored)
│   └── processed/
├── src/                    # importable, tested logic — math lives here
├── notebooks/               # thin — call src/ functions, narrative + figures
├── figures/                 # exported PNGs for README
└── tests/
```

## Status as of 2026-09-29
- Local folder created and renamed to `causal-marketing-lift`; `data/`,
  `figures/`, `notebooks/`, `tests/`, `src/` scaffolding + `.gitignore` +
  `requirements.txt` exist.
- No analysis code written yet (two earlier drafts — `config.py`,
  `power_analysis.py` — were deleted; user will write these themselves).
- GitHub: `gh` CLI install in progress; repo not yet created/connected.
- No Kaggle CSV placed in `data/raw/` yet.
- **Next step**: finish GitHub setup, then start Tier 2 — first concept to
  cover is power analysis (why it matters, what a two-proportion power
  calculation needs), then the user writes `src/power_analysis.py` themselves.
