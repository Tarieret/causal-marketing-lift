"""Run the simulation replications and save results (SYNTHETIC data).
Usage: python -m src.run_sim"""
import sys
from pathlib import Path

import pandas as pd

from . import replicate

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    df = replicate.run(n_reps=n_reps, n_ci_reps=100)
    df.to_csv(OUT / "sim_replications.csv", index=False)
    replicate.summarise(df).to_csv(OUT / "sim_summary.csv", index=False)
    print("done", len(df))
