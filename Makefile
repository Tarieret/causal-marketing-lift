# One command to reproduce everything: make all
PY = venv/bin/python

all: test sim notebooks

test:
	venv/bin/pytest -q

sim:  # ~15-25 min; writes data/processed/sim_*.csv
	$(PY) -m src.run_sim 500

notebooks:
	venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_holdout_analysis.ipynb
	venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_simulation.ipynb
