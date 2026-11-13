.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/lt01_historical_clv.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/lt01_historical_clv.ipynb

reproduce: test notebook
