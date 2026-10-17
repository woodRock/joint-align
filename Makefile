# Local equivalents of the CI jobs. `make check` runs all three.
.PHONY: install check lint types test test-fast site figures

# Editable install next to an editable alignbench checkout (--no-deps keeps pip from replacing it
# with the commit pinned in pyproject.toml). Fresh environments: pip install -e ".[dev,neural]".
install:
	python3.11 -m pip install --no-deps -e .

check: lint types test

lint:
	ruff check .

types:
	mypy

test:
	python3.11 -m coverage run -m pytest && python3.11 -m coverage report

test-fast:
	python3.11 -m pytest -m "not slow"

site:
	python3.11 scripts/export_site_data.py
	@echo "Serving docs/ at http://localhost:8000"
	cd docs && python3.11 -m http.server 8000

figures:
	python3.11 scripts/04_paper_figures.py
