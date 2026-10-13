# Local equivalents of the CI jobs. `make check` runs all three.
.PHONY: check lint types test test-fast

check: lint types test

lint:
	ruff check .

types:
	mypy

test:
	python3.11 -m coverage run -m pytest && python3.11 -m coverage report

test-fast:
	python3.11 -m pytest -m "not slow"
