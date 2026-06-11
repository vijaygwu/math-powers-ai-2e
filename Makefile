.PHONY: install test outputs

install:
	pip install -e ".[capstone,dev]"

test:
	pytest -q

outputs:
	python examples/regenerate_outputs.py
