PYTHON ?= python3.14
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install lint format test data explore features benchmark evaluate mlflow cost clean

install: $(BIN)/python
	$(BIN)/pip install -r requirements.txt -e .
	$(BIN)/pre-commit install

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

data:
	$(BIN)/python -m fraudops.simulate

explore:
	$(BIN)/python -m fraudops.explore

features:
	$(BIN)/python -m fraudops.features

benchmark:
	$(BIN)/python -m fraudops.spark_features

evaluate:
	$(BIN)/python -m fraudops.evaluate

mlflow:
	$(BIN)/mlflow ui

cost:
	$(BIN)/python -m fraudops.cost

clean:
	rm -rf .pytest_cache .ruff_cache
