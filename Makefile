.PHONY: help setup data train benchmark test lint run-api run-ui docker-up docker-down clean

PYTHON ?= python

help:
	@echo "GraphRec - Explainable Hybrid Recommender Platform"
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies in editable mode"
	@echo "  make data       - Download and process MovieLens 1M with KG enrichment"
	@echo "  make train      - Train all baseline and neural recommendation models"
	@echo "  make benchmark  - Run quantitative evaluation across all models"
	@echo "  make test       - Run pytest test suite with coverage"
	@echo "  make lint       - Run ruff linter checks"
	@echo "  make run-api    - Start FastAPI serving layer on port 8000"
	@echo "  make run-ui     - Start Streamlit interactive UI on port 8501"
	@echo "  make docker-up  - Start Neo4j, FastAPI, and Streamlit containers"
	@echo "  make clean      - Remove build and test caches"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/download_and_prepare.py

train:
	$(PYTHON) scripts/train_all_models.py

benchmark:
	$(PYTHON) scripts/benchmark_models.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=src/graphrec --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run-api:
	$(PYTHON) -m uvicorn src.graphrec.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
