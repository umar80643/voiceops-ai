.PHONY: install test lint format typecheck run docker-up docker-down evaluate

install:
	pip install --break-system-packages -e ".[dev]"

test:
	pytest -v --cov=apps --cov=shared --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy apps shared ml

run:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

evaluate:
	python -m ml.evaluation.run_all
