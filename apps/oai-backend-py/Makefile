.PHONY: install dev lint test run build

# Install dependencies
install:
	uv sync

# Run the development server
dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Lint code
lint:
	uv run ruff check src/
	uv run black --check src/
	uv run mypy --explicit-package-bases src/

# Format code
format:
	uv run black src/
	uv run ruff check --fix src/

# Run tests
test:
	uv run pytest

# Run the production server
run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Build Docker image (optional)
build:
	docker build -t oai-backend-py .

