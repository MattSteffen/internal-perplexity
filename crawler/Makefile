.PHONY: install check-install lint format test

# Install dependencies
install:
	uv sync

# Verify installation works by importing the package
check-install:
	uv run python -c "import crawler; from crawler import Crawler, CrawlerConfig; print('✓ Installation verified')"

# Lint code
lint:
	uv run ruff check src/
	uv run black --check src/
	# TODO: Enable mypy
	# uv run mypy --explicit-package-bases src/

# Format code
format:
	uv run black src/
	uv run ruff check --fix src/ || true

# Run tests
test:
	uv run pytest

