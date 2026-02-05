"""Tests package overview."""

# tests/ Package Overview

This package contains pytest-based tests for the FastAPI application.

## Files

### `__init__.py`
Package marker for tests.

### `test_main.py`
Smoke test for the `/health` endpoint using FastAPI's TestClient.

### `test_chat.py`
Chat completions endpoint coverage for milvuschat parameter validation and token inference.

### `test_search.py`
Regression coverage for the search handler when a collection description is missing.
