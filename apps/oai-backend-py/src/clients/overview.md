# clients/ Package Overview

This package contains client implementations for model providers.

## Files

### `__init__.py`
Package marker for client modules.

### `base.py`
Protocol definitions for `ChatCompletionClient` and `EmbeddingClient` interfaces.

### `ollama.py`
Ollama client implementation that proxies OpenAI-compatible requests to Ollama.

### `router.py`
Client router that maps model names to the correct client implementation.

## Related

Agent implementations live in `src/agents/`.
