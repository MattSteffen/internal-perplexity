# clients/ Package Overview

This package contains client implementations for different model providers and agents.

## Files

### `__init__.py`
Package marker for client modules.

### `base.py`
Protocol definitions for `ChatCompletionClient` and `EmbeddingClient` interfaces.

### `milvuschat.py`
MilvusChat agent client for collection-specific RAG workflows with tool-calling support.

### `ollama.py`
Ollama client implementation that proxies OpenAI-compatible requests to Ollama.

### `radchat.py`
RadChat agent client with a fixed collection and agentic tool-calling loop.

### `router.py`
Client router that maps model names to the correct client implementation.
