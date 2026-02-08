# agents/ Package Overview

This package contains agent implementations invoked via `/v1/agents/{agent}`.

## Files

### `__init__.py`
Exports the `agent_registry` singleton.

### `base.py`
Agent protocol definition for agent execution.

### `milvuschat.py`
MilvusChat agent for collection-aware RAG workflows with tool calling.

### `registry.py`
Agent registry that routes agent names to implementations.
