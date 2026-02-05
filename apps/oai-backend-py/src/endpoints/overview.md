# endpoints/ Package Overview

This package contains FastAPI endpoint handlers for the OpenAI-compatible API.

## Files

### `__init__.py`
Package marker for endpoint modules.

### `agent.py`
Agent endpoint handler for collection-aware, tool-calling RAG conversations.

### `auth.py`
OAuth2 login, callback, logout, and user info endpoints for Keycloak.

### `chat.py`
Chat completions handler with streaming and tool-calling support.

### `collections.py`
Collection management endpoints for listing, creating, and role/user access queries.

### `documents.py`
Document processing and upload handlers for collection ingestion workflows.

### `embeddings.py`
Embeddings endpoint handler for OpenAI-compatible embedding requests.

### `models.py`
Models listing handler that surfaces both Ollama models and custom agents.

### `pipeline_registry.py`
Pipeline registry utilities exposed via endpoints for pipeline listing and retrieval.

### `search.py`
Search endpoint wrapper over the Milvus search tool.

### `tools.py`
Direct tool invocation endpoint for calling registered tools without chat.
