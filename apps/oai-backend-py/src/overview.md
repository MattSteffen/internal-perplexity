# src/ Package Overview

This package contains the main application code for the OpenAI-compatible backend proxy with Milvus vector database integration.

## Files

### `__init__.py`
Empty package marker file.

### `main.py`
FastAPI application entry point. Defines the main app instance and routes all endpoints. Contains:
- FastAPI app initialization
- OAuth initialization with Keycloak
- Health check endpoint (`/health`)
- Route registration for all API endpoints including:
  - Authentication
  - Chat completions and agent endpoint
  - Embeddings and models listing
  - Collections, pipelines, and document management
  - Search and tool calling

### `auth.py`
OAuth2 authentication setup with Keycloak. Initializes the OAuth client and registers Keycloak as an OAuth provider:
- `init_oauth(app)`: Initializes OAuth with FastAPI app and registers Keycloak provider
- Registers Keycloak using OpenID Connect discovery endpoint
- Configures GitLab identity provider hint for automatic redirect

### `auth_utils.py`
Authentication utilities for token verification. Provides:
- `verify_token(credentials)`: Verifies token and returns decoded claims with `milvus_token` and `username`
- `get_optional_token(credentials)`: Optional token verification that returns None if token is missing
- `extract_username_from_token(token)`: Extracts username from `username:password` token format
- Raises HTTP 401 for invalid/expired tokens

### `config.py`
Configuration management using Pydantic Settings. Includes:

**Application Settings** (OAI_ prefix):
- `ollama_base_url`: Base URL for Ollama API (default: `http://localhost:11434/v1`)
- `api_key`: API key for Ollama (default: `ollama`)
- `host`: Server host binding (default: `0.0.0.0`)
- `port`: Server port (default: `8000`)
- Keycloak OAuth2 settings (keycloak_url, client_id, client_secret, redirect_uri)

**RadChat Config** (for agent functionality):
- `OllamaConfig`: embedding_model, llm_model, request_timeout, context_length
- `MilvusConfig`: host, port, username, password, collection_name
- `SearchConfig`: nprobe, search_limit, hybrid_limit, rrf_k, drop_ratio
- `AgentConfig`: max_tool_calls, default_role, logging_level

### `milvus_client.py`
Milvus client pool and connection helpers for database operations. Provides:
- `MilvusClientPool`: Thread-safe TTL + LRU pool keyed by stable user identifiers; falls back to a token fingerprint only when needed (tokens are still held by the client in memory).
- `get_milvus_context()` / `get_milvus_client()`: FastAPI dependencies that use `get_current_user()` and a stub `get_milvus_token_for_user()` to retrieve per-user clients from the pool.
- `get_milvus_uri()`: Returns the configured Milvus URI (e.g. for building crawler DatabaseClientConfig).
- `parse_milvus_uri(uri)`: Parses a URI string into (host, port).
- `parse_milvus_token(token)`: Parses username:password token into (username, password) for DatabaseClientConfig.

### `utils.py`
Shared utility functions. Contains `map_openai_error_to_http()` which converts OpenAI API errors to appropriate HTTP exceptions.

## Subpackages

### `clients/`
Contains client implementations for different model providers.

#### `clients/__init__.py`
Empty package marker.

#### `clients/base.py`
Defines `ChatCompletionClient` and `EmbeddingClient` protocols for unified interfaces.

#### `clients/ollama.py`
Ollama client implementation. Wraps AsyncOpenAI client to proxy requests to Ollama.

#### `clients/radchat.py`
RadChat agent with hardcoded collection. Full agentic RAG implementation with:
- Milvus vector search tool
- System prompt with database schema and preliminary context
- Citation building for source attribution
- Streaming response support

#### `clients/milvuschat.py`
MilvusChat agent that dynamically connects to any collection. Features:
- Uses collection's `llm_prompt` from CollectionDescription for system prompt
- Performs initial search for preliminary context
- Agentic tool-calling loop with search tool
- Both streaming and non-streaming modes
- Collection-specific metadata summaries
- Pulls the canonical `search` tool schema from the shared tool registry

#### `clients/router.py`
Client router that selects clients based on model name:
- `radchat` -> RadChatClient
- `milvuschat` -> MilvusChatClient
- All other models -> OllamaClient

### `endpoints/`
Contains endpoint handler functions for API routes.

#### `endpoints/__init__.py`
Package exports for all endpoint modules.

#### `endpoints/agent.py`
Agent endpoint handler. Implements `POST /v1/agent` for agentic RAG conversations:
- Connects to specified Milvus collection
- Uses collection's llm_prompt for system prompt
- Supports streaming and non-streaming responses
- Requires authentication with Milvus token

#### `endpoints/chat.py`
Chat completions endpoint handler. Implements `POST /v1/chat/completions` with:
- Streaming responses (Server-Sent Events format)
- Tool/function calling with multi-turn support (up to 5 iterations)
- Automatic tool injection from registry
- Error handling for streaming responses
- `milvuschat` requests require `collection` and default `token` from authenticated user context when omitted

#### `endpoints/collections.py`
Collections management. Implements:
- `GET /v1/collections`: List collections with metadata, document counts, and access levels
- `POST /v1/collections`: Create new collections from full CrawlerConfig JSON
  - Request: `access_level`, `access_groups` (when group_only), `crawler_config`
  - Collection name from `crawler_config.database.collection`
  - Persists `security_groups` so the creator (and, for group_only, access_groups) pass app-level RBAC on upload
  - Grants Milvus `CollectionReadWrite` to the creator role and, when `access_level` is group_only, to each `access_groups` role; fails with 400 if any role is missing in Milvus (best-effort drop_collection on failure)
  - Validates `crawler_config` with CrawlerConfig.from_dict(); returns 400 with detail on failure
- `GET /v1/pipelines`: List pipeline templates (PipelineInfo: name, description, metadata_schema, chunk_size, embedding_model, llm_model). No auth.
- `GET /v1/pipelines/{name}`: Return full crawler-config JSON for the given pipeline. 404 if not found. No auth.
- `GET /v1/roles`: List Milvus roles
- `GET /v1/users`: List Milvus users with their roles

#### `endpoints/documents.py`
Document processing and upload. Implements:
- `POST /v1/collections/{collection_name}/process`: Process document to extract metadata and markdown
  - Returns `ProcessedDocument` with metadata, markdown_content, file_name, file_size
- `POST /v1/collections/{collection_name}/upload`: Upload documents to collection
  - Both modes (file and pre-processed markdown) use `Crawler.crawl_document`; crawler skips convert/extract when markdown/metadata are already set
  - Supports raw file upload with automatic processing
  - Supports pre-processed markdown_content with metadata_override
  - RBAC check verifies user has write access to collection
  - Optional security_groups parameter for document-level access control
  - Helpers: `_load_config_from_collection()`, `_parse_security_groups()`

#### `endpoints/embeddings.py`
Embeddings endpoint. Implements `POST /v1/embeddings` for generating embeddings.

#### `endpoints/models.py`
Models listing. Implements `GET /v1/models` listing:
- All Ollama models from `/api/tags`
- Custom agents: `radchat`, `milvuschat`

#### `endpoints/pipeline_registry.py`
Name-to-JSON pipeline registry. Pipelines are crawler-config-shaped dicts only; no CrawlerConfig creation or schema validation in this module.
- `PipelineRegistry`: Stores pipeline name → full config JSON (dict). `register(name, config_dict)`, `get(name)` returns a copy of the JSON, `list_pipelines()`, `has_pipeline(name)`, `get_pipeline_info()` (best-effort display fields from JSON, no validation).
- Call sites obtain config by: `CrawlerConfig.from_dict(registry.get(name))`; validation happens at use site via `from_dict`.
- Built-in pipelines: `standard` (general document processing, 2000 char chunks), `academic` (research papers, 10000 char chunks). `get_registry()` returns the global registry instance.

#### `endpoints/search.py`
Search endpoint. Implements `POST /v1/search` as a thin wrapper over the Milvus search tool:
- Delegates to `tools/milvus_search.search_async()` for both semantic and filter-only queries.
- Consolidates `SearchResult` into crawler `Document` via `tools/milvus_search.consolidate_documents(...)`.
- Returns `SearchResponse(results=list[Document], total=len)`; security group filtering is applied by the crawler’s MilvusDB.

#### `endpoints/auth.py`
Authentication endpoints for Keycloak OAuth2:
- `GET /login`: Initiates OAuth2 login flow
- `GET /auth/callback`: Handles OAuth2 callback
- `GET /logout`: Logout and cookie clearing
- `GET /auth/me`: Current user information

#### `endpoints/tools.py`
Direct tool calling endpoint. Implements `POST /v1/tools` for invoking tools outside of chat.

### `tools/`
Tool system for function/tool calling.

#### `tools/__init__.py`
Exports the `tool_registry` singleton.

#### `tools/base.py`
Defines the `Tool` protocol:
- `get_definition()`: Returns ChatCompletionToolParam
- `execute(arguments)`: Async execution returning JSON string

#### `tools/milvus_search.py`
Milvus search tool implementation; delegates to crawler MilvusDB. Provides:
- `MilvusSearchTool` class: single "search" tool; no text/queries => filter-only query
- `search()`: Builds DatabaseClientConfig from env + token, instantiates MilvusDB (crawler_config=None), connects, then calls db.search(). Filter-only: db.search(texts=[], filters=..., limit=...). Semantic: db.get_collection() for embedding model, set_embedder(), db.search(texts=[query_text], ...)
- `consolidate_documents(search_results)`: Groups SearchResult by document_id, returns list of crawler Document
- `build_citations(documents)`: OI-style citations from list of Document
- `render_document(doc, include_text)`: Renders crawler Document to markdown
- Uses crawler types: DatabaseDocument, SearchResult, Document; token from metadata or constructor

#### `tools/weather.py`
Mock weather tool for demonstration purposes.

#### `tools/registry.py`
Tool registry managing available tools:
- `get_tool_definitions()`: Returns all tool definitions
- `execute_tool(name, arguments, metadata)`: Executes tool with optional user metadata
- Default tools: `get_weather`, `search`

