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
Milvus client factory for database operations. Provides `get_milvus_client(token)` which creates a MilvusClient with authentication.

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

#### `endpoints/collections.py`
Collections management. Implements:
- `GET /v1/collections`: List collections with metadata, document counts, and access levels
- `POST /v1/collections`: Create new collections with template or custom config
  - Supports `template_name` (standard, academic) or `custom_config`
  - Sets security groups based on `roles` parameter
  - Grants CollectionReadOnly privileges to specified roles
- `GET /v1/roles`: List Milvus roles
- `GET /v1/users`: List Milvus users with their roles

#### `endpoints/documents.py`
Document processing and upload. Implements:
- `POST /v1/collections/{collection_name}/process`: Process document to extract metadata and markdown
  - Returns `ProcessedDocument` with metadata, markdown_content, file_name, file_size
- `POST /v1/collections/{collection_name}/upload`: Upload documents to collection
  - Supports raw file upload with automatic processing
  - Supports pre-processed markdown_content with metadata_override
  - RBAC check verifies user has write access to collection
  - Optional security_groups parameter for document-level access control

#### `endpoints/embeddings.py`
Embeddings endpoint. Implements `POST /v1/embeddings` for generating embeddings.

#### `endpoints/models.py`
Models listing. Implements `GET /v1/models` listing:
- All Ollama models from `/api/tags`
- Custom agents: `radchat`, `milvuschat`

#### `endpoints/pipeline_registry.py`
Pipeline template registry for crawler configurations. Provides:
- `PipelineRegistry` class for managing predefined templates
- Two built-in templates:
  - `standard`: General document processing (2000 char chunks)
  - `academic`: Research paper processing (10000 char chunks, richer metadata)
- Templates define LLM, embedding, chunking, and metadata extraction settings
- `get_registry()`: Returns global registry instance

#### `endpoints/search.py`
Search endpoint. Implements `POST /v1/search` with:
- Hybrid search (dense + sparse embeddings)
- Automatic security group filtering based on user roles
- Custom filter expression support
- Returns combined Document objects with chunks merged by document_id

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
Milvus search tool implementation. Provides:
- `MilvusSearchTool` class for semantic search
- `perform_search()`: Hybrid search with dense + sparse embeddings
- `perform_query()`: Filter-based queries
- `consolidate_documents()`: Groups chunks by document_id
- `build_citations()`: Creates OpenWebUI-format citations

#### `tools/weather.py`
Mock weather tool for demonstration purposes.

#### `tools/registry.py`
Tool registry managing available tools:
- `get_tool_definitions()`: Returns all tool definitions
- `execute_tool(name, arguments, metadata)`: Executes tool with optional user metadata
- Default tools: `get_weather`, `search`

