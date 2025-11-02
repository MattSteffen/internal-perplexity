# src/ Package Overview

This package contains the main application code for the OpenAI-compatible backend proxy.

## Files

### `__init__.py`
Empty package marker file.

### `main.py`
FastAPI application entry point. Defines the main app instance and routes all endpoints. Contains:
- FastAPI app initialization
- Health check endpoint (`/health`)
- Route registration for chat completions, embeddings, and models listing

### `config.py`
Configuration management using Pydantic Settings. Loads settings from environment variables with `OAI_` prefix:
- `ollama_base_url`: Base URL for Ollama API (default: `http://localhost:11434/v1`)
- `api_key`: API key for Ollama (default: `ollama`)
- `host`: Server host binding (default: `0.0.0.0`)
- `port`: Server port (default: `8000`)

### `client.py`
Legacy OpenAI client instance (deprecated in favor of client router). This file is kept for backwards compatibility but new code should use `app.clients.router`.

## Subpackages

### `clients/`
Contains client implementations for different model providers with a unified interface.

#### `clients/__init__.py`
Empty package marker.

#### `clients/base.py`
Defines the `ChatCompletionClient` protocol that all clients must implement. This provides a unified interface for any chat completion provider.

#### `clients/ollama.py`
Ollama client implementation. Wraps the OpenAI-compatible AsyncOpenAI client to proxy requests to Ollama. Handles all Ollama models.

#### `clients/radchat.py`
RadChat custom agent implementation. A simple agent that returns "hello world" as a demonstration of custom agents. Returns two streaming chunks: "hello" and " world".

#### `clients/router.py`
Client router that selects the appropriate client based on model name. Routes `radchat` to the RadChatClient and all other models to the OllamaClient.

### `utils.py`
Shared utility functions for error handling and common operations. Contains `map_openai_error_to_http()` which converts OpenAI API errors to appropriate HTTP exceptions with proper status codes (404 for model not found, 400 for bad requests, 503 for service unavailable, etc.).

### `endpoints/`
Contains endpoint handler functions for API routes.

#### `endpoints/__init__.py`
Empty package marker.

#### `endpoints/chat.py`
Chat completions endpoint handler. Implements `POST /v1/chat/completions` with support for:
- Streaming responses (Server-Sent Events format)
- Non-streaming responses
- Full OpenAI-compatible request/response format
- Tool/function calling:
  - Automatically adds all registered tools to requests (unless user provides `tools` parameter)
  - Executes tool calls and injects results back into conversation
  - Supports multi-turn tool calling (up to 5 iterations)
  - Handles tool calls in both streaming and non-streaming modes
- Comprehensive error handling:
  - 400: Invalid JSON or missing required fields
  - 404: Model not found
  - 503: Ollama service unavailable
  - Proper error formatting for streaming responses

#### `endpoints/embeddings.py`
Embeddings endpoint handler. Implements `POST /v1/embeddings` for generating embeddings using Ollama models. Includes error handling:
- 400: Invalid JSON or missing required fields (model, input)
- 404: Model not found
- 503: Ollama service unavailable

#### `endpoints/models.py`
Models listing endpoint handler. Implements `GET /v1/models` for listing available models in OpenAI-compatible format. Fetches Ollama models from Ollama's native `/api/tags` endpoint and includes custom agent models (like `radchat`) in the response.

#### `endpoints/collections.py`
Collections endpoint handler. Implements `GET /v1/collections` for listing all Milvus collections with their metadata. Returns:
- `collections`: List of collection names
- `collection_metadata`: Dictionary mapping collection names to their metadata (includes all fields returned by Milvus `describe_collection`)

### `milvus_client.py`
Milvus client singleton for database operations. Provides a single `MilvusClient` instance that can be reused across the application. Connection URI is configurable via `MILVUS_URI` environment variable (defaults to `http://localhost:19530`).

### `tools/`
Contains the tool system for function/tool calling functionality.

#### `tools/__init__.py`
Package marker that exports the `tool_registry` singleton instance.

#### `tools/base.py`
Defines the `Tool` protocol that all tools must implement. Tools must provide:
- `get_definition()`: Returns an OpenAI-compatible `ChatCompletionToolParam` definition
- `execute(arguments)`: Async method that executes the tool with given arguments and returns a JSON string result

#### `tools/weather.py`
Weather tool implementation. Provides a mock weather tool that returns weather data for a given location. Includes:
- Location parameter (required): city and state
- Unit parameter (optional): celsius or fahrenheit (defaults to celsius)
- Returns temperature, conditions, and humidity as JSON

#### `tools/registry.py`
Tool registry that manages all available tools. Provides:
- `get_tool_definitions()`: Returns list of all tool definitions in OpenAI format
- `execute_tool(name, arguments)`: Executes a tool by name with given arguments
- `register_tool(name, tool)`: Registers a new tool instance

The registry automatically includes the weather tool. Tools are automatically added to all chat completion requests unless the user explicitly provides a `tools` parameter.

