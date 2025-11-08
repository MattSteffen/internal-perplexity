# API Overview

This document provides an overview of the OpenAI-Compatible Backend API and how to use it.

## Table of Contents

- [Introduction](#introduction)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Authentication Endpoints](#authentication-endpoints)
  - [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
  - [Collection Management](#collection-management)
  - [Document Processing](#document-processing)
  - [Tool Calling](#tool-calling)
- [Examples](#examples)
- [Error Handling](#error-handling)

## Introduction

The OpenAI-Compatible Backend API is a FastAPI-based service that provides:

1. **OpenAI-compatible endpoints** for chat completions, embeddings, and model listing
2. **OAuth2 authentication** via Keycloak with GitLab identity provider
3. **Document processing pipelines** for uploading and processing documents into a vector database
4. **Milvus vector database** integration for collection and user management
5. **Tool calling system** for extending LLM capabilities

The API is designed to be compatible with OpenAI's API format, making it easy to integrate with existing OpenAI-based applications while adding custom functionality.

## Authentication

Most endpoints require authentication via JWT tokens obtained through the OAuth2 flow.

### OAuth2 Flow

1. **Initiate Login**: `GET /login`
   - Redirects to Keycloak/GitLab for authentication
   - No authentication required

2. **Callback**: `GET /auth/callback?code=<code>&state=<state>`
   - Handled automatically by the browser
   - Sets an HTTP-only cookie with the access token

3. **Use Token**: Include the token in requests:
   ```bash
   Authorization: Bearer <access_token>
   ```

4. **Logout**: `GET /logout`
   - Clears the access token cookie
   - Redirects to Keycloak logout

### Getting Current User

```bash
curl -X GET http://localhost:8000/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

## Endpoints

### Health Check

#### `GET /health`

Check if the API server is running.

**No authentication required**

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Authentication Endpoints

#### `GET /login`

Initiates OAuth2 login flow. Redirects to Keycloak/GitLab.

**No authentication required**

**Example:**
```bash
curl -X GET http://localhost:8000/login
```

#### `GET /auth/callback`

OAuth2 callback handler. Called automatically after authentication.

**No authentication required**

#### `GET /logout`

Logs out the user and clears the access token cookie.

**No authentication required**

#### `GET /auth/me`

Returns the current authenticated user's information.

**Requires authentication**

**Example:**
```bash
curl -X GET http://localhost:8000/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

Create chat completions with streaming support and automatic tool calling.

**No authentication required** (but tools may require it)

**Request Body:**
```json
{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7
}
```

**Example (non-streaming):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Example (streaming):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Features:**
- Automatically includes available tools in the request
- Handles tool calls by executing them and re-querying until a final response
- Supports both streaming and non-streaming modes
- Compatible with OpenAI's chat completion format

#### `POST /v1/embeddings`

Generate vector embeddings for text input.

**No authentication required**

**Request Body:**
```json
{
  "model": "all-minilm:v2",
  "input": "The food was delicious"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-minilm:v2",
    "input": "The food was delicious"
  }'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "all-minilm:v2",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

#### `GET /v1/models`

List all available models from Ollama and custom agents.

**No authentication required**

**Example:**
```bash
curl -X GET http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2:1b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ollama"
    },
    {
      "id": "radchat",
      "object": "model",
      "created": 1234567890,
      "owned_by": "custom"
    }
  ]
}
```

### Collection Management

#### `GET /v1/collections`

List all Milvus collections with their metadata.

**Requires authentication** (JWT token with Milvus credentials)

**Example:**
```bash
curl -X GET http://localhost:8000/v1/collections \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "collections": ["collection1", "collection2"],
  "collection_metadata": {
    "collection1": {
      "name": "collection1",
      "description": "...",
      ...
    }
  }
}
```

#### `GET /v1/users`

List all Milvus users with their roles.

**Requires authentication** (JWT token with Milvus credentials)

**Example:**
```bash
curl -X GET http://localhost:8000/v1/users \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "users": [
    {
      "id": "user1",
      "name": "user1",
      "roles": ["admin", "public"]
    },
    {
      "id": "user2",
      "name": "user2",
      "roles": ["public"]
    }
  ]
}
```

#### `GET /v1/roles`

List all Milvus roles with their privileges.

**Requires authentication** (JWT token with Milvus credentials)

**Example:**
```bash
curl -X GET http://localhost:8000/v1/roles \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "roles": [
    {
      "role": "admin",
      "privileges": ["CreateCollection", "DropCollection", "ManageUser"]
    },
    {
      "role": "public",
      "privileges": ["CollectionRead", "CollectionQuery"]
    }
  ]
}
```

#### `POST /v1/collections`

Create a new collection with pipeline configuration and permissions.

**Requires authentication** (JWT token with Milvus credentials)

**Request Body:**
```json
{
  "collection_name": "my_collection",
  "pipeline_name": "irads",
  "config_overrides": {
    "embedding_model": "nomic-embed-text"
  },
  "description": "My collection description",
  "default_permissions": "public",
  "metadata_schema": {}
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/collections \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_collection",
    "pipeline_name": "irads",
    "default_permissions": "public"
  }'
```

**Response:**
```json
{
  "collection_name": "my_collection",
  "message": "Collection created successfully",
  "pipeline_name": "irads",
  "permissions": {
    "default": "public"
  }
}
```

**Request Fields:**
- `collection_name` (string, required): Name of the collection to create
- `pipeline_name` (string, optional): Name of predefined pipeline to use (required if `custom_config` not provided)
- `custom_config` (object, optional): Full CrawlerConfig dict for custom pipeline (required if `pipeline_name` not provided)
- `config_overrides` (object, optional): Configuration overrides for predefined pipeline
- `description` (string, optional): Human-readable description of the collection
- `default_permissions` (string, optional): Default permission level - `"admin_only"` or `"public"` (default: `"admin_only"`)
- `metadata_schema` (object, optional): Optional JSON schema override for metadata

### Document Processing

#### `POST /v1/documents/upload/{pipeline_name}`

Upload and process a document through a predefined pipeline.

**Requires authentication** (JWT token with Milvus credentials)

**Available Pipelines:**
- `irads`: IRADS document processing pipeline
- `arxiv_math`: ArXiv mathematics paper processing pipeline

**Request:**
- `file`: Document file (multipart/form-data)
- `config_overrides`: Optional JSON string with configuration overrides

**Example:**
```bash
curl -X POST http://localhost:8000/v1/documents/upload/irads \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F 'config_overrides={"embedding_model": "nomic-embed-text", "security_groups": ["group1"]}'
```

#### `POST /v1/documents/upload`

Upload and process a document to a collection (loads config from collection).

**Requires authentication** (JWT token with Milvus credentials)

**Request:**
- `file`: Document file (multipart/form-data)
- `collection_name`: Collection name (query parameter)
- `config_overrides`: Optional JSON string with configuration overrides

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/documents/upload?collection_name=my_collection" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F 'config_overrides={"security_groups": ["group1"]}'
```

#### `POST /v1/collections/{collection_name}/process`

Process a document to extract metadata without uploading (collection-specific).

**Requires authentication** (JWT token with Milvus credentials)

**Request:**
- `file`: Document file (multipart/form-data)

**Example:**
```bash
curl -X POST http://localhost:8000/v1/collections/my_collection/process \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "metadata": {
    "title": "Example Document",
    "author": "John Doe"
  },
  "file_name": "document.pdf",
  "file_size": 12345
}
```

#### `POST /v1/collections/{collection_name}/upload`

Upload a document to a collection with metadata.

**Requires authentication** (JWT token with Milvus credentials)

**Request:**
- `file`: Document file (multipart/form-data)
- `metadata`: JSON string with document metadata

**Example:**
```bash
curl -X POST http://localhost:8000/v1/collections/my_collection/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F 'metadata={"title":"Example","author":"John Doe"}'
```

**Response:**
```json
{
  "message": "Document processed and uploaded successfully",
  "pipeline_name": "my_collection",
  "document_id": "upload_12345",
  "chunks_created": 0,
  "processing_time_sec": 12.34
}
```

**Configuration Overrides (for pipeline-based uploads):**
```json
{
  "embedding_model": "nomic-embed-text",
  "llm_model": "llama3.2:1b",
  "vision_model": "llava:latest",
  "security_groups": ["group1", "group2"]
}
```

**Processing Steps:**
1. Document is uploaded and saved temporarily
2. Document is converted to markdown format
3. Document is chunked into smaller pieces
4. Chunks are embedded using the specified embedding model
5. Chunks are stored in the Milvus vector database
6. Temporary file is cleaned up

### Tool Calling

#### `POST /v1/tools`

Execute a tool directly without going through chat completions.

**No authentication required** (but tools may require it)

**Available Tools:**
- `get_weather`: Get weather information for a location
- `search`: Search the Milvus vector database

**Request Body:**
```json
{
  "name": "get_weather",
  "arguments": {
    "location": "San Francisco, CA"
  },
  "metadata": {
    "user_id": "user123"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/tools \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "arguments": {"location": "San Francisco, CA"},
    "metadata": {"user_id": "user123"}
  }'
```

**Response:**
```json
{
  "result": "{\"temperature\": 72, \"condition\": \"sunny\"}"
}
```

**Tool: `get_weather`**
- **Arguments:**
  - `location` (string, required): Location to get weather for
- **Returns:** JSON string with weather information

**Tool: `search`**
- **Arguments:**
  - `query` (string, required): Search query
  - `collection_name` (string, optional): Collection to search
  - `partition_name` (string, optional): Partition to search
  - `limit` (integer, optional): Maximum number of results
- **Returns:** JSON string with search results

## Examples

### Complete Chat Completion with Tool Calling

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "stream": false
  }'
```

The API will automatically:
1. Include available tools in the request
2. If the LLM calls a tool, execute it
3. Re-query the LLM with the tool results
4. Return the final response

### Upload and Process Document

```bash
# Upload a PDF document
curl -X POST http://localhost:8000/v1/documents/upload/irads \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@research_paper.pdf" \
  -F 'config_overrides={"embedding_model": "nomic-embed-text"}'
```

### Search Vector Database via Tool

```bash
curl -X POST http://localhost:8000/v1/tools \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search",
    "arguments": {
      "query": "machine learning",
      "limit": 5
    }
  }'
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `302`: Redirect (for OAuth endpoints)
- `400`: Bad Request - Invalid input
- `401`: Unauthorized - Missing or invalid token
- `404`: Not Found - Resource doesn't exist
- `500`: Internal Server Error
- `502`: Bad Gateway - Failed to connect to upstream service
- `503`: Service Unavailable - Service not configured or unavailable

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Errors:**

1. **Missing Authentication:**
   ```json
   {
     "detail": "Milvus token is required"
   }
   ```

2. **Invalid Pipeline:**
   ```json
   {
     "detail": "Pipeline 'invalid' not found. Available pipelines: ['irads', 'arxiv_math']"
   }
   ```

3. **Invalid JSON:**
   ```json
   {
     "detail": "Invalid JSON in request body: ..."
   }
   ```

4. **Tool Not Found:**
   ```json
   {
     "detail": "Tool 'invalid_tool' is not registered"
   }
   ```

## Configuration

The API can be configured via environment variables with the `OAI_` prefix:

- `OAI_OLLAMA_BASE_URL`: Base URL for Ollama (default: `http://localhost:11434/v1`)
- `OAI_API_KEY`: API key for Ollama (default: `ollama`)
- `OAI_HOST`: Host to bind server (default: `0.0.0.0`)
- `OAI_PORT`: Port to bind server (default: `8000`)
- `OAI_KEYCLOAK_URL`: Keycloak realm URL
- `OAI_CLIENT_ID`: OAuth2 client ID
- `OAI_CLIENT_SECRET`: OAuth2 client secret
- `OAI_REDIRECT_URI`: OAuth2 redirect URI
- `OAI_FRONTEND_REDIRECT_URL`: Frontend redirect URL after authentication

## OpenAPI Specification

A complete OpenAPI 3.1 specification is available at `openapi.yaml` in the project root. This can be used with tools like Swagger UI or Postman to explore and test the API.

## Additional Resources

- See `overview.md` files in each package directory for detailed documentation of internal components
- See `src/endpoints/` for endpoint implementation details
- See `src/tools/` for available tools and how to add new ones
- See `src/crawler/` for document processing pipeline details

