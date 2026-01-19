"""Main FastAPI application."""

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from src.auth import init_oauth
from src.auth_utils import get_optional_token, verify_token
from src.config import settings
from src.endpoints import agent, auth, chat, collections, documents, embeddings, models, search, tools

app = FastAPI(
    title="OpenAI-Compatible Backend",
    description="OpenAI-compatible API proxy for Ollama",
    version="0.1.0",
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth with the FastAPI app
init_oauth(app)

# Include authentication routes
app.include_router(auth.router)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    user: dict[str, any] | None = Depends(get_optional_token),
) -> StreamingResponse | ChatCompletion:
    """OpenAI-compatible chat completions endpoint.

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
      }'
    """
    return await chat.create_chat_completion(request, user=user)


@app.post("/v1/embeddings")
async def embeddings_endpoint(request: Request) -> CreateEmbeddingResponse:
    """OpenAI-compatible embeddings endpoint."""
    return await embeddings.create_embedding(request)


@app.get("/v1/models")
async def list_models_endpoint(request: Request) -> models.ModelList:
    """OpenAI-compatible models listing endpoint."""
    return await models.list_models(request)


@app.get("/v1/collections")
async def list_collections_endpoint(user: dict = Depends(verify_token)) -> collections.CollectionsResponse:
    """List all Milvus collections with metadata.

    curl -X GET http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN"
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")
    return await collections.list_collections(token)


@app.post("/v1/collections")
async def create_collection_endpoint(
    request: collections.CreateCollectionRequest,
    user: dict = Depends(verify_token),
) -> collections.CreateCollectionResponse:
    """Create a new collection with pipeline configuration.

    Using a template:

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "template_name": "standard",
        "access_level": "public"
      }'

    Or with custom config:

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "custom_config": {...},
        "access_level": "public"
      }'
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")
    return await collections.create_collection(request, token)


@app.get("/v1/pipelines")
async def list_pipelines_endpoint() -> collections.PipelinesResponse:
    """List available pipeline templates.

    curl -X GET http://localhost:8000/v1/pipelines
    """
    return await collections.list_pipelines()


@app.get("/v1/roles")
async def list_roles_endpoint(user: dict = Depends(verify_token)) -> collections.RolesResponse:
    """List all Milvus roles with their privileges.

    curl -X GET http://localhost:8000/v1/roles \
      -H "Authorization: Bearer $TOKEN"
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")
    return await collections.list_roles(token)


@app.get("/v1/users")
async def list_users_endpoint(user: dict = Depends(verify_token)) -> collections.UsersResponse:
    """List all Milvus users with their roles.

    curl -X GET http://localhost:8000/v1/users \
      -H "Authorization: Bearer $TOKEN"
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")
    return await collections.list_users(token)


@app.post("/v1/collections/{collection_name}/process")
async def process_document_for_collection_endpoint(
    collection_name: str,
    file: UploadFile = File(...),
    user: dict = Depends(verify_token),
) -> documents.ProcessedDocument:
    """Process a document to extract metadata without uploading (collection-specific).

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/process \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf"
    """
    return await documents.process_document(
        file=file,
        collection_name=collection_name,
        user=user,
    )


@app.post("/v1/collections/{collection_name}/upload")
async def upload_document_for_collection_endpoint(
    collection_name: str,
    file: UploadFile = File(None),
    markdown_content: str | None = Form(None),
    metadata_override: str | None = Form(None),
    security_groups: str | None = Form(None),
    user: dict = Depends(verify_token),
) -> documents.UploadResponse:
    """Upload and process a document to a collection (loads config from collection).

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf"
    """
    return await documents.upload_document(
        collection_name=collection_name,
        user=user,
        file=file,
        markdown_content=markdown_content,
        metadata_override=metadata_override,
        security_groups=security_groups,
    )


@app.post("/v1/search")
async def search_endpoint(
    request: search.SearchRequest,
    user: dict = Depends(verify_token),
) -> search.SearchResponse:
    """Search a Milvus collection with hybrid search.

    curl -X POST http://localhost:8000/v1/search \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection": "my_collection",
        "text": "query text",
        "filters": ["title == \"example\""],
        "limit": 100
      }'
    """
    return await search.search(request, user)


@app.post("/v1/tools")
async def call_tool_endpoint(request: Request) -> tools.ToolCallResponse:
    """Direct tool calling endpoint.

    curl -X POST http://localhost:8000/v1/tools \
      -H "Content-Type: application/json" \
      -d '{
        "name": "get_weather",
        "arguments": {"location": "San Francisco, CA"},
        "metadata": {"user_id": "user123"}
      }'
    """
    return await tools.call_tool(request)


@app.post("/v1/agent", response_model=None)
async def agent_endpoint(
    request: Request,
    user: dict = Depends(verify_token),
) -> StreamingResponse | ChatCompletion:
    """Agentic RAG endpoint connected to a specific Milvus collection.

    Creates an agentic conversation that uses the collection's llm_prompt
    for system prompt generation and performs semantic search as needed.

    curl -X POST http://localhost:8000/v1/agent \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "model": "milvuschat",
        "collection": "my_collection",
        "messages": [{"role": "user", "content": "What documents discuss machine learning?"}],
        "stream": false
      }'
    """
    return await agent.create_agent_completion(request, user=user)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
