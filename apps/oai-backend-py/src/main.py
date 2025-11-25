"""Main FastAPI application."""

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from src.auth import init_oauth
from src.auth_utils import get_optional_token, verify_token
from src.config import settings
from src.endpoints import auth, chat, collections, document_pipelines, embeddings, models, tools

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
        "model": "llama3.2:1b",
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

    curl -X POST http://localhost:8000/v1/collections \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection_name": "my_collection",
        "pipeline_name": "irads",
        "default_permissions": "public"
      }'
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")
    return await collections.create_collection(request, token)


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
) -> document_pipelines.ProcessedDocument:
    """Process a document to extract metadata without uploading (collection-specific).

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/process \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf"
    """
    return await document_pipelines.process_document(
        file=file,
        collection_name=collection_name,
        user=user,
    )


@app.post("/v1/collections/{collection_name}/upload")
async def upload_document_to_collection_endpoint(
    collection_name: str,
    file: UploadFile = File(...),
    metadata: str = Form(...),
    user: dict = Depends(verify_token),
) -> document_pipelines.UploadResponse:
    """Upload a document to a collection with metadata.

    curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'metadata={"title":"Example","author":"John Doe"}'
    """
    return await document_pipelines.upload_document_to_collection(
        collection_name=collection_name,
        file=file,
        metadata=metadata,
        user=user,
    )


@app.post("/v1/documents/upload/{pipeline_name}")
async def upload_document_endpoint_path(
    pipeline_name: str,
    file: UploadFile = File(...),
    config_overrides: str | None = Form(None),
    user: dict = Depends(verify_token),
) -> document_pipelines.UploadResponse:
    """Upload and process a document through a predefined pipeline.

    curl -X POST http://localhost:8000/v1/documents/upload/{pipeline_name} \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"embedding_model": "nomic-embed-text", "security_groups": ["group1"]}'
    """
    return await document_pipelines.upload_document(
        pipeline_name=pipeline_name,
        collection_name=None,
        file=file,
        config_overrides=config_overrides,
        user=user,
    )


@app.post("/v1/documents/upload")
async def upload_document_endpoint(
    collection_name: str | None = None,
    file: UploadFile = File(...),
    config_overrides: str | None = Form(None),
    user: dict = Depends(verify_token),
) -> document_pipelines.UploadResponse:
    """Upload and process a document to a collection (loads config from collection).

    curl -X POST "http://localhost:8000/v1/documents/upload?collection_name=my_collection" \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@document.pdf" \
      -F 'config_overrides={"security_groups": ["group1"]}'
    """
    return await document_pipelines.upload_document(
        pipeline_name=None,
        collection_name=collection_name,
        file=file,
        config_overrides=config_overrides,
        user=user,
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
