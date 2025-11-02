"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from src.auth import init_oauth
from src.config import settings
from src.endpoints import auth, chat, collections, embeddings, models, tools

app = FastAPI(
    title="OpenAI-Compatible Backend",
    description="OpenAI-compatible API proxy for Ollama",
    version="0.1.0",
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
async def chat_completions(request: Request) -> StreamingResponse | ChatCompletion:
    """OpenAI-compatible chat completions endpoint."""
    return await chat.create_chat_completion(request)


@app.post("/v1/embeddings")
async def embeddings_endpoint(request: Request) -> CreateEmbeddingResponse:
    """OpenAI-compatible embeddings endpoint."""
    return await embeddings.create_embedding(request)


@app.get("/v1/models")
async def list_models_endpoint(request: Request) -> models.ModelList:
    """OpenAI-compatible models listing endpoint."""
    return await models.list_models(request)


@app.get("/v1/collections")
async def list_collections_endpoint() -> collections.CollectionsResponse:
    """List all Milvus collections with metadata.

    curl -X GET http://localhost:8000/v1/collections
    """
    return await collections.list_collections()


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
