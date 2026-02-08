"""Route registration for the FastAPI application."""

from typing import Any

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
from pymilvus import MilvusClient  # type: ignore

import src.endpoints.agent as agent
import src.endpoints.auth as auth
import src.endpoints.chat as chat
import src.endpoints.collections as collections
import src.endpoints.documents as documents
import src.endpoints.embeddings as embeddings
import src.endpoints.models as models
import src.endpoints.search as search
import src.endpoints.tools as tools
from src.auth_utils import get_optional_token
from src.milvus_client import MilvusClientContext, get_milvus_client, get_milvus_context


def include_routes(app: FastAPI) -> None:
    """Register all API routes on the given FastAPI app."""
    # Include authentication routes
    app.include_router(auth.router)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(content={"status": "healthy"})

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        user: dict[str, Any] | None = Depends(get_optional_token),
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
    async def list_collections_endpoint(
        context: MilvusClientContext = Depends(get_milvus_context),
    ) -> collections.CollectionsResponse:
        """List all Milvus collections with metadata.

        curl -X GET http://localhost:8000/v1/collections \
          -H "Authorization: Bearer $TOKEN"
        """
        return await collections.list_collections(context)

    @app.post("/v1/collections")
    async def create_collection_endpoint(
        request: collections.CreateCollectionRequest,
        context: MilvusClientContext = Depends(get_milvus_context),
    ) -> collections.CreateCollectionResponse:
        """Create a new collection from crawler_config.

        curl -X POST http://localhost:8000/v1/collections \
          -H "Authorization: Bearer $TOKEN" \
          -H "Content-Type: application/json" \
          -d '{
            "access_level": "public",
            "access_groups": [],
            "crawler_config": {"name": "my_collection", "database": {"collection": "my_collection", ...}, ...}
          }'
        """
        return await collections.create_collection(request, context)

    @app.get("/v1/pipelines")
    async def list_pipelines_endpoint() -> collections.PipelinesResponse:
        """List available pipeline templates.

        curl -X GET http://localhost:8000/v1/pipelines
        """
        return await collections.list_pipelines()

    @app.get("/v1/pipelines/{name}")
    async def get_pipeline_config_endpoint(name: str) -> dict:
        """Return full crawler-config JSON for the given pipeline.

        curl -X GET http://localhost:8000/v1/pipelines/standard
        """
        return await collections.get_pipeline_config(name)

    @app.get("/v1/roles")
    async def list_roles_endpoint(
        client: MilvusClient = Depends(get_milvus_client),
    ) -> collections.RolesResponse:
        """List all Milvus roles with their privileges.

        curl -X GET http://localhost:8000/v1/roles \
          -H "Authorization: Bearer $TOKEN"
        """
        return await collections.list_roles(client)

    @app.get("/v1/users")
    async def list_users_endpoint(
        client: MilvusClient = Depends(get_milvus_client),
    ) -> collections.UsersResponse:
        """List all Milvus users with their roles.

        curl -X GET http://localhost:8000/v1/users \
          -H "Authorization: Bearer $TOKEN"
        """
        return await collections.list_users(client)

    @app.post("/v1/collections/{collection_name}/process")
    async def process_document_for_collection_endpoint(
        collection_name: str,
        file: UploadFile = File(...),
        context: MilvusClientContext = Depends(get_milvus_context),
    ) -> documents.ProcessedDocument:
        """Process a document to extract metadata without uploading (collection-specific).

        curl -X POST http://localhost:8000/v1/collections/{collection_name}/process \
          -H "Authorization: Bearer $TOKEN" \
          -F "file=@document.pdf"
        """
        return await documents.process_document(
            file=file,
            collection_name=collection_name,
            user=context.user,
            milvus_client=context.client,
        )

    @app.post("/v1/collections/{collection_name}/upload")
    async def upload_document_for_collection_endpoint(
        collection_name: str,
        file: UploadFile = File(None),
        markdown_content: str | None = Form(None),
        metadata_override: str | None = Form(None),
        security_groups: str | None = Form(None),
        context: MilvusClientContext = Depends(get_milvus_context),
    ) -> documents.UploadResponse:
        """Upload and process a document to a collection (loads config from collection).

        curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
          -H "Authorization: Bearer $TOKEN" \
          -F "file=@document.pdf"
        """
        return await documents.upload_document(
            collection_name=collection_name,
            user=context.user,
            file=file,
            markdown_content=markdown_content,
            metadata_override=metadata_override,
            security_groups=security_groups,
            milvus_client=context.client,
        )

    @app.post("/v1/search")
    async def search_endpoint(
        request: search.SearchRequest,
        context: MilvusClientContext = Depends(get_milvus_context),
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
        return await search.search(request, context)

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

    @app.post("/v1/agents/{agent_name}", response_model=None)
    async def agent_endpoint(
        agent_name: str,
        request: Request,
        context: MilvusClientContext = Depends(get_milvus_context),
    ) -> StreamingResponse | ChatCompletion:
        """Agentic endpoint invoked by agent name.

        curl -X POST http://localhost:8000/v1/agents/milvuschat \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $TOKEN" \
          -d '{
            "collection": "my_collection",
            "model": "gpt-oss:20b",
            "messages": [{"role": "user", "content": "What documents discuss machine learning?"}],
            "stream": false
          }'
        """
        return await agent.create_agent_completion(
            agent_name,
            request,
            user=context.user,
            milvus_client=context.client,
        )
