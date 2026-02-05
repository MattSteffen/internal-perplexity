"""Agent endpoint handler for MilvusChat."""

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from pymilvus import MilvusClient  # type: ignore

from src.clients.milvuschat import milvuschat_client

logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    """Request model for agent endpoint."""

    model: str = Field(default="milvuschat", description="Agent model to use")
    collection: str = Field(..., min_length=1, description="Name of the Milvus collection to connect to")
    messages: list[dict[str, Any]] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Whether to stream the response")
    llm_model: str | None = Field(None, description="Underlying LLM model (defaults to config)")


async def create_agent_completion(
    request: Request,
    user: dict[str, Any] | None = None,
    milvus_client: MilvusClient | None = None,
) -> StreamingResponse | ChatCompletion:
    """Handle agent completion requests.

    The agent endpoint creates an agentic RAG conversation bound to a specific
    Milvus collection. It uses the collection's llm_prompt for system prompt
    generation and performs semantic search as needed.

    curl -X POST http://localhost:8000/v1/agent \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "model": "milvuschat",
        "collection": "my_collection",
        "messages": [{"role": "user", "content": "What documents discuss machine learning?"}],
        "stream": false
      }'

    Args:
        request: FastAPI request object containing JSON body
        user: Optional user dict with authentication info

    Returns:
        StreamingResponse for streaming or ChatCompletion for non-streaming

    Raises:
        HTTPException: For various error conditions
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}",
        ) from e

    # Validate required fields
    if "collection" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'collection'",
        )
    if "messages" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'messages'",
        )

    # Extract parameters
    model = body.get("model", "milvuschat")
    collection = body["collection"]
    messages = body["messages"]
    stream = body.get("stream", False)
    llm_model = body.get("llm_model")

    # Get token from user if authenticated
    token: str | None = None
    if user:
        token = user.get("milvus_token")

    # Validate token is present for authenticated operations
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Milvus token is required. Please authenticate.",
        )
    if milvus_client is None:
        raise HTTPException(
            status_code=500,
            detail="Milvus client dependency is missing",
        )

    try:
        # Create completion using MilvusChat client
        kwargs: dict[str, Any] = {}
        if llm_model:
            kwargs["llm_model"] = llm_model

        result = await milvuschat_client.create_completion(
            model=model,
            messages=messages,
            stream=stream,
            collection=collection,
            token=token,
            milvus_client=milvus_client,
            **kwargs,
        )

        if stream:
            # Return streaming response
            async def generate():
                try:
                    async for chunk in result:
                        chunk_json = chunk.model_dump_json()
                        yield f"data: {chunk_json}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_msg = str(e) if str(e) else "Stream error occurred"
                    import json

                    error_obj = {
                        "error": {
                            "message": error_msg,
                            "type": "stream_error",
                        }
                    }
                    yield f"data: {json.dumps(error_obj)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception(f"Agent completion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent completion failed: {str(e)}",
        ) from e
