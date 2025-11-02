"""Embeddings endpoint handler."""

from typing import Any

from fastapi import HTTPException, Request
from openai import APIError, AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from src.client import ollama_client
from src.utils import map_openai_error_to_http


async def create_embedding(
    request: Request,
    client: AsyncOpenAI = ollama_client,
) -> CreateEmbeddingResponse:
    """Handle embedding requests.

    curl -X POST http://localhost:8000/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "model": "all-minilm:v2",
        "input": "The food was delicious"
      }'
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}",
        ) from e

    # Validate required fields
    if "model" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'model'",
        )
    if "input" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'input'",
        )

    try:
        # Create embedding using OpenAI client
        response: CreateEmbeddingResponse = await client.embeddings.create(**body)
        return response
    except APIError as e:
        raise map_openai_error_to_http(e) from e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e
