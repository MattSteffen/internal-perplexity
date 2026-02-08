"""Models endpoint handler."""

import time
from typing import Any

import httpx
from fastapi import HTTPException, Request
from openai.types import Model
from pydantic import BaseModel

from src.agents import agent_registry
from src.config import settings


class ModelList(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[Model]


async def list_models(request: Request) -> ModelList:
    """Handle model listing requests.

    Fetches models from Ollama's native API and formats them as OpenAI-compatible response.

    curl -X GET http://localhost:8000/v1/models
    """
    # Extract base URL from Ollama base URL (remove /v1 suffix if present)
    ollama_base = settings.ollama_base_url.removesuffix("/v1")
    ollama_base = ollama_base.removesuffix("/")

    # Call Ollama's native /api/tags endpoint
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{ollama_base}/api/tags")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch models from Ollama: {str(e)}",
            ) from e

    # Format response to OpenAI-compatible format
    models: list[Model] = []
    current_time = int(time.time())

    # Add Ollama models
    for model_info in data.get("models", []):
        model_data: dict[str, Any] = {
            "id": model_info.get("name", ""),
            "object": "model",
            "created": current_time,
            "owned_by": "ollama",
        }
        models.append(Model(**model_data))

    # Add custom agent models
    for agent_name in agent_registry.list_agents():
        models.append(
            Model(
                id=agent_name,
                object="model",
                created=current_time,
                owned_by="custom",
            )
        )

    return ModelList(data=models, object="list")
