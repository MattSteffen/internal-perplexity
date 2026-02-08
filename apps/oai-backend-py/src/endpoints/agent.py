"""Agent endpoint handler."""

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion
from pymilvus import MilvusClient  # type: ignore

from src.agents import agent_registry

logger = logging.getLogger(__name__)


async def create_agent_completion(
    agent_name: str,
    request: Request,
    user: dict[str, Any] | None = None,
    milvus_client: MilvusClient | None = None,
) -> StreamingResponse | ChatCompletion:
    """Handle agent completion requests.

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
    try:
        body: dict[str, Any] = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}",
        ) from e

    if "messages" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'messages'",
        )

    try:
        return await agent_registry.create_completion(
            agent_name=agent_name,
            body=body,
            user=user,
            milvus_client=milvus_client,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent completion error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Agent completion failed: {str(e)}",
        ) from e
