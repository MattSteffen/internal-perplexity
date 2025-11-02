"""Direct tool calling endpoint handler."""

from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel

from src.tools import tool_registry


class ToolCallRequest(BaseModel):
    """Request model for direct tool calling."""

    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] | None = None


class ToolCallResponse(BaseModel):
    """Response model for tool execution."""

    result: str


async def call_tool(request: Request) -> ToolCallResponse:
    """Handle direct tool calling requests.

    curl -X POST http://localhost:8000/v1/tools \
      -H "Content-Type: application/json" \
      -d '{
        "name": "get_weather",
        "arguments": {"location": "San Francisco, CA"},
        "metadata": {"user_id": "user123"}
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
    if "name" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'name'",
        )
    if "arguments" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'arguments'",
        )

    # Extract parameters
    tool_name = body["name"]
    arguments = body["arguments"]
    metadata = body.get("metadata")

    # Validate arguments is a dict
    if not isinstance(arguments, dict):
        raise HTTPException(
            status_code=400,
            detail="Field 'arguments' must be a dictionary",
        )

    # Validate metadata is a dict if provided
    if metadata is not None and not isinstance(metadata, dict):
        raise HTTPException(
            status_code=400,
            detail="Field 'metadata' must be a dictionary if provided",
        )

    try:
        # Execute the tool with metadata
        result = await tool_registry.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            metadata=metadata,
        )
        return ToolCallResponse(result=result)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution error: {str(e)}",
        ) from e
