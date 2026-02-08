"""Chat completions endpoint handler."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function,
)

from src.clients.router import client_router
from src.tools import tool_registry
from src.utils import map_openai_error_to_http


async def _execute_tool_calls(
    tool_calls: list[ChatCompletionMessageFunctionToolCall],
    user_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute tool calls and return tool messages.

    Args:
        tool_calls: List of function tool calls from the LLM response.
        user_metadata: Optional user metadata containing authentication token.

    Returns:
        List of tool message dictionaries to add to the conversation.
    """
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            # Parse the arguments JSON
            arguments = json.loads(tool_call.function.arguments)
            # Execute the tool with user metadata if available
            result = await tool_registry.execute_tool(tool_name, arguments, metadata=user_metadata)
            # Add tool message to conversation
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                }
            )
        except Exception as e:
            # If tool execution fails, return error message
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    return tool_messages


async def _handle_non_streaming_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_iterations: int = 5,
    user_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """Handle non-streaming completion with tool call support.

    This function will recursively handle tool calls until the LLM returns
    a final response (no tool calls) or max_iterations is reached.

    Args:
        model: The model identifier.
        messages: List of message dictionaries.
        max_iterations: Maximum number of tool call iterations (prevents infinite loops).
        **kwargs: Additional parameters for the completion request.

    Returns:
        Final ChatCompletion response.
    """
    iteration = 0
    current_messages = messages.copy()

    while iteration < max_iterations:
        iteration += 1

        # Make completion request
        completion = await client_router.create_completion(
            model=model,
            messages=current_messages,
            stream=False,
            **kwargs,
        )
        tool_calls_count = 0
        if (
            isinstance(completion, ChatCompletion)
            and completion.choices
            and completion.choices[0].message.tool_calls
        ):
            tool_calls_count = len(completion.choices[0].message.tool_calls)
        logging.info(
            "chat_tool_loop iteration=%s model=%s tool_calls=%s",
            iteration,
            model,
            tool_calls_count,
        )

        # Check if response is a ChatCompletion
        if not isinstance(completion, ChatCompletion):
            # This shouldn't happen in non-streaming mode, but handle it
            return completion  # type: ignore[return-value]

        # Get the assistant message
        assistant_message = completion.choices[0].message
        if not isinstance(assistant_message, ChatCompletionMessage):
            return completion

        # Check if there are tool calls (only function tool calls are supported)
        function_tool_calls = [tc for tc in (assistant_message.tool_calls or []) if isinstance(tc, ChatCompletionMessageFunctionToolCall)]

        if function_tool_calls:
            # Add assistant message with tool calls to conversation
            current_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in function_tool_calls
                    ],
                }
            )

            # Execute tool calls
            tool_messages = await _execute_tool_calls(function_tool_calls, user_metadata)
            current_messages.extend(tool_messages)

            # Continue loop to get final response
            continue

        # No tool calls, return the completion
        return completion

    # If we've hit max_iterations, return the last completion
    # At this point completion should be a ChatCompletion, but handle the type
    if isinstance(completion, ChatCompletion):
        return completion
    # Fallback (shouldn't happen)
    return completion  # type: ignore[return-value]


async def create_chat_completion(
    request: Request,
    user: dict[str, Any] | None = None,
) -> StreamingResponse | ChatCompletion:
    """Handle chat completion requests with streaming support.

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
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
    if "messages" not in body:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'messages'",
        )

    # Extract parameters
    model = body["model"]
    messages = body["messages"]
    stream = body.get("stream", False)
    # Extract other parameters (like temperature, etc.)
    other_params = {k: v for k, v in body.items() if k not in ("model", "messages", "stream")}

    # Prepare user metadata for tools if user is authenticated
    user_metadata = None
    if user:
        user_metadata = {
            "milvus_token": user.get("milvus_token"),
            "username": user.get("username"),
        }

    model_name = str(model).lower()
    if model_name == "milvuschat":
        if not other_params.get("token"):
            inferred_token = user_metadata.get("milvus_token") if user_metadata else None
            if inferred_token:
                other_params["token"] = inferred_token
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Milvus token is required for milvuschat",
                )

    # Add tools to the request if not explicitly provided
    # User can override by providing their own tools parameter
    if "tools" not in other_params:
        other_params["tools"] = tool_registry.get_tool_definitions()

    try:
        if stream:
            # Use router to get the appropriate client and create completion
            stream_response = await client_router.create_completion(
                model=model,
                messages=messages,
                stream=True,
                **other_params,
            )

            # Response is an AsyncIterator when streaming
            async def generate() -> AsyncGenerator[str]:
                try:
                    # Accumulate chunks to detect tool calls
                    chunks: list[ChatCompletionChunk] = []
                    has_tool_calls = False

                    async for chunk in stream_response:  # type: ignore[union-attr]
                        chunks.append(chunk)
                        # Check if this chunk contains tool calls
                        if chunk.choices and chunk.choices[0].delta.tool_calls:
                            has_tool_calls = True
                        # Stream chunks as they come
                        chunk_json = chunk.model_dump_json()
                        yield f"data: {chunk_json}\n\n"

                    # If tool calls were detected, execute them and get final response
                    if has_tool_calls:
                        # Reconstruct the assistant message from chunks
                        assistant_message_dict: dict[str, Any] = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [],
                        }
                        tool_call_index_map: dict[int, dict[str, Any]] = {}

                        for chunk in chunks:
                            if chunk.choices and chunk.choices[0].delta:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    assistant_message_dict["content"] = assistant_message_dict.get("content", "") + delta.content
                                if delta.tool_calls:
                                    for tc_delta in delta.tool_calls:
                                        if tc_delta.index is not None:
                                            idx = tc_delta.index
                                            if idx not in tool_call_index_map:
                                                tool_call_index_map[idx] = {
                                                    "id": tc_delta.id or "",
                                                    "type": tc_delta.type or "function",
                                                    "function": {"name": "", "arguments": ""},
                                                }
                                            if tc_delta.function:
                                                if tc_delta.function.name:
                                                    tool_call_index_map[idx]["function"]["name"] = tc_delta.function.name
                                                if tc_delta.function.arguments:
                                                    tool_call_index_map[idx]["function"]["arguments"] += tc_delta.function.arguments

                        # Convert tool calls to list
                        if tool_call_index_map:
                            assistant_message_dict["tool_calls"] = list(tool_call_index_map.values())
                            current_messages = messages.copy()
                            current_messages.append(assistant_message_dict)

                            # Execute tool calls
                            tool_calls_list: list[ChatCompletionMessageFunctionToolCall] = []
                            for tc_dict in assistant_message_dict["tool_calls"]:
                                tool_calls_list.append(
                                    ChatCompletionMessageFunctionToolCall(
                                        id=tc_dict["id"],
                                        type="function",
                                        function=Function(
                                            name=tc_dict["function"]["name"],
                                            arguments=tc_dict["function"]["arguments"],
                                        ),
                                    )
                                )

                            tool_messages = await _execute_tool_calls(tool_calls_list, user_metadata)
                            current_messages.extend(tool_messages)

                            # Get final response (streaming)
                            final_stream = await client_router.create_completion(
                                model=model,
                                messages=current_messages,
                                stream=True,
                                **other_params,
                            )

                            # Stream the final response
                            async for chunk in final_stream:  # type: ignore[union-attr]
                                chunk_json = chunk.model_dump_json()
                                yield f"data: {chunk_json}\n\n"

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    # If streaming fails, send an error in SSE format
                    error_msg = str(e) if str(e) else "Stream error occurred"
                    error_obj = {
                        "error": {
                            "message": error_msg,
                            "type": "stream_error",
                        }
                    }
                    error_json = json.dumps(error_obj)
                    yield f"data: {error_json}\n\n"
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
            # Non-streaming: handle tool calls by executing and re-querying
            if model_name == "milvuschat":
                return await client_router.create_completion(
                    model=model,
                    messages=messages,
                    stream=False,
                    **other_params,
                )
            return await _handle_non_streaming_completion(
                model=model,
                messages=messages,
                user_metadata=user_metadata,
                **other_params,
            )
    except APIError as e:
        raise map_openai_error_to_http(e) from e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e
