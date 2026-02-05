"""MilvusChat agent client - connects to a specified collection with dynamic system prompts."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)
from pymilvus import MilvusClient

from src.config import radchat_config, settings
from src.milvus_client import get_milvus_uri
from src.tools import tool_registry
from src.tools.milvus_search import (
    CollectionNotFoundError,
    consolidate_documents,
    render_document,
    search,
)

logger = logging.getLogger(__name__)


# Base system prompt template - collection-specific info will be injected
MILVUSCHAT_SYSTEM_PROMPT_TEMPLATE = """# Role and Task

You are a specialized document retrieval assistant connected to the "{collection_name}" collection.
Your task is to help users find and extract information from documents in this collection.

## Context

You have access to a document database through a search tool.

**Collection Info**:
{collection_info}

**Available Documents** (sample):

<available_documents>
{database_metadata}
</available_documents>

**Preliminary Context** (based on your query):

<preliminary_context>
{preliminary_context}
</preliminary_context>

## Available Tools

### search Tool

Retrieves relevant documents from the collection.

**Parameters**:
- `queries` (list[str], optional): Semantic search terms for finding conceptually similar content
- `filters` (list[str], optional): Milvus filter expressions for metadata-based filtering

**Returns**: List of matching document chunks with metadata

## Instructions

### Step 1: Check Available Information

Before calling the search tool, check if you can answer from:
1. Preliminary context provided above
2. Previous search results in this conversation

If yes, answer directly without searching.

### Step 2: Search Strategy

**Semantic Search** (use `queries`):
- User asks about concepts, topics, or content
- Example: "What papers discuss machine learning?"

**Metadata Filtering** (use `filters`):
- User asks for documents by author, date, or title
- Example: "Show me all documents by Dr. Smith from 2023"

**Hybrid Search** (use both):
- Combines topical and metadata requirements

### Step 3: Filter Construction

{filter_instructions}

### Step 4: Respond

**Content questions**: Synthesize information from retrieved documents
**Metadata queries**: Be concise and direct
**Information not found**: State explicitly that you couldn't find relevant documents

## Constraints

1. **Document-grounded only**: All responses must come from retrieved documents
2. **No external knowledge**: Never supplement with information outside the collection
3. **Source attribution**: Reference which documents informed your response
"""

DEFAULT_FILTER_INSTRUCTIONS = """
- Strings in the filters array must escape internal double quotes
- Example: "json_contains(metadata[\\"author\\"], \\"Jane Doe\\")"
- Date is typically an integer year (e.g., 2023)
- Use `json_contains` for array fields in metadata
"""


def build_system_prompt(
    collection_name: str,
    collection_info: str,
    preliminary_context: str,
    database_metadata: str,
    filter_instructions: str | None = None,
) -> str:
    """Build the system prompt for MilvusChat agent.

    Args:
        collection_name: Name of the collection
        collection_info: Collection description or llm_prompt from CollectionDescription
        preliminary_context: Rendered preliminary search results
        database_metadata: Summary of available documents
        filter_instructions: Optional custom filter instructions from llm_prompt

    Returns:
        Complete system prompt string
    """
    return MILVUSCHAT_SYSTEM_PROMPT_TEMPLATE.format(
        collection_name=collection_name,
        collection_info=collection_info,
        preliminary_context=preliminary_context,
        database_metadata=database_metadata,
        filter_instructions=filter_instructions or DEFAULT_FILTER_INSTRUCTIONS,
    )


def get_metadata_summary(
    collection_name: str,
    token: str | None = None,
    limit: int = 50,
) -> list[str]:
    """Get a summary of documents in the collection.

    Args:
        collection_name: Name of the collection
        token: Authentication token (username:password)
        limit: Maximum number of documents to summarize

    Returns:
        List of formatted metadata strings
    """
    try:
        search_results = search(text=None, filters=[], token=token, collection_name=collection_name, limit=limit)
        all_docs = consolidate_documents(search_results)[:limit]
    except Exception as e:
        logger.warning(f"Failed to get metadata summary: {e}")
        return ["Unable to retrieve document metadata"]

    summaries = []
    for doc in all_docs:
        meta = doc.metadata or {}
        summary_parts = []
        if meta.get("title"):
            summary_parts.append(f"Title: {meta['title']}")
        if meta.get("author"):
            authors = meta["author"]
            if isinstance(authors, list):
                summary_parts.append(f"Authors: {', '.join(authors[:3])}")
            else:
                summary_parts.append(f"Author: {authors}")
        if meta.get("date"):
            summary_parts.append(f"Date: {meta['date']}")
        if summary_parts:
            summaries.append("\n".join(summary_parts))

    return summaries if summaries else ["No documents found in collection"]


class MilvusChatClient:
    """MilvusChat agent that connects to a specified collection.

    Unlike the hardcoded RadChat, MilvusChat dynamically connects to any
    collection and uses the collection's llm_prompt for system prompt generation.
    """

    def __init__(self) -> None:
        """Initialize the MilvusChat client."""
        self._client: MilvusClient | None = None
        self._collection_name: str | None = None

    async def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        collection: str | None = None,
        token: str | None = None,
        milvus_client: MilvusClient | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using MilvusChat agent.

        Args:
            model: The model identifier (milvuschat or underlying LLM)
            messages: List of message dictionaries
            stream: Whether to stream the response
            collection: Name of the Milvus collection to connect to
            token: Milvus authentication token (username:password)
            **kwargs: Additional parameters

        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        if not collection:
            raise ValueError("collection parameter is required for MilvusChat")

        # The chat endpoint may inject a global `tools` list into kwargs.
        # MilvusChat intentionally restricts tools to only its `search` tool,
        # and we pass that explicitly to the internal completion methods.
        # Remove any incoming `tools` to avoid "multiple values for keyword" errors.
        kwargs.pop("tools", None)

        # Connect to Milvus
        try:
            if milvus_client is None:
                milvus_client = MilvusClient(uri=get_milvus_uri(), token=token)
            if not milvus_client.has_collection(collection_name=collection):
                raise CollectionNotFoundError(collection)
            milvus_client.load_collection(collection_name=collection)
        except CollectionNotFoundError as e:
            error_msg = f"Collection '{e.collection_name}' not found"
            if stream:

                async def error_stream() -> AsyncIterator[ChatCompletionChunk]:
                    yield ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[
                            {  # type: ignore
                                "index": 0,
                                "delta": {"content": f"Error: {error_msg}"},
                                "finish_reason": "error",
                            }
                        ],
                    )

                return error_stream()
            else:
                return ChatCompletion(
                    id=str(uuid.uuid4()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        {  # type: ignore
                            "index": 0,
                            "message": ChatCompletionMessage(
                                role="assistant",
                                content=f"Error: {error_msg}",
                            ),
                            "finish_reason": "error",
                        }
                    ],
                )

        # Get collection description for llm_prompt
        try:
            from crawler.vector_db import CollectionDescription

            collection_info_dict = milvus_client.describe_collection(collection)
            description_str = collection_info_dict.get("description", "")
            collection_desc = CollectionDescription.from_json(description_str) if description_str else None

            # Extract llm_prompt if available
            llm_prompt = ""
            if collection_desc:
                llm_prompt = collection_desc.llm_prompt or ""
                collection_info = collection_desc.description or f"Collection: {collection}"
            else:
                collection_info = f"Collection: {collection}"
        except Exception as e:
            logger.warning(f"Failed to get collection description: {e}")
            llm_prompt = ""
            collection_info = f"Collection: {collection}"

        # Get user query from messages
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Perform initial search
        initial_results: list[Any] = []
        if user_query:
            try:
                search_results = search(
                    text=user_query,
                    queries=[user_query],
                    filters=[],
                    collection_name=collection,
                    token=token,
                )
                initial_results = consolidate_documents(search_results)
            except Exception as e:
                logger.warning(f"Initial search failed: {e}")

        # Build preliminary context
        preliminary_context = "\n\n".join([render_document(doc, include_text=True) for doc in initial_results[:5]]) if initial_results else "No preliminary results found."

        # Get database metadata summary
        metadata_list = get_metadata_summary(collection, token)
        database_metadata = "\n\n".join(metadata_list)

        # Build system prompt
        system_prompt = build_system_prompt(
            collection_name=collection,
            collection_info=collection_info,
            preliminary_context=preliminary_context,
            database_metadata=database_metadata,
            filter_instructions=llm_prompt if llm_prompt else None,
        )

        # Prepare messages with system prompt
        if messages and messages[0].get("role") == "system":
            # Replace existing system prompt
            all_messages = [{"role": "system", "content": system_prompt}] + messages[1:]
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        search_tool_definitions = [
            tool_def
            for tool_def in tool_registry.get_tool_definitions()
            if tool_def.get("function", {}).get("name") == "search"
        ]
        if not search_tool_definitions:
            raise RuntimeError("Search tool definition not found in tool registry")
        search_tool_schema = search_tool_definitions[0]

        # Create completion with agentic loop
        if stream:
            return self._stream_completion(
                model=model,
                messages=all_messages,
                tools=[search_tool_schema],
                milvus_client=milvus_client,
                collection=collection,
                token=token,
                initial_results=initial_results,
                **kwargs,
            )
        else:
            return await self._non_streaming_completion(
                model=model,
                messages=all_messages,
                tools=[search_tool_schema],
                milvus_client=milvus_client,
                collection=collection,
                token=token,
                initial_results=initial_results,
                **kwargs,
            )

    async def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        milvus_client: MilvusClient,
        collection: str,
        token: str | None,
        initial_results: list[Any],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion with tool calling."""
        # Use underlying LLM model
        llm_model = kwargs.pop("llm_model", radchat_config.ollama.llm_model)

        base_url = radchat_config.ollama.base_url
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=settings.api_key,
            timeout=radchat_config.ollama.request_timeout,
        )

        all_sources = initial_results.copy()
        current_messages = messages.copy()
        final_content = ""

        for iteration in range(radchat_config.agent.max_tool_calls):
            response = await client.chat.completions.create(
                model=llm_model,
                messages=current_messages,
                tools=tools,
                stream=True,
            )

            tool_calls_dict: dict[int, dict[str, Any]] = {}
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None

                if choice and choice.delta.content:
                    final_content += choice.delta.content

                if choice and choice.delta.tool_calls:
                    for tc_delta in choice.delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.function.name:
                            tool_calls_dict[idx]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_dict[idx]["function"]["arguments"] += tc_delta.function.arguments
                else:
                    yield chunk

                if choice and choice.finish_reason == "tool_calls":
                    break

            # Process tool calls if any
            tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]

            if not tool_calls:
                break

            # Add assistant message with tool calls
            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            )

            # Execute tool calls
            for tool in tool_calls:
                function_name = tool["function"]["name"]
                try:
                    function_args = json.loads(tool["function"]["arguments"])
                except json.JSONDecodeError:
                    function_args = {}

                if function_name == "search":
                    try:
                        text = function_args.get("text")
                        queries = function_args.get("queries", [])
                        filters = function_args.get("filters", [])
                        collection_name = function_args.get("collection_name") or collection
                        partition_name = function_args.get("partition_name")
                        results = search(
                            text=text,
                            queries=queries,
                            filters=filters,
                            collection_name=collection_name,
                            partition_name=partition_name,
                            token=token,
                        )
                        consolidated = consolidate_documents(results)
                        all_sources.extend(consolidated)

                        rendered = "\n\n".join([render_document(d, include_text=True) for d in consolidated]) if consolidated else "No documents found"

                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool.get("id") or "",
                                "content": rendered,
                                "name": function_name,
                            }
                        )
                    except Exception as e:
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool.get("id") or "",
                                "content": f"Search error: {str(e)}",
                                "name": function_name,
                            }
                        )
                else:
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool.get("id") or "",
                            "content": f"Unknown tool: {function_name}",
                            "name": function_name,
                        }
                    )

        # Yield final chunk with done signal
        yield ChatCompletionChunk(
            id=str(uuid.uuid4()),
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                {  # type: ignore
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        )

    async def _non_streaming_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        milvus_client: MilvusClient,
        collection: str,
        token: str | None,
        initial_results: list[Any],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Handle non-streaming completion with tool calling."""
        llm_model = kwargs.pop("llm_model", radchat_config.ollama.llm_model)

        base_url = radchat_config.ollama.base_url
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=settings.api_key,
            timeout=radchat_config.ollama.request_timeout,
        )

        all_sources = initial_results.copy()
        current_messages = messages.copy()

        for iteration in range(radchat_config.agent.max_tool_calls):
            response = await client.chat.completions.create(
                model=llm_model,
                messages=current_messages,
                tools=tools,
                stream=False,
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                return response

            # Add assistant message with tool calls
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
                        for tc in tool_calls
                    ],
                }
            )

            # Execute tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_args = {}

                if function_name == "search":
                    try:
                        text = function_args.get("text")
                        queries = function_args.get("queries", [])
                        filters = function_args.get("filters", [])
                        collection_name = function_args.get("collection_name") or collection
                        partition_name = function_args.get("partition_name")
                        results = search(
                            text=text,
                            queries=queries,
                            filters=filters,
                            collection_name=collection_name,
                            partition_name=partition_name,
                            token=token,
                        )
                        consolidated = consolidate_documents(results)
                        all_sources.extend(consolidated)

                        rendered = "\n\n".join([render_document(d, include_text=True) for d in consolidated]) if consolidated else "No documents found"

                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": rendered,
                                "name": function_name,
                            }
                        )
                    except Exception as e:
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Search error: {str(e)}",
                                "name": function_name,
                            }
                        )
                else:
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {function_name}",
                            "name": function_name,
                        }
                    )

        # Final response after max iterations
        return await client.chat.completions.create(
            model=llm_model,
            messages=current_messages,
            stream=False,
        )


# Singleton instance
milvuschat_client = MilvusChatClient()
