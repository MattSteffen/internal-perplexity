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

You have access to a document database through the `milvus_search` tool.

**Collection Info**:
{collection_info}

## Available Tools

### milvus_search Tool

Retrieves relevant documents from the collection.

**Parameters**:
- `queries` (list[str], optional): Semantic search terms for finding conceptually similar content
- `filters` (list[str], optional): Milvus filter expressions for metadata-based filtering
- `collection_name` (string, optional): Defaults to the current collection

**Returns**: List of matching document chunks with metadata

## Instructions

### Step 1: Search Strategy

**Semantic Search** (use `queries`):
- User asks about concepts, topics, or content
- Example: "What papers discuss machine learning?"

**Metadata Filtering** (use `filters`):
- User asks for documents by author, date, or title
- Example: "Show me all documents by Dr. Smith from 2023"

**Hybrid Search** (use both):
- Combines topical and metadata requirements

### Step 2: Filter Construction

{filter_instructions}

### Step 3: Respond

**Content questions**: Synthesize information from retrieved documents
**Metadata queries**: Be concise and direct
**Information not found**: State explicitly that you couldn't find relevant documents

## Constraints

1. **Document-grounded only**: All responses must come from retrieved documents
2. **No external knowledge**: Never supplement with information outside the collection
3. **Source attribution**: Reference which documents informed your response
"""

MILVUSCHAT_MULTI_COLLECTION_PROMPT_TEMPLATE = """# Role and Task

You are a specialized document retrieval assistant with access to multiple Milvus collections.
Your task is to select the most relevant collection and answer using documents from that collection only.

## Available Collections

{available_collections}

## Available Tools

### milvus_search Tool

Retrieves relevant documents from the specified collection.

**Parameters**:
- `queries` (list[str], optional): Semantic search terms for finding conceptually similar content
- `filters` (list[str], optional): Milvus filter expressions for metadata-based filtering
- `collection_name` (string, required): The collection you choose from the list above

**Returns**: List of matching document chunks with metadata

## Instructions

1. Pick the most relevant collection for the user question.
2. Call `milvus_search` with `collection_name` and `queries`/`filters`.
3. If you get no relevant results, try one other likely collection.
4. Answer only from retrieved documents.

### Filter Construction

{filter_instructions}

## Constraints

1. **Document-grounded only**: All responses must come from retrieved documents
2. **No external knowledge**: Never supplement with information outside the collections
3. **Source attribution**: Reference which documents informed your response
"""

MILVUSCHAT_COLLECTION_SELECTOR_PROMPT_TEMPLATE = """# Role

You are a routing assistant. Your job is to pick the single best Milvus collection for the user question.

## Available Collections

{available_collections}

## Instructions

1. Choose the single best collection from the list above.
2. Provide up to two fallback collections (different from the primary).
3. Provide 1-3 short query phrases that would work well for semantic search.
4. If you are unsure, choose the closest match rather than refusing.

## Output

Return a JSON object only. No prose, no markdown, no code fences.

{{
  "collection_name": "name_from_list",
  "fallbacks": ["name_from_list", "name_from_list"],
  "reason": "brief reason",
  "first_queries": ["short query", "short query"]
}}
"""

DEFAULT_FILTER_INSTRUCTIONS = """
- Strings in the filters array must escape internal double quotes
- Example: "json_contains(metadata[\\"author\\"], \\"Jane Doe\\")"
- Date is typically an integer year (e.g., 2023)
- Use `json_contains` for array fields in metadata
"""


TOOL_NAME = "milvus_search"
MAX_TOOL_DOCS = 6
MAX_TOOL_DOC_CHARS = 4000
MAX_COLLECTIONS_IN_PROMPT = 60
MAX_COLLECTION_DESCRIPTION_CHARS = 240
MAX_COLLECTION_PROMPT_CHARS = 240


def build_single_collection_prompt(
    collection_name: str,
    collection_info: str,
    filter_instructions: str | None = None,
) -> str:
    """Build the system prompt for single-collection MilvusChat."""
    return MILVUSCHAT_SYSTEM_PROMPT_TEMPLATE.format(
        collection_name=collection_name,
        collection_info=collection_info,
        filter_instructions=filter_instructions or DEFAULT_FILTER_INSTRUCTIONS,
    )


def build_multi_collection_prompt(
    available_collections: str,
    filter_instructions: str | None = None,
) -> str:
    """Build the system prompt for multi-collection MilvusChat."""
    return MILVUSCHAT_MULTI_COLLECTION_PROMPT_TEMPLATE.format(
        available_collections=available_collections,
        filter_instructions=filter_instructions or DEFAULT_FILTER_INSTRUCTIONS,
    )


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to a maximum number of characters with a marker."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    return f"{truncated}\n\n[truncated]"


def _render_documents_for_tool(documents: list[Any]) -> str:
    """Render and truncate documents for tool responses."""
    if not documents:
        return "No documents found"
    rendered_docs: list[str] = []
    for doc in documents[:MAX_TOOL_DOCS]:
        rendered = render_document(doc, include_text=True)
        rendered_docs.append(_truncate_text(rendered, MAX_TOOL_DOC_CHARS))
    return "\n\n".join(rendered_docs)


def _format_collection_catalog(entries: list[str]) -> str:
    """Format available collections section for the system prompt."""
    if not entries:
        return "No collections are available for this token."
    return "\n".join(entries)


def _build_extra_body() -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    if radchat_config.ollama.context_length:
        extra_body["num_ctx"] = radchat_config.ollama.context_length
    return extra_body


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _extract_json(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def _format_catalog_entry(
    name: str,
    description: str,
    metadata_fields: str,
    llm_prompt: str,
) -> str:
    desc = _truncate_text(description, MAX_COLLECTION_DESCRIPTION_CHARS).replace("\n", " ").strip()
    prompt_hint = _truncate_text(llm_prompt, MAX_COLLECTION_PROMPT_CHARS).replace("\n", " ").strip()
    entry = f"- {name}: {desc} (metadata: {metadata_fields})"
    if prompt_hint:
        entry = f"{entry} | prompt_hint: {prompt_hint}"
    return entry


class MilvusChatClient:
    """MilvusChat agent that connects to a specified collection.

    Unlike the hardcoded RadChat, MilvusChat dynamically connects to any
    collection and uses the collection's llm_prompt for system prompt generation.
    """

    def __init__(self) -> None:
        """Initialize the MilvusChat client."""
        self._client: MilvusClient | None = None
        self._collection_name: str | None = None

    def _load_collection_context(
        self,
        milvus_client: MilvusClient,
        collection_name: str,
    ) -> tuple[str, str]:
        if not milvus_client.has_collection(collection_name=collection_name):
            raise CollectionNotFoundError(collection_name)
        milvus_client.load_collection(collection_name=collection_name)

        collection_info = f"Collection: {collection_name}"
        llm_prompt = ""
        try:
            from crawler.vector_db import CollectionDescription

            collection_info_dict = milvus_client.describe_collection(collection_name)
            description_str = collection_info_dict.get("description", "")
            collection_desc = CollectionDescription.from_json(description_str) if description_str else None

            if collection_desc:
                llm_prompt = collection_desc.llm_prompt or ""
                collection_info = collection_desc.description or collection_info
        except Exception as e:
            logger.warning("Failed to get collection description: %s", e)

        return collection_info, llm_prompt

    def _build_collection_catalog(
        self,
        milvus_client: MilvusClient,
    ) -> tuple[list[str], list[str]]:
        try:
            from crawler.vector_db import CollectionDescription

            collection_names = milvus_client.list_collections()
            entries: list[str] = []
            for name in collection_names[:MAX_COLLECTIONS_IN_PROMPT]:
                description = "Description unavailable"
                metadata_fields = "metadata fields unknown"
                llm_prompt = ""
                try:
                    info = milvus_client.describe_collection(name)
                    description_str = info.get("description", "")
                    collection_desc = CollectionDescription.from_json(description_str) if description_str else None
                    if collection_desc:
                        description_val = collection_desc.description.strip()
                        description = description_val or "No description provided"
                        metadata_keys = list((collection_desc.metadata_schema or {}).keys())
                        metadata_fields = ", ".join(metadata_keys) if metadata_keys else "metadata fields not specified"
                        llm_prompt = collection_desc.llm_prompt or ""
                except Exception as e:
                    logger.warning("Failed to describe collection '%s': %s", name, e)
                entries.append(_format_catalog_entry(name, description, metadata_fields, llm_prompt))
            return entries, collection_names
        except Exception as e:
            logger.warning("Failed to list collections for prompt: %s", e)
            return [], []

    async def _select_collection(
        self,
        client: AsyncOpenAI,
        llm_model: str,
        messages: list[dict[str, Any]],
        available_collections_block: str,
        collection_names: list[str],
        invocation_id: str,
    ) -> tuple[str | None, list[str], list[str]]:
        user_question = _extract_last_user_message(messages)
        selection_prompt = MILVUSCHAT_COLLECTION_SELECTOR_PROMPT_TEMPLATE.format(
            available_collections=available_collections_block,
        )
        selection_messages = [
            {"role": "system", "content": selection_prompt},
            {"role": "user", "content": user_question or "Select the best collection."},
        ]

        logger.info(
            "milvuschat_llm_call invocation_id=%s phase=selector messages=%s tools_enabled=false",
            invocation_id,
            len(selection_messages),
        )
        response = await client.chat.completions.create(
            model=llm_model,
            messages=selection_messages,
            stream=False,
            extra_body=_build_extra_body(),
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        if not parsed:
            logger.warning(
                "milvuschat_collection_selected invocation_id=%s selected=none fallbacks=none reason=invalid_json",
                invocation_id,
            )
            return (collection_names[0] if collection_names else None), [], []

        selected = str(parsed.get("collection_name") or "").strip()
        if selected not in collection_names:
            selected = collection_names[0] if collection_names else ""

        fallbacks_raw = parsed.get("fallbacks") or []
        if isinstance(fallbacks_raw, str):
            fallbacks = [fallbacks_raw]
        else:
            fallbacks = [str(item) for item in fallbacks_raw if item]
        fallbacks = [name for name in fallbacks if name in collection_names and name != selected]
        fallbacks = fallbacks[:2]

        queries_raw = parsed.get("first_queries") or []
        if isinstance(queries_raw, str):
            first_queries = [queries_raw]
        else:
            first_queries = [str(item) for item in queries_raw if item]

        logger.info(
            "milvuschat_collection_selected invocation_id=%s selected=%s fallbacks=%s",
            invocation_id,
            selected or "none",
            ",".join(fallbacks) if fallbacks else "none",
        )
        return selected or None, fallbacks, first_queries

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
        if not token:
            raise ValueError("token parameter is required for MilvusChat")

        invocation_id = str(uuid.uuid4())
        logger.info(
            "milvuschat_start invocation_id=%s stream=%s has_collection=%s num_messages=%s",
            invocation_id,
            stream,
            bool(collection),
            len(messages),
        )

        # The chat endpoint may inject a global `tools` list into kwargs.
        # MilvusChat intentionally restricts tools to only its `milvus_search` tool,
        # and we pass that explicitly to the internal completion methods.
        # Remove any incoming `tools` to avoid "multiple values for keyword" errors.
        kwargs.pop("tools", None)

        if milvus_client is None:
            milvus_client = MilvusClient(uri=get_milvus_uri(), token=token)

        fallback_collections: list[str] = []
        first_queries: list[str] = []
        if not collection:
            entries, collection_names = self._build_collection_catalog(milvus_client)
            available_collections_block = _format_collection_catalog(entries)
            if collection_names and len(collection_names) > MAX_COLLECTIONS_IN_PROMPT:
                available_collections_block += "\n- ... (more collections omitted)"
            logger.info(
                "milvuschat_catalog_built invocation_id=%s collections_seen=%s catalog_chars=%s",
                invocation_id,
                len(collection_names),
                len(available_collections_block),
            )

            base_url = radchat_config.ollama.base_url
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            selector_client = AsyncOpenAI(
                base_url=base_url,
                api_key=settings.api_key,
                timeout=radchat_config.ollama.request_timeout,
            )
            llm_model = kwargs.get("llm_model", radchat_config.ollama.llm_model)
            collection, fallback_collections, first_queries = await self._select_collection(
                client=selector_client,
                llm_model=llm_model,
                messages=messages,
                available_collections_block=available_collections_block,
                collection_names=collection_names,
                invocation_id=invocation_id,
            )
            logger.info(
                "milvuschat_collection_routing invocation_id=%s selected=%s fallbacks=%s first_queries=%s",
                invocation_id,
                collection or "none",
                ",".join(fallback_collections) if fallback_collections else "none",
                "; ".join(first_queries) if first_queries else "none",
            )

        if not collection:
            error_msg = "No Milvus collections are available for selection."
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

        try:
            collection_info, llm_prompt = self._load_collection_context(milvus_client, collection)
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

        initial_results: list[Any] = []
        if collection and first_queries:
            try:
                search_results = search(
                    queries=first_queries,
                    collection_name=collection,
                    token=token,
                )
                initial_results = consolidate_documents(search_results)
                logger.info(
                    "milvuschat_presearch invocation_id=%s collection=%s queries=%s documents=%s",
                    invocation_id,
                    collection,
                    "; ".join(first_queries),
                    len(initial_results),
                )
            except Exception as e:
                logger.warning(
                    "milvuschat_presearch_failed invocation_id=%s collection=%s error=%s",
                    invocation_id,
                    collection,
                    e,
                )

        # Build system prompt
        system_prompt = build_single_collection_prompt(
            collection_name=collection,
            collection_info=collection_info,
            filter_instructions=llm_prompt if llm_prompt else None,
        )

        # Prepare messages with system prompt
        if messages and messages[0].get("role") == "system":
            # Replace existing system prompt
            all_messages = [{"role": "system", "content": system_prompt}] + messages[1:]
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        if initial_results and first_queries:
            rendered = _render_documents_for_tool(initial_results)
            tool_call_id = f"presearch-{invocation_id[:8]}"
            all_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps(
                                    {
                                        "queries": first_queries,
                                        "collection_name": collection,
                                    }
                                ),
                            },
                        }
                    ],
                }
            )
            all_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": rendered,
                    "name": TOOL_NAME,
                }
            )

        search_tool_schema = tool_registry.get_tool_definition(TOOL_NAME)

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
                fallback_collections=fallback_collections,
                invocation_id=invocation_id,
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
                fallback_collections=fallback_collections,
                invocation_id=invocation_id,
                **kwargs,
            )

    async def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        milvus_client: MilvusClient,
        collection: str | None,
        token: str | None,
        initial_results: list[Any],
        fallback_collections: list[str] | None,
        invocation_id: str,
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
        active_collection = collection
        fallback_queue = list((fallback_collections or [])[:1])
        extra_body = _build_extra_body()
        had_tool_calls_last_iteration = False

        for iteration in range(radchat_config.agent.max_tool_calls):
            logger.info(
                "milvuschat_llm_call invocation_id=%s phase=rag iteration=%s messages=%s tools_enabled=true",
                invocation_id,
                iteration + 1,
                len(current_messages),
            )
            response = await client.chat.completions.create(
                model=llm_model,
                messages=current_messages,
                tools=tools,
                stream=True,
                extra_body=extra_body,
            )

            tool_calls_dict: dict[int, dict[str, Any]] = {}
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None

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
                had_tool_calls_last_iteration = False
                break

            had_tool_calls_last_iteration = True

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

                if function_name == TOOL_NAME:
                    try:
                        logger.info(
                            "milvuschat_tool_call invocation_id=%s tool_name=%s",
                            invocation_id,
                            function_name,
                        )
                        text = function_args.get("text")
                        queries = function_args.get("queries", [])
                        filters = function_args.get("filters", [])
                        collection_name = function_args.get("collection_name") or active_collection
                        if not collection_name:
                            raise ValueError("collection_name is required for milvus_search")
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
                        if not consolidated and fallback_queue and collection_name == active_collection:
                            fallback = fallback_queue.pop(0)
                            logger.info(
                                "milvuschat_fallback_collection invocation_id=%s from_collection=%s to_collection=%s reason=no_documents",
                                invocation_id,
                                active_collection,
                                fallback,
                            )
                            active_collection = fallback
                            try:
                                collection_info, llm_prompt = self._load_collection_context(
                                    milvus_client, active_collection
                                )
                                current_messages[0] = {
                                    "role": "system",
                                    "content": build_single_collection_prompt(
                                        collection_name=active_collection,
                                        collection_info=collection_info,
                                        filter_instructions=llm_prompt if llm_prompt else None,
                                    ),
                                }
                            except Exception as e:
                                logger.warning(
                                    "milvuschat_fallback_collection invocation_id=%s error=%s",
                                    invocation_id,
                                    e,
                                )
                            results = search(
                                text=text,
                                queries=queries,
                                filters=filters,
                                collection_name=active_collection,
                                partition_name=partition_name,
                                token=token,
                            )
                            consolidated = consolidate_documents(results)

                        all_sources.extend(consolidated)

                        rendered = _render_documents_for_tool(consolidated)
                        logger.info(
                            "milvuschat_tool_result invocation_id=%s documents=%s rendered_chars=%s",
                            invocation_id,
                            len(consolidated),
                            len(rendered),
                        )

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

        if had_tool_calls_last_iteration:
            logger.info(
                "milvuschat_llm_call invocation_id=%s phase=final iteration=%s messages=%s tools_enabled=false",
                invocation_id,
                radchat_config.agent.max_tool_calls + 1,
                len(current_messages),
            )
            final_response = await client.chat.completions.create(
                model=llm_model,
                messages=current_messages,
                stream=True,
                extra_body=extra_body,
            )
            async for chunk in final_response:
                yield chunk

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
        collection: str | None,
        token: str | None,
        initial_results: list[Any],
        fallback_collections: list[str] | None,
        invocation_id: str,
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
        active_collection = collection
        fallback_queue = list((fallback_collections or [])[:1])
        extra_body = _build_extra_body()

        for iteration in range(radchat_config.agent.max_tool_calls):
            logger.info(
                "milvuschat_llm_call invocation_id=%s phase=rag iteration=%s messages=%s tools_enabled=true",
                invocation_id,
                iteration + 1,
                len(current_messages),
            )
            response = await client.chat.completions.create(
                model=llm_model,
                messages=current_messages,
                tools=tools,
                stream=False,
                extra_body=extra_body,
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

                if function_name == TOOL_NAME:
                    try:
                        logger.info(
                            "milvuschat_tool_call invocation_id=%s tool_name=%s",
                            invocation_id,
                            function_name,
                        )
                        text = function_args.get("text")
                        queries = function_args.get("queries", [])
                        filters = function_args.get("filters", [])
                        collection_name = function_args.get("collection_name") or active_collection
                        if not collection_name:
                            raise ValueError("collection_name is required for milvus_search")
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
                        if not consolidated and fallback_queue and collection_name == active_collection:
                            fallback = fallback_queue.pop(0)
                            logger.info(
                                "milvuschat_fallback_collection invocation_id=%s from_collection=%s to_collection=%s reason=no_documents",
                                invocation_id,
                                active_collection,
                                fallback,
                            )
                            active_collection = fallback
                            try:
                                collection_info, llm_prompt = self._load_collection_context(
                                    milvus_client, active_collection
                                )
                                current_messages[0] = {
                                    "role": "system",
                                    "content": build_single_collection_prompt(
                                        collection_name=active_collection,
                                        collection_info=collection_info,
                                        filter_instructions=llm_prompt if llm_prompt else None,
                                    ),
                                }
                            except Exception as e:
                                logger.warning(
                                    "milvuschat_fallback_collection invocation_id=%s error=%s",
                                    invocation_id,
                                    e,
                                )
                            results = search(
                                text=text,
                                queries=queries,
                                filters=filters,
                                collection_name=active_collection,
                                partition_name=partition_name,
                                token=token,
                            )
                            consolidated = consolidate_documents(results)

                        all_sources.extend(consolidated)

                        rendered = _render_documents_for_tool(consolidated)
                        logger.info(
                            "milvuschat_tool_result invocation_id=%s documents=%s rendered_chars=%s",
                            invocation_id,
                            len(consolidated),
                            len(rendered),
                        )

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
            extra_body=extra_body,
        )


# Singleton instance
milvuschat_client = MilvusChatClient()
