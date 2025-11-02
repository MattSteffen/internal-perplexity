"""RadChat custom agent client implementation with Milvus integration."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)
from pymilvus import MilvusClient

from src.config import UserValves, radchat_config, settings
from src.tools.milvus_search import (
    CollectionNotFoundError,
    MilvusDocument,
    build_citations,
    connect_milvus,
    consolidate_documents,
    perform_query,
    perform_search,
    render_document,
)

# System prompt for the Radchat agent
SYSTEM_PROMPT = """# Role and Task

You are a specialized document retrieval assistant. Your task is to help users find and
extract information from an internal research and development (IRAD) document collection
covering signal processing, AI, and ML topics.

## Context

You have access to a document database through a search tool. The database contains:

**Database Schema**:

<database_schema>
<<database_schema>>

</database_schema>

**Available Documents** (comprehensive sample):

<available_documents>
<<database_metadata>>

</available_documents>

**Preliminary Context**:

<preliminary_context>
<<preliminary_context>>

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
3. Document metadata shown above

If yes, answer directly without searching.

### Step 2: Decide on Search Strategy

**Semantic Search** (use `queries`):
- User asks about concepts, topics, or content
- Example: "What papers discuss transformer architectures?"

**Metadata Filtering** (use `filters`):
- User asks for documents by author, date, or title
- Example: "Show me all documents by Dr. Smith from 2023"

**Hybrid Search** (use both):
- Combines topical and metadata requirements
- Example: "Find recent papers about neural networks by Dr. Reed"

### Step 3: Construct Filter Expressions

System tips for constructing filters for the search tool

- The metadata field is a JSON field. Access keys with JSON path syntax: metadata["key"]

- Join conditions by adding multiple strings to filters. The system ANDs them together:
  - filters = ["cond1", "cond2"] becomes "cond1 and cond2"
  - If you need OR, put it inside a single string with parentheses: "(condA or condB)"

- Do not include security_group filters; they are added automatically.

- Do not use array_contains* on metadata. Use json_contains* for metadata arrays.

- JSON escaping is required inside the tool call:
  - Strings in the filters array must escape all internal double quotes with backslashes
  - Example of correct escaping:
    - "json_contains(metadata[\"author\"], \"Jane Doe\")"
    - "metadata[\"date\"] >= 2020"
    - "metadata[\"title\"] LIKE \"%graph%\""
  - Incorrect (will fail): "json_contains(metadata["author"], \"Jane Doe\")"

Schema-aligned examples

- Author (array of strings):
  - "json_contains(metadata[\"author\"], \"John Doe\")"
  - "json_contains_any(metadata[\"author\"], [\"John Doe\", \"Jane Smith\"])"
- Date (integer year):
  - "metadata[\"date\"] >= 2021"
  - "metadata[\"date\"] IN [2020, 2021, 2022]"
- Keywords (array of strings):
  - "json_contains(metadata[\"keywords\"], \"machine learning\")"
  - "json_contains_any(metadata[\"keywords\"], [\"AI\", \"ML\"])"
- Title/description/summaries (strings):
  - "metadata[\"title\"] LIKE \"%Analysis%\""

**Important**:
- Use `array_contains*` for array fields (author, keywords)
- Date is an integer year (e.g., 2023), not a date string
- Always use double quotes for string values

### Step 4: Synthesize and Respond

**When answering content questions** (e.g., "What are the key findings?"):
- Synthesize information from retrieved documents into clear, coherent answers
- Quote specific passages when directly relevant
- Include metadata (author, title, date) when it adds context
- If documents partially answer, state what's covered and what's missing

**When answering metadata queries** (e.g., "How many papers by Dr. Smith?"):
- Be concise and direct
- List titles/counts without unnecessary elaboration
- Provide details only if requested

**When information is not found**:
- State explicitly: "I could not find any documents in the collection that address [topic]"
- Do not speculate or use external knowledge
- Suggest alternative search terms if appropriate

## Constraints

1. **Document-grounded only**: All responses must come from retrieved documents
2. **No external knowledge**: Never supplement with information outside the collection
3. **Explicit limitations**: Clearly state when information isn't available
4. **Source attribution**: Reference which documents informed your response

## Examples

**User**: "What are the latest advancements in AI for signal processing?"
**Action**: `search(queries=["latest advancements AI signal processing"], filters=[])`

**User**: "Show me all documents by Dr. Reed"
**Action**: `search(queries=[], filters=["json_contains(metadata[\"author\"], \"Dr. Reed\")"])`

**User**: "Find recent neural network papers from 2022 or later"
**Action**: `search(queries=["neural networks"], filters=["date >= 2022"])`

**User**: "What papers by Dr. Smith mention transformers?"
**Action**: `search(queries=["transformers"], filters=["json_contains(metadata[\"author\"], \"Dr. Smith\")"])`

**User**: "How many IRADs did Dr. Johnson write?"
**Response**: If already in preliminary context or previous results, answer directly without searching.

---

Your value lies in accurate retrieval and synthesis from this specific document collection. Stay within these bounds for reliable, grounded responses.
"""


def get_metadata(client: MilvusClient, username: str | None = None) -> list[str]:
    """Get all entries in the database (first 1000) with their title, authors, and date.

    Args:
        client: MilvusClient instance.
        username: Username for role-based access control.

    Returns:
        List of formatted metadata strings.

    Raises:
        RuntimeError: If metadata retrieval fails.
    """
    username = username or radchat_config.milvus.username
    try:
        all_docs = consolidate_documents(perform_query([], client, username=username))
    except Exception as e:
        error_msg = f"Failed to query documents for metadata retrieval: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    data = []
    try:
        for doc in all_docs:
            data.append(f"Title: {doc.metadata.title}\n" f"Authors: {doc.metadata.author}\n" f"Date: {doc.metadata.date}")
    except Exception as e:
        error_msg = f"Failed to format document metadata: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    return data


async def generate_response(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str | None = None,
    stream: bool = True,
) -> AsyncIterator[ChatCompletionChunk]:
    """Generate a response from the LLM with tool calling support.

    Args:
        messages: Conversation history.
        tools: Available tools for the model.
        model: Model name to use.
        stream: Whether to stream the response.

    Yields:
        Streaming chunks or tool call information.
    """
    model = model or radchat_config.ollama.llm_model

    # Convert base_url to OpenAI-compatible format if needed
    base_url = radchat_config.ollama.base_url
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.api_key,
        timeout=radchat_config.ollama.request_timeout,
    )

    # Build request body - OpenAI-compatible API
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": stream,
    }

    # For Ollama, we can pass extra parameters via extra_body
    # Some providers support num_ctx for context length
    extra_body: dict[str, Any] = {}
    if radchat_config.ollama.context_length:
        extra_body["num_ctx"] = radchat_config.ollama.context_length

    if extra_body:
        body["extra_body"] = extra_body

    # Standard OpenAI-compatible API
    response_stream = await client.chat.completions.create(**body)
    async for chunk in response_stream:
        yield chunk


def build_response(
    content: str,
    documents: list[MilvusDocument],
    model: str | None = None,
) -> dict[str, Any]:
    """Build the final response object with content and citations.

    Args:
        content: Generated response content.
        documents: Source documents for citations.
        model: Model name used.

    Returns:
        Complete response dictionary.
    """
    model = model or radchat_config.ollama.llm_model
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.final",
        "created": int(time.time()),
        "model": model,
        "citations": build_citations(documents),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class Pipe:
    """Orchestrates a streaming agentic loop with real-time citations."""

    def __init__(self) -> None:
        """Initialize the Pipe with default configuration."""
        self.user_valves = UserValves()
        self.citations = False

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Any | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Orchestrate a streaming agentic loop with real-time citations.

        Args:
            body: Request body with messages.
            __event_emitter__: Optional event emitter function for status updates.
            __user__: Optional user dictionary for authentication.

        Yields:
            Streaming response chunks and final response.
        """
        messages = body.get("messages", [])
        # TODO: Replace with __user__ data from ldap
        try:
            milvus_client = connect_milvus(
                username=self.user_valves.MILVUS_USERNAME,
                password=self.user_valves.MILVUS_PASSWORD,
                collection_name=self.user_valves.COLLECTION_NAME,
            )
        except CollectionNotFoundError as e:
            error_msg = f"Unable to connect to Milvus: Collection '{e.collection_name}' does not exist."
            logging.error(error_msg)
            yield {"error": error_msg}
            return
        except Exception as e:
            error_msg = f"Unable to connect to Milvus vector database: {str(e)}"
            logging.error(error_msg)
            yield {"error": error_msg}
            return

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Fetching data...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

        collection_name = self.user_valves.COLLECTION_NAME
        username = self.user_valves.MILVUS_USERNAME

        # Initial document retrieval
        try:
            initial_search_results = perform_search(
                client=milvus_client,
                queries=[messages[-1].get("content", "")],
                username=username,
            )
        except Exception as e:
            error_msg = f"Failed to perform initial search: {str(e)}"
            logging.error(error_msg)
            yield {"error": error_msg}
            return

        # Consolidate initial search results by document_id
        try:
            consolidated_initial_results = consolidate_documents(initial_search_results)
        except Exception as e:
            error_msg = f"Failed to consolidate search results: {str(e)}"
            logging.error(error_msg)
            yield {"error": error_msg}
            return

        # Render preliminary context documents
        try:
            preliminary_context = "\n\n".join([render_document(d, include_text=True) for d in consolidated_initial_results])
        except Exception as e:
            error_msg = f"Failed to render preliminary context documents: {str(e)}"
            logging.error(error_msg)
            yield {"error": error_msg}
            return

        # Emit initial citations immediately (using consolidated documents)
        seen_doc_ids = set()
        if __event_emitter__:
            for doc in consolidated_initial_results:
                if doc.default_document_id not in seen_doc_ids:
                    seen_doc_ids.add(doc.default_document_id)
                    try:
                        rendered_doc = render_document(doc, include_text=False)
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "source": {"name": doc.default_source, "url": ""},
                                    "document": [rendered_doc],
                                    "metadata": doc.model_dump(exclude={"default_text", "distance"}),
                                    "distance": doc.distance,
                                },
                            }
                        )
                    except Exception as e:
                        error_msg = f"Failed to render citation for document {doc.default_document_id}: {str(e)}"
                        logging.error(error_msg)
                        # Continue with other documents rather than failing completely

        try:
            schema_info = str(milvus_client.describe_collection(collection_name).get("description", ""))
        except Exception as e:
            error_msg = f"Warning: Failed to retrieve collection schema from '{collection_name}': {str(e)}. Continuing without schema information."
            logging.warning(error_msg)
            schema_info = f"{{'error': 'Unable to retrieve schema: {str(e)}'}}"

        try:
            metadata_list = get_metadata(milvus_client, username)
        except Exception as e:
            error_msg = f"Warning: Failed to retrieve document metadata: {str(e)}. Continuing without metadata list."
            logging.warning(error_msg)
            metadata_list = [error_msg]

        system_prompt = SYSTEM_PROMPT.replace("<<database_schema>>", schema_info).replace("<<preliminary_context>>", preliminary_context).replace("<<database_metadata>>", "\n\n".join(metadata_list))

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt
            all_messages = messages
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        available_tools = {"search": perform_search}
        all_sources = consolidated_initial_results
        final_content = ""

        # Why is this here? should be in the tool registry
        search_tool_schema = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Performs a semantic search using the given queries and optional filters.",
                "parameters": {
                    "type": "object",
                    "required": [],
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of queries for semantic search",
                        },
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of filter expressions to apply to the search",
                            "default": [],
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the Milvus collection to search. If not provided, uses the default collection.",
                        },
                        "partition_name": {
                            "type": "string",
                            "description": "Name of the partition to search within the collection. If not provided, searches all partitions.",
                        },
                    },
                },
            },
        }

        for i in range(radchat_config.agent.max_tool_calls):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Waiting on Ollama",
                            "done": False,
                            "hidden": False,
                        },
                    }
                )

            # Use generate_response function
            try:
                stream = generate_response(
                    messages=all_messages,
                    tools=[search_tool_schema],
                    model=radchat_config.ollama.llm_model,
                    stream=True,
                )
            except Exception as e:
                error_msg = f"Failed to generate response from LLM (model: {radchat_config.ollama.llm_model}, base_url: {radchat_config.ollama.base_url}): {str(e)}"
                logging.error(error_msg)
                yield {"error": error_msg}
                return

            tool_calls: list[Any] = []
            tool_calls_dict: dict[int, dict[str, Any]] = {}  # Accumulate by index
            try:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None

                    # Extract content from chunk
                    if choice and choice.delta.content:
                        final_content += choice.delta.content

                    # Handle tool_calls (accumulated across chunks)
                    if choice and choice.delta.tool_calls:
                        for tool_call_delta in choice.delta.tool_calls:
                            idx = tool_call_delta.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {
                                    "id": tool_call_delta.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tool_call_delta.function.name:
                                tool_calls_dict[idx]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls_dict[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                    else:
                        yield chunk.model_dump()

                    # Check if stream finished with tool_calls
                    if choice and choice.finish_reason == "tool_calls":
                        # Convert accumulated tool_calls dict to list
                        tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
                        logging.info(f"Tool calls received: {tool_calls}")
                        if tool_calls and __event_emitter__:
                            first_tool_name = tool_calls[0].get("function", {}).get("name", "unknown")
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Tool called: {first_tool_name}",
                                        "done": False,
                                        "hidden": False,
                                    },
                                }
                            )

                # Final check: if we accumulated tool_calls but didn't see finish_reason
                if not tool_calls and tool_calls_dict:
                    tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
                    logging.info(f"Tool calls accumulated: {tool_calls}")
            except Exception as e:
                error_msg = f"Error processing streaming response from LLM: {str(e)}"
                logging.error(error_msg)
                yield {"error": error_msg}
                return

            if not tool_calls:
                logging.info("No tool calls, breaking loop.")
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Done researching.",
                                "done": True,
                                "hidden": True,
                            },
                        }
                    )
                break

            all_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

            for tool in tool_calls:
                # Handle both dict format (from our accumulation) and OpenAI SDK format
                if isinstance(tool, dict):
                    function_name = tool["function"]["name"]
                    function_args = tool["function"]["arguments"]
                else:
                    function_name = tool.function.name
                    function_args = tool.function.arguments

                # Parse arguments if they're a string (JSON)
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse tool arguments as JSON for '{function_name}': {str(e)}. Arguments: {function_args[:200]}"
                        logging.error(error_msg)
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": error_msg,
                                "name": function_name,
                            }
                        )
                        continue

                if func := available_tools.get(function_name):
                    try:
                        try:
                            tool_output = func(
                                client=milvus_client,
                                **function_args,
                                username=username,
                            )
                        except CollectionNotFoundError as e:
                            error_msg = f"Search failed: Collection '{e.collection_name}' does not exist."
                            logging.error(error_msg)
                            all_messages.append(
                                {
                                    "role": "tool",
                                    "content": error_msg,
                                    "name": function_name,
                                }
                            )
                            continue
                        except Exception as e:
                            error_msg = f"Search tool '{function_name}' failed with arguments {function_args}: {str(e)}"
                            logging.error(error_msg)
                            all_messages.append(
                                {
                                    "role": "tool",
                                    "content": error_msg,
                                    "name": function_name,
                                }
                            )
                            continue

                        # Consolidate tool output by document_id
                        try:
                            consolidated_tool_output = consolidate_documents(tool_output)
                        except Exception as e:
                            error_msg = f"Failed to consolidate search results from '{function_name}': {str(e)}"
                            logging.error(error_msg)
                            all_messages.append(
                                {
                                    "role": "tool",
                                    "content": error_msg,
                                    "name": function_name,
                                }
                            )
                            continue

                        all_sources.extend(consolidated_tool_output)

                        # Emit citations immediately for new documents
                        if __event_emitter__:
                            for doc in consolidated_tool_output:
                                if doc.default_document_id not in seen_doc_ids:
                                    seen_doc_ids.add(doc.default_document_id)
                                    try:
                                        rendered_doc = render_document(doc, include_text=False)
                                        await __event_emitter__(
                                            {
                                                "type": "citation",
                                                "data": {
                                                    "source": {
                                                        "name": doc.default_source,
                                                        "url": "",
                                                    },
                                                    "document": [rendered_doc],
                                                    "metadata": doc.model_dump(exclude={"default_text", "distance"}),
                                                    "distance": doc.distance,
                                                },
                                            }
                                        )
                                    except Exception as e:
                                        error_msg = f"Failed to render citation for document {doc.default_document_id}: {str(e)}"
                                        logging.error(error_msg)
                                        # Continue with other documents rather than failing completely

                        try:
                            rendered_docs = "\n\n".join([render_document(d, include_text=True) for d in consolidated_tool_output]) if len(consolidated_tool_output) > 0 else "No documents found"
                            all_messages.append(
                                {
                                    "role": "tool",
                                    "content": rendered_docs,
                                    "name": function_name,
                                }
                            )
                        except Exception as e:
                            error_msg = f"Failed to render documents from '{function_name}': {str(e)}"
                            logging.error(error_msg)
                            all_messages.append(
                                {
                                    "role": "tool",
                                    "content": error_msg,
                                    "name": function_name,
                                }
                            )
                    except Exception as e:
                        error_msg = f"Unexpected error in tool execution '{function_name}': {str(e)}"
                        logging.error(error_msg)
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": error_msg,
                                "name": function_name,
                            }
                        )
                else:
                    # Tool not found - add error response to maintain message flow
                    available_tools_str = ", ".join(available_tools.keys())
                    error_msg = f"Error: Tool '{function_name}' not found. " f"Available tools: {available_tools_str}"
                    logging.error(error_msg)
                    all_messages.append(
                        {
                            "role": "tool",
                            "content": error_msg,
                            "name": function_name,
                        }
                    )

        # Use build_response function
        try:
            yield build_response(final_content, all_sources, radchat_config.ollama.llm_model)
        except Exception as e:
            error_msg = f"Failed to build final response: {str(e)}"
            logging.error(error_msg)
            yield {"error": error_msg}


class RadChatClient:
    """RadChat custom agent with Milvus integration."""

    MODEL_NAME = "radchat"

    def __init__(self) -> None:
        """Initialize the RadChat client."""
        self.pipe = Pipe()

    async def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using RadChat agent.

        Args:
            model: The model identifier (should be "radchat").
            messages: List of message dictionaries.
            stream: Whether to stream the response.
            **kwargs: Additional parameters (event_emitter, user, etc.).

        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk].
        """
        # Extract optional parameters
        event_emitter = kwargs.get("event_emitter")
        user = kwargs.get("user")

        # Create body dict for pipe
        body = {"messages": messages}

        if stream:
            # Return async iterator that yields chunks
            async def _generate_chunks() -> AsyncIterator[ChatCompletionChunk]:
                async for chunk in self.pipe.pipe(body=body, __event_emitter__=event_emitter, __user__=user):
                    # Convert dict chunks to ChatCompletionChunk
                    if "error" in chunk:
                        # Handle error case
                        yield ChatCompletionChunk(
                            id=str(uuid.uuid4()),
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[
                                {  # type: ignore[list-item]
                                    "index": 0,
                                    "delta": {"content": chunk["error"]},
                                    "finish_reason": "error",
                                }
                            ],
                        )
                    elif chunk.get("object") == "chat.completion.chunk":
                        # Already in OpenAI format
                        yield ChatCompletionChunk(**chunk)  # type: ignore[arg-type]
                    elif chunk.get("object") == "chat.completion.final":
                        # Final chunk with citations
                        final_chunk = ChatCompletionChunk(
                            id=chunk["id"],
                            object="chat.completion.chunk",
                            created=chunk["created"],
                            model=chunk["model"],
                            choices=[
                                {  # type: ignore[list-item]
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        )
                        yield final_chunk

            return _generate_chunks()
        else:
            # For non-streaming, collect all chunks and return final completion
            chunks: list[dict[str, Any]] = []
            async for chunk in self.pipe.pipe(body=body, __event_emitter__=event_emitter, __user__=user):
                chunks.append(chunk)

            # Find the final chunk with the complete response
            final_chunk: dict[str, Any] | None = None
            for chunk in chunks:
                if chunk.get("object") == "chat.completion.final":
                    final_chunk = chunk
                    break

            if final_chunk:
                return ChatCompletion(
                    id=final_chunk["id"],
                    object="chat.completion",
                    created=final_chunk["created"],
                    model=final_chunk["model"],
                    choices=[
                        {  # type: ignore[list-item]
                            "index": 0,
                            "message": ChatCompletionMessage(
                                role="assistant",
                                content=final_chunk["choices"][0]["message"]["content"],
                            ),
                            "finish_reason": "stop",
                        }
                    ],
                )
            else:
                # Fallback if no final chunk found
                return ChatCompletion(
                    id=str(uuid.uuid4()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        {  # type: ignore[list-item]
                            "index": 0,
                            "message": ChatCompletionMessage(role="assistant", content=""),
                            "finish_reason": "stop",
                        }
                    ],
                )

    async def create_embedding(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Ollama (RadChat delegates embeddings to Ollama).

        Args:
            model: The model identifier.
            input: Input text(s) to embed.
            **kwargs: Additional parameters.

        Returns:
            CreateEmbeddingResponse with embeddings.

        Note:
            RadChat client delegates embedding requests to Ollama since embeddings
            are not part of the RadChat agent functionality.
        """
        # RadChat doesn't handle embeddings directly, delegate to Ollama
        from src.clients.ollama import ollama_client

        return await ollama_client.create_embedding(
            model=model,
            input=input,
            **kwargs,
        )


# Singleton instance
radchat_client = RadChatClient()
