"""
title: Radchat Function
author: Rincon [mjst, kca]
version: 0.3
requirements: pymilvus, ollama, crawler
# Note, make sure that the tasks from settings.interface are not using this model.
TODO:
**TODAY**
- Document Rendering: the heading and markdown formatting is not nice.
  - For example: the title is `###` and then the text starts with `#`
  - Maybe I need to wrap in xml tags <document_data> and <document_text>
- Make sure the full system prompt has a good set of markdown formatting.
  - Don't intermix and inter layer the heading `#` and `##` everywhere.
  - The crawler seems to use `#` in too many places like in the milvus description.
**LATER**
- Make sure tool calls properly emit the fact that they are being called.
- User with auth:
  - The pipe function accepts __user__.
  - `from open_webui.models.users import Users`, `user = Users.get_user_by_id(__user__["id"])`, then use this as the auth stuff for the milvus client etc.
- Citations:
  - await __event_emitter__({
        "type": "citation",
        "data": {
            "document": [content],                    # Array of content strings
            "metadata": [                             # Array of metadata objects
                {
                    "date_accessed": datetime.now().isoformat(),
                    "source": title,
                    "author": "Author Name",          # Optional
                    "publication_date": "2024-01-01", # Optional
                    "url": "https://source-url.com"   # Optional
                }
            ],
            "source": {"name": title, "url": url}    # Primary source info
        }
    })
  - https://docs.openwebui.com/features/plugin/tools/development
"""

import os
import uuid
import time
import logging
from typing import Dict, Any, List
from collections import defaultdict

import ollama
from pydantic import BaseModel, Field

# Import from crawler package
from crawler import (
    DatabaseClientConfig,
    DatabaseClient,
    DatabaseDocument,
    EmbedderConfig,
    get_db,
    get_embedder,
    CrawlerConfig,
    LLMConfig,
)
from crawler.vector_db import CollectionDescription, SearchResult
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.chunker import ChunkingConfig


# -------------------------------
# --- Simplified Config ---
# -------------------------------


class RadchatConfig(BaseModel):
    """Simplified config for Ollama LLM settings (not covered by crawler)."""

    ollama_base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_embedding_model: str = Field(
        default=os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:v2")
    )
    ollama_llm_model: str = Field(
        default=os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b")
    )
    request_timeout: int = Field(
        default=int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
    )
    context_length: int = Field(
        default=int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32000"))
    )
    max_tool_calls: int = Field(default=int(os.getenv("AGENT_MAX_TOOL_CALLS", "5")))

    # Milvus connection defaults (can be overridden by UserValves)
    milvus_host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    milvus_port: int = Field(default=int(os.getenv("MILVUS_PORT", "19530")))
    default_collection: str = Field(
        default=os.getenv("IRAD_COLLECTION_NAME", "arxiv3")
    )
    default_username: str = Field(default=os.getenv("MILVUS_USERNAME", "matt"))
    default_password: str = Field(default=os.getenv("MILVUS_PASSWORD", "steffen"))


# Instantiate a global config
CONFIG = RadchatConfig()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Tool Schemas ---
SearchInputSchema = {
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
            },
        },
    },
}


# --- Core Functions ---
def connect_database(
    username: str,
    password: str,
    collection_name: str,
    host: str | None = None,
    port: int | None = None,
) -> tuple[DatabaseClient, CollectionDescription | None]:
    """
    Connect to Milvus using crawler's DatabaseClient.

    Args:
        username: Milvus username
        password: Milvus password
        collection_name: Name of the collection to connect to
        host: Milvus host (defaults to CONFIG.milvus_host)
        port: Milvus port (defaults to CONFIG.milvus_port)

    Returns:
        Tuple of (DatabaseClient, CollectionDescription or None)
    """
    host = host or CONFIG.milvus_host
    port = port or CONFIG.milvus_port

    # Create database config
    db_config = DatabaseClientConfig.milvus(
        collection=collection_name,
        host=host,
        port=port,
        username=username,
        password=password,
    )

    # Get embedder for search operations
    embed_config = EmbedderConfig(
        provider="ollama",
        model=CONFIG.ollama_embedding_model,
        base_url=CONFIG.ollama_base_url,
    )
    embedder = get_embedder(embed_config)

    # Create minimal crawler config (required for get_db)
    # The actual config will be loaded from the collection description
    llm_config = LLMConfig(
        provider="ollama",
        base_url=CONFIG.ollama_base_url,
        model_name=CONFIG.ollama_llm_model,
    )
    
    crawler_config = CrawlerConfig(
        name=collection_name,
        embeddings=embed_config,
        llm=llm_config,
        vision_llm=llm_config,
        database=db_config,
        converter=PyMuPDF4LLMConfig(),
        extractor=MetadataExtractorConfig(json_schema={}, llm=llm_config),
        chunking=ChunkingConfig(),
    )

    # Get and connect database client
    db = get_db(db_config, embedder.dimension, crawler_config, embedder)
    db.connect()

    # Get collection description with LLM prompt
    collection_desc = db.get_collection()

    return db, collection_desc


def perform_search(
    db: DatabaseClient,
    queries: list[str] | None = None,
    filters: list[str] | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """
    Use crawler's DatabaseClient.search() method.

    Args:
        db: DatabaseClient instance
        queries: List of query strings for semantic search
        filters: List of filter expressions
        limit: Maximum number of results

    Returns:
        List of SearchResult objects
    """
    queries = queries or []
    return db.search(texts=queries, filters=filters, limit=limit)


def render_document(result: SearchResult, include_text: bool = True) -> str:
    """
    Renders a SearchResult to markdown format.

    Args:
        result: The SearchResult to render
        include_text: If True, includes the full document text. If False, only metadata.

    Returns:
        Markdown-formatted string representation of the document
    """
    doc = result.document
    metadata = doc.metadata
    parts = []

    # Title
    if metadata.get("title"):
        parts.append(f"### {metadata['title']}")

    # Authors
    if metadata.get("author"):
        authors = metadata["author"]
        if isinstance(authors, list):
            parts.append(f"**Authors:** {', '.join(authors)}")
        else:
            parts.append(f"**Authors:** {authors}")

    # Date
    if metadata.get("date"):
        parts.append(f"**Date:** {metadata['date']}")

    # Source
    if doc.source:
        parts.append(f"**Source:** `{doc.source}` (Chunk: {doc.chunk_index})")

    # Distance/Score
    parts.append(f"**Relevance Score:** {result.score:.4f}")

    # Dynamically render other metadata fields, excluding those already handled or empty
    fields_to_ignore = {
        "title",
        "author",
        "date",
        "keywords",
        "unique_words",
    }

    other_metadata = []
    for key, value in metadata.items():
        if key not in fields_to_ignore and value:
            key_title = key.replace("_", " ").title()
            if isinstance(value, list):
                value_str = "`, `".join(map(str, value))
                other_metadata.append(f"**{key_title}:** `{value_str}`")
            else:
                other_metadata.append(f"**{key_title}:** {value}")

    if other_metadata:
        parts.extend(other_metadata)

    # Include full text if requested
    if include_text and doc.text:
        parts.append("\n---\n" + doc.text)

    return "\n".join(parts)


def consolidate_results(results: list[SearchResult]) -> list[SearchResult]:
    """
    Consolidates SearchResults with the same document_id, combining their text and metadata,
    and sorts the results by distance.

    Groups by document_id and combines:
    - Text chunks in order
    - Unique keywords
    - Unique words
    - Unique authors
    - Uses minimum distance / maximum score
    """
    if not results:
        return []

    # Group by document_id
    doc_groups: dict[str, list[SearchResult]] = defaultdict(list)
    for result in results:
        doc_groups[result.document.document_id].append(result)

    consolidated_results = []
    for doc_id, group_results in doc_groups.items():
        # Sort chunks by index to maintain reading order
        sorted_results = sorted(group_results, key=lambda r: r.document.chunk_index)
        base_result = sorted_results[0]
        base_doc = base_result.document

        # Combine text chunks with separator
        combined_text = "\n\n---\n\n".join(
            [r.document.text for r in sorted_results if r.document.text]
        )

        # Get combined metadata
        combined_metadata = dict(base_doc.metadata)

        # Combine unique keywords
        all_keywords = []
        for r in sorted_results:
            all_keywords.extend(r.document.metadata.get("keywords", []) or [])
        combined_metadata["keywords"] = sorted(list(set(all_keywords)))

        # Combine unique words
        all_unique_words = []
        for r in sorted_results:
            all_unique_words.extend(r.document.metadata.get("unique_words", []) or [])
        combined_metadata["unique_words"] = sorted(list(set(all_unique_words)))

        # Combine unique authors
        all_authors = []
        for r in sorted_results:
            authors = r.document.metadata.get("author", [])
            if authors:
                if isinstance(authors, list):
                    all_authors.extend(authors)
                else:
                    all_authors.append(authors)
        if all_authors:
            combined_metadata["author"] = sorted(list(set(all_authors)))

        # Use maximum score (minimum distance) across all chunks
        max_score = max(r.score for r in sorted_results)
        min_distance = min(r.distance for r in sorted_results)

        # Create consolidated document
        consolidated_doc = DatabaseDocument(
            id=base_doc.id,
            document_id=base_doc.document_id,
            text=combined_text,
            text_embedding=base_doc.text_embedding,
            chunk_index=0,  # Represents the consolidated document
            source=base_doc.source,
            security_group=base_doc.security_group,
            metadata=combined_metadata,
        )

        # Create consolidated SearchResult
        consolidated_result = SearchResult(
            document=consolidated_doc,
            distance=min_distance,
            score=max_score,
        )
        consolidated_results.append(consolidated_result)

    # Sort by score (higher is better)
    return sorted(consolidated_results, key=lambda r: r.score, reverse=True)


def get_metadata_summary(results: list[SearchResult]) -> str:
    """
    Gets a summary of metadata from search results.

    Args:
        results: List of SearchResult objects

    Returns:
        Formatted string of metadata summaries
    """
    consolidated = consolidate_results(results)
    data = []
    for result in consolidated:
        metadata = result.document.metadata
        title = metadata.get("title", "Unknown")
        author = metadata.get("author", [])
        date = metadata.get("date", "Unknown")
        data.append(f"Title: {title}\nAuthors: {author}\nDate: {date}")
    return "\n\n".join(data)


def build_citations(results: list[SearchResult]) -> list:
    """
    Builds citation objects from SearchResults for display in the UI.

    Args:
        results: List of SearchResult objects

    Returns:
        List of citation dictionaries
    """
    consolidated = consolidate_results(results)

    citations = []
    for result in consolidated:
        doc = result.document
        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [render_document(result, include_text=False)],
                "metadata": doc.metadata,
                "distance": result.distance,
                "score": result.score,
            }
        )
    return citations


def build_system_prompt(
    collection_desc: CollectionDescription | None,
    preliminary_context: str,
    database_metadata: str,
) -> str:
    """
    Build system prompt using collection description's llm_prompt.

    Args:
        collection_desc: CollectionDescription containing llm_prompt
        preliminary_context: Rendered preliminary search results
        database_metadata: Summary of available documents

    Returns:
        Complete system prompt string
    """
    # Get llm_prompt from collection description, or use fallback
    if collection_desc and collection_desc.llm_prompt:
        schema_info = collection_desc.llm_prompt
    else:
        schema_info = "No collection schema information available."

    return f"""# Role and Task

You are a specialized document retrieval assistant. Your task is to help users find and extract information from a document collection.

## Context

You have access to a document database through a search tool.

**Database Schema and Filtering Instructions**:
<database_schema>
{schema_info}
</database_schema>

**Available Documents** (comprehensive sample):
<available_documents>
{database_metadata}
</available_documents>

**Preliminary Context**:
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

### Step 3: Synthesize and Respond

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

---

Your value lies in accurate retrieval and synthesis from this specific document collection. Stay within these bounds for reliable, grounded responses.
"""


# --- Response Generation Functions ---
def generate_response(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str | None = None,
    stream: bool = True,
):
    """
    Generates a response from the LLM with tool calling support.

    Args:
        messages: Conversation history
        tools: Available tools for the model
        model: Model name to use
        stream: Whether to stream the response

    Yields:
        Streaming chunks or tool call information
    """
    model = model or CONFIG.ollama_llm_model
    ollama_client = ollama.Client(
        host=CONFIG.ollama_base_url, timeout=CONFIG.request_timeout
    )
    response_stream = ollama_client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"num_ctx": CONFIG.context_length},
        stream=stream,
    )
    for chunk in response_stream:
        yield chunk


def build_response(
    content: str,
    results: list[SearchResult],
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Builds the final response object with content and citations.

    Args:
        content: Generated response content
        results: Source SearchResults for citations
        model: Model name used

    Returns:
        Complete response dictionary
    """
    model = model or CONFIG.ollama_llm_model
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.final",
        "created": int(time.time()),
        "model": model,
        "citations": build_citations(results),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def to_openai_chunk(ollama_chunk: ollama.ChatResponse) -> dict:
    """Converts an Ollama streaming chunk to the OpenAI format."""
    delta = {}
    if ollama_chunk.message.thinking:
        delta["thinking"] = ollama_chunk.message.thinking
    else:
        delta["content"] = ollama_chunk.message.content

    finish_reason = ollama_chunk.done_reason

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.chunk",
        "created": ollama_chunk.created_at,
        "model": ollama_chunk.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


class Pipe:
    """Open WebUI Pipe for RAG-based document retrieval using crawler package."""

    class UserValves(BaseModel):
        COLLECTION_NAME: str = Field(default_factory=lambda: CONFIG.default_collection)
        MILVUS_USERNAME: str = Field(default_factory=lambda: CONFIG.default_username)
        MILVUS_PASSWORD: str = Field(default_factory=lambda: CONFIG.default_password)

    def __init__(self):
        self.user_valves = self.UserValves()
        self.db: DatabaseClient | None = None
        self.collection_desc: CollectionDescription | None = None

    async def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict = None,
    ):
        """Orchestrates a streaming agentic loop with real-time citations."""

        messages = body.get("messages", [])

        # Connect using crawler's database client
        try:
            self.db, self.collection_desc = connect_database(
                username=self.user_valves.MILVUS_USERNAME,
                password=self.user_valves.MILVUS_PASSWORD,
                collection_name=self.user_valves.COLLECTION_NAME,
            )
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            yield {"error": f"Unable to connect to the vector database: {e}"}
            return

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

        # Initial document retrieval using db.search()
        initial_results = perform_search(
            db=self.db,
            queries=[messages[-1].get("content")] if messages else [],
            limit=10,
        )

        # Consolidate initial search results by document_id
        consolidated_results = consolidate_results(initial_results)

        preliminary_context = "\n\n".join(
            [render_document(r, include_text=True) for r in consolidated_results]
        )

        if consolidated_results:
            logging.info(
                f"RENDERED DOC WITHOUT TEXT: {render_document(consolidated_results[0], include_text=False)}"
            )

        # Emit initial citations immediately (using consolidated documents)
        seen_doc_ids: set[str] = set()
        for result in consolidated_results:
            doc = result.document
            if doc.document_id not in seen_doc_ids:
                seen_doc_ids.add(doc.document_id)
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": doc.source, "url": ""},
                            "document": [render_document(result, include_text=False)],
                            "metadata": doc.metadata,
                            "distance": result.distance,
                            "score": result.score,
                        },
                    }
                )

        # Build system prompt using collection description
        system_prompt = build_system_prompt(
            collection_desc=self.collection_desc,
            preliminary_context=preliminary_context,
            database_metadata=get_metadata_summary(consolidated_results),
        )

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
            all_messages = messages
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        # Create search wrapper that uses self.db
        def search_tool(
            queries: list[str] | None = None, filters: list[str] | None = None
        ) -> list[SearchResult]:
            return perform_search(db=self.db, queries=queries, filters=filters)

        available_tools = {"search": search_tool}
        all_sources = list(consolidated_results)
        final_content = ""

        for i in range(CONFIG.max_tool_calls):
            logging.info(f"Agent loop iteration {i+1}/{CONFIG.max_tool_calls}")
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
            stream = generate_response(
                messages=all_messages,
                tools=[SearchInputSchema],
                model=CONFIG.ollama_llm_model,
                stream=True,
            )

            tool_calls: list[ollama.Message.ToolCall] = []
            for chunk in stream:
                if new_tool_calls := chunk.message.tool_calls:
                    logging.info(f"Tool calls received: {new_tool_calls}")
                    tool_calls.extend(new_tool_calls)
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Tool called: {new_tool_calls[0].function.name}",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                else:
                    yield to_openai_chunk(chunk)
                if content_chunk := chunk.message.content:
                    final_content += content_chunk

            if not tool_calls:
                logging.info("No tool calls, breaking loop.")
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

            all_messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

            for tool in tool_calls:
                function_name = tool.function.name
                function_args = tool.function.arguments

                if func := available_tools.get(function_name):
                    logging.info(f"Tool found: {function_name}")
                    try:
                        tool_output = func(**function_args)

                        # Consolidate tool output by document_id
                        consolidated_tool_output = consolidate_results(tool_output)
                        logging.info(
                            f"Consolidated tool output: {len(consolidated_tool_output)} documents"
                        )
                        all_sources.extend(consolidated_tool_output)

                        # Emit citations immediately for new documents
                        for result in consolidated_tool_output:
                            doc = result.document
                            if doc.document_id not in seen_doc_ids:
                                seen_doc_ids.add(doc.document_id)
                                await __event_emitter__(
                                    {
                                        "type": "citation",
                                        "data": {
                                            "source": {
                                                "name": doc.source,
                                                "url": "",
                                            },
                                            "document": [
                                                render_document(
                                                    result, include_text=False
                                                )
                                            ],
                                            "metadata": doc.metadata,
                                            "distance": result.distance,
                                            "score": result.score,
                                        },
                                    }
                                )

                        all_messages.append(
                            {
                                "role": "tool",
                                "content": (
                                    "\n\n".join(
                                        [
                                            render_document(r, include_text=True)
                                            for r in consolidated_tool_output
                                        ]
                                    )
                                    if len(consolidated_tool_output) > 0
                                    else "No documents found"
                                ),
                                "name": function_name,
                            }
                        )
                    except Exception as e:
                        logging.error(f"Error executing tool {function_name}: {e}")
                        all_messages.append(
                            {
                                "role": "tool",
                                "content": f"Error executing tool {function_name}: {e}",
                                "name": function_name,
                            }
                        )
                else:
                    logging.error(f"Tool not found: {function_name}")
                    # Tool not found - add error response to maintain message flow
                    error_msg = f"Error: Tool '{function_name}' not found. Available tools: {', '.join(available_tools.keys())}"
                    all_messages.append(
                        {
                            "role": "tool",
                            "content": error_msg,
                            "name": function_name,
                        }
                    )

        # Use build_response function
        yield build_response(final_content, all_sources, CONFIG.ollama_llm_model)
