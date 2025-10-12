"""
title: Radchat Function
author: Rincon [mjst, kca]
version: 0.2
requirements: pymilvus, ollama
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
import json
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict

import ollama
from pydantic import BaseModel, Field, ConfigDict
from pymilvus import (
    MilvusClient,
    AnnSearchRequest,
    RRFRanker,
)


# -------------------------------
# --- Config Component Models ---
# -------------------------------


class OllamaConfig(BaseModel):
    base_url: str = Field(
        default="http://localhost:11434"
        # default="http://host.docker.internal:11434"
        # default="http://ollama.a1.autobahn.rinconres.com"
        # default=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    )
    embedding_model: str = Field(
        default=os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:v2")
    )
    llm_model: str = Field(default=os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b"))
    request_timeout: int = Field(
        default=int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
    )
    context_length: int = Field(
        default=int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32000"))
    )


class MilvusConfig(BaseModel):
    host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    port: str = Field(default=os.getenv("MILVUS_PORT", "19530"))
    username: str = Field(default=os.getenv("MILVUS_USERNAME", "matt"))
    password: str = Field(default=os.getenv("MILVUS_PASSWORD", "steffen"))
    collection_name: str = Field(default=os.getenv("IRAD_COLLECTION_NAME", "arxiv3"))


class SearchConfig(BaseModel):
    nprobe: int = Field(default=int(os.getenv("MILVUS_NPROBE", "10")))
    search_limit: int = Field(default=int(os.getenv("MILVUS_SEARCH_LIMIT", "5")))
    hybrid_limit: int = Field(
        default=int(os.getenv("MILVUS_HYBRID_SEARCH_LIMIT", "10"))
    )
    rrf_k: int = Field(default=int(os.getenv("MILVUS_RRF_K", "100")))
    drop_ratio: float = Field(default=float(os.getenv("MILVUS_DROP_RATIO", "0.2")))
    output_fields: List[str] = Field(
        default=[
            "metadata",
            "default_text",
            "default_document_id",
            "default_chunk_index",
            "default_source",
        ]
    )


class AgentConfig(BaseModel):
    max_tool_calls: int = Field(default=int(os.getenv("AGENT_MAX_TOOL_CALLS", "5")))
    default_role: str = Field(default=os.getenv("AGENT_DEFAULT_ROLE", "system"))
    logging_level: str = Field(default=os.getenv("AGENT_LOGGING_LEVEL", "INFO"))


# -------------------------------
# --- Unified Config Manager ---
# -------------------------------


class RadchatConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def update_from_valves(self, valves: "Pipe.UserValves"):
        """
        Override runtime configuration from user-provided valves (e.g. in Pipe).
        """
        if hasattr(valves, "MILVUS_USERNAME"):
            self.milvus.username = valves.MILVUS_USERNAME
        if hasattr(valves, "MILVUS_PASSWORD"):
            self.milvus.password = valves.MILVUS_PASSWORD
        if hasattr(valves, "COLLECTION_NAME"):
            self.milvus.collection_name = valves.COLLECTION_NAME


# Instantiate a global, immutable base config
CONFIG = RadchatConfig()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Pydantic Models for Type Safety ---
class MilvusDocumentMetadata(BaseModel):
    title: Optional[str] = ""
    author: Optional[Union[List[str], str]] = Field(default_factory=list)
    date: Optional[int] = 0
    keywords: Optional[List[str]] = Field(default_factory=list)
    unique_words: Optional[List[str]] = Field(default_factory=list)

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class MilvusDocument(BaseModel):
    id: Optional[Any] = -1
    default_document_id: str
    default_text: str
    default_chunk_index: int
    default_source: str
    security_group: List[str] = Field(default_factory=lambda: ["public"])
    metadata: MilvusDocumentMetadata = Field(default_factory=MilvusDocumentMetadata)
    distance: Optional[float] = 1.0

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusDocument":
        """Create a document from a dict (with validation)."""
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Dump the model as a dict."""
        return self.model_dump()


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
def connect_milvus(
    username: str = None,
    password: str = None,
    collection_name: str = None,
) -> Optional[MilvusClient]:
    username = username or CONFIG.milvus.username
    password = password or CONFIG.milvus.password
    collection_name = collection_name or CONFIG.milvus.collection_name
    uri = f"http://{CONFIG.milvus.host}:{CONFIG.milvus.port}"
    try:
        client = MilvusClient(uri=uri, token=f"{username}:{password}")
        if not client.has_collection(collection_name=collection_name):
            logging.error(f"Error: Collection '{collection_name}' does not exist.")
            return None
        client.load_collection(collection_name=collection_name)
        logging.info(f"Collection '{collection_name}' loaded.")
        return client
    except Exception as e:
        logging.error(
            f"Error connecting to or loading Milvus collection '{collection_name}': {e}"
        )
        return None


def get_embedding(text: str | list[str]) -> Optional[List[float]]:
    """
    Gets embedding for a single text string.

    Args:
        text: Text to embed or list of texts to embed

    Returns:
        Embedding vector or None if failed
    """
    try:
        client = ollama.Client(host=CONFIG.ollama.base_url)
        response = client.embed(model=CONFIG.ollama.embedding_model, input=text)
        return response.get("embeddings")
    except Exception as e:
        logging.error(
            f"Error getting embedding from Ollama ({CONFIG.ollama.base_url}): {e}"
        )
        return None


def perform_search(
    client: MilvusClient,
    queries: list[str] = [],
    filters: list[str] = [],
    collection_name: str = None,
    username: str = None,
) -> list[MilvusDocument]:
    """
    Performs hybrid search (dense + sparse embeddings) on the Milvus collection.

    Args:
        client: MilvusClient instance
        queries: List of query strings for semantic search
        filters: List of filter expressions (will not be mutated)
        collection_name: Name of the collection to search
        username: Username for role-based access control

    Returns:
        List of matching MilvusDocument objects
    """
    collection_name = collection_name or CONFIG.milvus.collection_name
    username = username or CONFIG.milvus.username

    search_requests = []

    # Build filter expression without mutating the original list
    user_roles = list(client.describe_user(user_name=username).get("roles", []))
    security_filter = f"array_contains_any(security_group, {user_roles})"

    # Combine security filter with user filters
    all_filters = [security_filter] + filters
    filter_expr = " and ".join(all_filters)

    search_configs = [
        {
            "field": "default_text_sparse_embedding",
            "param": {"drop_ratio_search": CONFIG.search.drop_ratio},
            "data_transform": lambda q: [q],
        },
        {
            "field": "default_metadata_sparse_embedding",
            "param": {"drop_ratio_search": CONFIG.search.drop_ratio},
            "data_transform": lambda q: [q],
        },
    ]

    for query in queries:
        embeddings = get_embedding(query)
        if embeddings:
            # TODO: get the params from the index describe in the collection
            search_requests.append(
                AnnSearchRequest(
                    data=embeddings,
                    anns_field="default_text_embedding",
                    param={
                        "metric_type": "COSINE",
                        "params": {"nprobe": CONFIG.search.nprobe},
                    },
                    expr=filter_expr,
                    limit=CONFIG.search.search_limit,
                )
            )

        for config in search_configs:
            search_requests.append(
                AnnSearchRequest(
                    data=config["data_transform"](query),
                    anns_field=config["field"],
                    param=config["param"],
                    expr=filter_expr,
                    limit=CONFIG.search.search_limit,
                )
            )

    if not search_requests:
        if len(all_filters) > 0:
            return perform_query(all_filters, client, collection_name)
        return []

    result = client.hybrid_search(
        collection_name=collection_name,
        reqs=search_requests,
        ranker=RRFRanker(k=CONFIG.search.rrf_k),
        output_fields=CONFIG.search.output_fields,
        limit=CONFIG.search.hybrid_limit,
    )

    return consolidate_documents(
        [MilvusDocument(**doc["entity"], distance=doc["distance"]) for doc in result[0]]
    )


def perform_query(
    filters: list[str],
    client: MilvusClient,
    collection_name: str = None,
) -> List[MilvusDocument]:
    collection_name = collection_name or CONFIG.milvus.collection_name
    query_results = client.query(
        collection_name=collection_name,
        filter=" and ".join(filters),
        output_fields=CONFIG.search.output_fields,
        limit=100,
    )
    return consolidate_documents([MilvusDocument(**doc) for doc in query_results])


def get_metadata(client: MilvusClient) -> list[str]:
    """
    Gets all the entries in the database (first 1000) and their title + authors + date.

    Args:
        client: MilvusClient instance

    Returns:
        List of formatted metadata strings
    """
    all_docs = consolidate_documents(perform_query([], client))
    data = []
    for doc in all_docs:
        data.append(
            f"Title: {doc.metadata.title}\nAuthors: {doc.metadata.author}\nDate: {doc.metadata.date}"
        )
    return data


def render_document(document: MilvusDocument, include_text: bool = True) -> str:
    """
    Renders a MilvusDocument to markdown format.

    Args:
        document: The document to render
        include_text: If True, includes the full document text. If False, only metadata.

    Returns:
        Markdown-formatted string representation of the document
    """
    parts = []

    # Title
    if document.metadata.title:
        parts.append(f"### {document.metadata.title}")

    # Authors
    if document.metadata.author:
        authors = document.metadata.author
        if isinstance(authors, list):
            parts.append(f"**Authors:** {', '.join(authors)}")
        else:
            parts.append(f"**Authors:** {authors}")

    # Date
    if document.metadata.date:
        parts.append(f"**Date:** {document.metadata.date}")

    # Source
    if document.default_source:
        parts.append(
            f"**Source:** `{document.default_source}` (Chunk: {document.default_chunk_index})"
        )

    # Dynamically render other metadata fields, excluding those already handled or empty
    fields_to_ignore = {
        "title",
        "author",
        "date",
        "keywords",
        "unique_words",
    }

    other_metadata = []
    for key, value in document.metadata.model_dump().items():
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
    if include_text and document.default_text:
        parts.append("\n---\n" + document.default_text)

    return "\n".join(parts)


def consolidate_documents(documents: List[MilvusDocument]) -> List[MilvusDocument]:
    """
    Consolidates documents with the same document_id, combining their text and metadata,
    and sorts the results by distance.

    Groups by default_document_id and combines:
    - Text chunks in order
    - Unique keywords
    - Unique words
    - Unique authors
    - Uses minimum distance
    """
    if not documents:
        return []

    # Group by document_id instead of source
    doc_groups = defaultdict(list)
    for doc in documents:
        doc_groups[doc.default_document_id].append(doc)

    consolidated_docs = []
    for doc_id, docs in doc_groups.items():
        # Sort chunks by index to maintain reading order
        sorted_chunks = sorted(docs, key=lambda d: d.default_chunk_index)
        base_doc = sorted_chunks[0]

        # Combine text chunks with separator
        combined_text = "\n\n---\n\n".join(
            [d.default_text for d in sorted_chunks if d.default_text]
        )

        # Combine unique keywords
        combined_keywords = sorted(
            list(set(kw for d in sorted_chunks for kw in (d.metadata.keywords or [])))
        )

        # Combine unique words
        combined_unique_words = sorted(
            list(
                set(
                    word
                    for d in sorted_chunks
                    for word in (d.metadata.unique_words or [])
                )
            )
        )

        # Combine unique authors
        all_authors = []
        for d in sorted_chunks:
            if d.metadata.author:
                if isinstance(d.metadata.author, list):
                    all_authors.extend(d.metadata.author)
                else:
                    all_authors.append(d.metadata.author)
        combined_authors = sorted(list(set(all_authors))) if all_authors else []

        # Use minimum distance across all chunks
        min_distance = min(
            (d.distance for d in sorted_chunks if d.distance is not None), default=None
        )

        # Create consolidated document
        consolidated_data = base_doc.model_dump()
        consolidated_data.update(
            {
                "default_text": combined_text,
                "distance": min_distance,
                "default_chunk_index": 0,  # Represents the consolidated document
            }
        )

        # Update metadata with combined fields
        updated_metadata = consolidated_data["metadata"].copy()
        updated_metadata["keywords"] = combined_keywords
        updated_metadata["unique_words"] = combined_unique_words
        updated_metadata["author"] = (
            combined_authors if combined_authors else updated_metadata.get("author")
        )
        consolidated_data["metadata"] = updated_metadata

        consolidated_docs.append(MilvusDocument(**consolidated_data))

    # Sort the final list of consolidated documents by distance
    return sorted(
        consolidated_docs,
        key=lambda d: d.distance if d.distance is not None else float("inf"),
    )


def build_citations(documents: List[MilvusDocument]) -> list:
    """
    Builds citation objects from documents for display in the UI.

    Args:
        documents: List of documents to create citations from

    Returns:
        List of citation dictionaries
    """
    consolidated_docs = consolidate_documents(documents)

    citations = []
    for doc in consolidated_docs:
        citations.append(
            {
                "source": {"name": doc.default_source, "url": ""},
                "document": [render_document(doc, include_text=False)],
                "metadata": doc.model_dump(exclude={"default_text", "distance"}),
                "distance": doc.distance,
            }
        )
    return citations


# --- Testable Component Functions ---
def generate_response(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str = None,
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
    model = model or CONFIG.ollama.llm_model
    ollama_client = ollama.Client(
        host=CONFIG.ollama.base_url, timeout=CONFIG.ollama.request_timeout
    )
    # Don't await - chat() with stream=True returns an async generator
    response_stream = ollama_client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"num_ctx": CONFIG.ollama.context_length},
        stream=stream,
    )
    # Now await the generator
    for chunk in response_stream:
        yield chunk


def build_response(
    content: str,
    documents: List[MilvusDocument],
    model: str = None,
) -> Dict[str, Any]:
    """
    Builds the final response object with content and citations.

    Args:
        content: Generated response content
        documents: Source documents for citations
        model: Model name used

    Returns:
        Complete response dictionary
    """
    model = model or CONFIG.ollama.llm_model
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
    class UserValves(BaseModel):
        COLLECTION_NAME: str = Field(
            default_factory=lambda: CONFIG.milvus.collection_name
        )
        MILVUS_USERNAME: str = Field(default_factory=lambda: CONFIG.milvus.username)
        MILVUS_PASSWORD: str = Field(default_factory=lambda: CONFIG.milvus.password)

    def __init__(self):
        self.user_valves = self.UserValves()
        self.citations = False

    async def pipe(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict = None,
    ):
        """Orchestrates a streaming agentic loop with real-time citations."""

        messages = body.get("messages", [])
        # TODO: Replace with __user__ data from ldap
        milvus_client = connect_milvus(
            username=self.user_valves.MILVUS_USERNAME,
            password=self.user_valves.MILVUS_PASSWORD,
            collection_name=self.user_valves.COLLECTION_NAME,
        )
        if not milvus_client:
            yield {"error": "Unable to connect to the vector database."}
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

        # Initial document retrieval
        initial_search_results = perform_search(
            client=milvus_client,
            queries=[messages[-1].get("content")],
            username=self.user_valves.MILVUS_USERNAME,
        )

        # Consolidate initial search results by document_id
        consolidated_initial_results = consolidate_documents(initial_search_results)

        preliminary_context = "\n\n".join(
            [
                render_document(d, include_text=True)
                for d in consolidated_initial_results
            ]
        )
        print(
            "RENDERED DOC WITHOUT TEXT:",
            render_document(consolidated_initial_results[0], include_text=False),
        )
        print(
            "RENDERED DOC WITH TEXT:",
            render_document(consolidated_initial_results[0], include_text=True),
        )
        print("\n\n")

        # Emit initial citations immediately (using consolidated documents)
        seen_doc_ids = set()
        for doc in consolidated_initial_results:
            if doc.default_document_id not in seen_doc_ids:
                seen_doc_ids.add(doc.default_document_id)
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": doc.default_source, "url": ""},
                            "document": [render_document(doc, include_text=False)],
                            "metadata": doc.model_dump(
                                exclude={"default_text", "distance"}
                            ),
                            "distance": doc.distance,
                        },
                    }
                )

        try:
            schema_info = str(
                milvus_client.describe_collection(self.user_valves.COLLECTION_NAME).get(
                    "description", ""
                )
            )
        except Exception as e:
            logging.error(f"Failed to describe collection: {e}")
            schema_info = "{}"

        print(
            "schema info:",
            milvus_client.describe_collection(self.user_valves.COLLECTION_NAME).get(
                "description", ""
            ),
        )
        # print("preliminary context:", preliminary_context)
        # print("metadata:", get_metadata(milvus_client))

        system_prompt = (
            SystemPrompt.replace("<<database_schema>>", schema_info)
            .replace("<<preliminary_context>>", preliminary_context)
            .replace("<<database_metadata>>", "\n\n".join(get_metadata(milvus_client)))
        )

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
            all_messages = messages
        else:
            all_messages = [{"role": "system", "content": system_prompt}] + messages

        available_tools = {"search": perform_search}
        all_sources = consolidated_initial_results
        final_content = ""

        for i in range(CONFIG.agent.max_tool_calls):
            logging.info(f"Agent loop iteration {i+1}/{CONFIG.agent.max_tool_calls}")
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
            # print("all messages:", all_messages)
            stream = generate_response(
                messages=all_messages,
                tools=[SearchInputSchema],
                model=CONFIG.ollama.llm_model,
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
                    print(f"Tool found: {function_name}")
                    try:
                        tool_output = func(
                            client=milvus_client,
                            **function_args,
                            username=self.user_valves.MILVUS_USERNAME,
                        )

                        # Consolidate tool output by document_id
                        consolidated_tool_output = consolidate_documents(tool_output)
                        print(f"Consolidated tool output: {consolidated_tool_output}")
                        all_sources.extend(consolidated_tool_output)

                        # Emit citations immediately for new documents
                        for doc in consolidated_tool_output:
                            if doc.default_document_id not in seen_doc_ids:
                                seen_doc_ids.add(doc.default_document_id)
                                await __event_emitter__(
                                    {
                                        "type": "citation",
                                        "data": {
                                            "source": {
                                                "name": doc.default_source,
                                                "url": "",
                                            },
                                            "document": [
                                                render_document(doc, include_text=False)
                                            ],
                                            "metadata": doc.model_dump(
                                                exclude={"default_text", "distance"}
                                            ),
                                            "distance": doc.distance,
                                        },
                                    }
                                )

                        all_messages.append(
                            {
                                "role": "tool",
                                "content": (
                                    "\n\n".join(
                                        [
                                            render_document(d, include_text=True)
                                            for d in consolidated_tool_output
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
                    print(f"Tool not found: {function_name}")
                    # Tool not found - add error response to maintain message flow
                    error_msg = f"Error: Tool '{function_name}' not found. Available tools: {', '.join(available_tools.keys())}"
                    logging.error(error_msg)
                    all_messages.append(
                        {
                            "role": "tool",
                            "content": error_msg,
                            "name": function_name,
                        }
                    )

        # Use build_response function
        yield build_response(final_content, all_sources, CONFIG.ollama.llm_model)


SystemPrompt = """# Role and Task

You are a specialized document retrieval assistant. Your task is to help users find and extract information from an internal research and development (IRAD) document collection covering signal processing, AI, and ML topics.

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
