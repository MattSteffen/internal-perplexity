"""Milvus search tool implementation for semantic document retrieval.

Delegates to the crawler's MilvusDB: search() builds a DatabaseClientConfig from
env + token, instantiates MilvusDB (crawler_config=None for search-only), and calls
db.search(). Filter-only: no text/queries => db.search(texts=[], filters=..., limit=...).
Semantic search: uses db.get_collection() for the stored embedding model, sets
embedder, then db.search(texts=[query_text], ...). Uses crawler DatabaseDocument,
SearchResult, and Document for consolidation and citations.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from src.milvus_client import (
    get_milvus_uri,
    parse_milvus_token,
    parse_milvus_uri,
)

try:
    from crawler.document import Document
    from crawler.vector_db import (
        DatabaseClientConfig,
        DatabaseDocument,
        MilvusDB,
        SearchResult,
    )
except ImportError as e:
    raise ImportError("crawler package not available") from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_COLLECTION_NAME = os.environ.get("IRAD_COLLECTION_NAME", "arxiv3")

_search_executor: ThreadPoolExecutor | None = None


def _get_search_executor() -> ThreadPoolExecutor:
    global _search_executor
    if _search_executor is None:
        _search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")
    return _search_executor


class CollectionNotFoundError(Exception):
    """Raised when a Milvus collection does not exist."""

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' does not exist.")


# --- Server embedder (sync wrapper for tool/search path) ---


class _ServerEmbedder:
    """Sync embedder that calls the server's async create_embedding from a worker thread."""

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, query: str) -> list[float]:
        from src.clients.router import client_router

        async def _embed() -> list[float]:
            r = await client_router.create_embedding(model=self.model, input=query)
            if not r.data or len(r.data) == 0:
                raise RuntimeError("Empty embedding response")
            return r.data[0].embedding

        return asyncio.run(_embed())


# --- Unified search ---


def search(
    text: str | None = None,
    queries: list[str] | None = None,
    filters: list[str] | None = None,
    collection_name: str | None = None,
    partition_name: str | None = None,
    token: str | None = None,
    limit: int = 100,
) -> list[SearchResult]:
    """Single search entry point. Delegates to crawler MilvusDB.search().

    If no text/queries provided: filter-only query (db.search(texts=[], ...)).
    With text: loads collection description via db.get_collection(), sets embedder
    from stored config, then db.search(texts=[query_text], ...).

    Args:
        text: Optional single query string.
        queries: Optional list of query strings (first used if text not set).
        filters: Optional Milvus filter expressions.
        collection_name: Collection to search (default from env IRAD_COLLECTION_NAME).
        partition_name: Optional partition name (unused; kept for API compatibility).
        token: Milvus token (username:password). Required.
        limit: Max results.

    Returns:
        List of SearchResult (crawler type).
    """
    coll = collection_name or DEFAULT_COLLECTION_NAME
    flist = list(filters) if filters else []
    if not token:
        raise ValueError("token is required for search")

    # Resolve query text: prefer text, then first of queries
    query_text: str | None = None
    if text and text.strip():
        query_text = text.strip()
    elif queries:
        for q in queries:
            if q and str(q).strip():
                query_text = str(q).strip()
                break

    uri = get_milvus_uri()
    host, port = parse_milvus_uri(uri)
    username, password = parse_milvus_token(token)
    database_config = DatabaseClientConfig.milvus(
        collection=coll,
        host=host,
        port=port,
        partition=partition_name,
        username=username,
        password=password,
    )
    db = MilvusDB(
        database_config,
        embedding_dimension=1,
        crawler_config=None,
        embedder=None,
    )
    try:
        db.connect(create_if_missing=False)
    except RuntimeError as e:
        if "does not exist" in str(e):
            raise CollectionNotFoundError(coll) from e
        raise

    try:
        if not query_text:
            return db.search(texts=[], filters=flist, limit=limit)

        collection_desc = db.get_collection()
        if not collection_desc:
            raise RuntimeError(
                f"Collection '{coll}' has no description; cannot run semantic search without pipeline config"
            )
        embedder = _ServerEmbedder(model=collection_desc.collection_config.embeddings.model)
        db.set_embedder(embedder)
        return db.search(texts=[query_text], filters=flist, limit=limit)
    finally:
        db.disconnect()


# --- Consolidation and citations ---


def consolidate_documents(search_results: list[SearchResult]) -> list[Document]:
    """Group SearchResult by document_id and build Document per group (crawler types)."""
    if not search_results:
        return []

    by_id: dict[str, list[DatabaseDocument]] = defaultdict(list)
    for sr in search_results:
        by_id[sr.document.document_id].append(sr.document)

    combined: list[Document] = []
    for document_id, db_docs in by_id.items():
        db_docs.sort(key=lambda d: d.chunk_index)
        doc = Document.from_database_documents(db_docs, include_chunks=False)
        combined.append(doc)
    return combined


def build_citations(documents: list[Document]) -> list[dict[str, Any]]:
    """Build OI-style citation dicts from crawler Documents."""
    citations: list[dict[str, Any]] = []
    for doc in documents:
        meta = doc.metadata or {}
        author_val = meta.get("author")
        if isinstance(author_val, list):
            author_str = ", ".join(author_val)
        else:
            author_str = str(author_val) if author_val else ""
        publication_date = str(meta.get("date", "")) if meta.get("date") else ""
        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [render_document(doc, include_text=False)],
                "metadata": [
                    {
                        "date_accessed": datetime.now().isoformat(),
                        "source": meta.get("title") or doc.source,
                        "author": author_str,
                        "publication_date": publication_date,
                        "url": "",
                    }
                ],
                "distance": None,
            }
        )
    return citations


def render_document(doc: Document, include_text: bool = True) -> str:
    """Render a crawler Document to markdown for tool/UI (public for callers)."""
    parts: list[str] = []
    meta = doc.metadata or {}
    if meta.get("title"):
        parts.append(f"### {meta['title']}")
    author = meta.get("author")
    if author:
        if isinstance(author, list):
            parts.append(f"**Authors:** {', '.join(author)}")
        else:
            parts.append(f"**Authors:** {author}")
    if meta.get("date"):
        parts.append(f"**Date:** {meta['date']}")
    if doc.source:
        parts.append(f"**Source:** `{doc.source}`")
    for key, value in meta.items():
        if key in ("title", "author", "date") or not value:
            continue
        key_title = key.replace("_", " ").title()
        if isinstance(value, list):
            value_str = "`, `".join(map(str, value))
            parts.append(f"**{key_title}:** `{value_str}`")
        else:
            parts.append(f"**{key_title}:** {value}")
    if include_text and doc.markdown:
        parts.append("\n---\n" + doc.markdown)
    return "\n".join(parts)


# --- Tool ---


class MilvusSearchTool:
    """Tool for semantic search on Milvus. Single search; no text => filter-only query."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    def get_definition(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": "milvus_search",
                "description": "Performs a semantic search. If no text or queries are provided, returns documents matching the filters (query by filter).",
                "parameters": {
                    "type": "object",
                    "required": [],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Query text for semantic search. If omitted and queries is empty, search is treated as filter-only query.",
                        },
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative to text: list of query strings (first non-empty used).",
                        },
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter expressions to apply",
                            "default": [],
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return.",
                            "default": 100,
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Milvus collection name. Default from env IRAD_COLLECTION_NAME.",
                        },
                        "partition_name": {
                            "type": "string",
                            "description": "Partition name; if omitted, all partitions are searched.",
                        },
                    },
                },
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        logging.info(
            "milvus_search_execute collection_name=%s limit=%s token_present=%s",
            arguments.get("collection_name"),
            arguments.get("limit"),
            bool((arguments.get("_metadata") or {}).get("milvus_token") or self._token),
        )
        metadata = arguments.get("_metadata") or {}
        token = metadata.get("milvus_token") or self._token
        if not token:
            return json.dumps({
                "error": "Milvus token is required. Provide via metadata.milvus_token or tool init.",
                "error_type": "MissingToken",
            })

        text = arguments.get("text")
        queries = arguments.get("queries", [])
        filters = arguments.get("filters", [])
        limit = arguments.get("limit")
        collection_name = arguments.get("collection_name")
        partition_name = arguments.get("partition_name")

        try:
            search_results = await search_async(
                text=text,
                queries=queries,
                filters=filters,
                collection_name=collection_name,
                partition_name=partition_name,
                limit=limit,
                token=token,
            )
        except CollectionNotFoundError as e:
            return json.dumps({"error": str(e), "error_type": "CollectionNotFoundError"})
        except RuntimeError as e:
            return json.dumps({"error": str(e), "error_type": "RuntimeError"})
        except ValueError as e:
            return json.dumps({"error": str(e), "error_type": "ValueError"})
        except Exception as e:
            logging.exception("Search tool error: %s", e)
            return json.dumps({"error": str(e), "error_type": type(e).__name__})

        documents = consolidate_documents(search_results)
        try:
            rendered = [render_document(doc, include_text=True) for doc in documents]
            return json.dumps(rendered, indent=2)
        except Exception as e:
            logging.error("Failed to render search results: %s", e)
            return json.dumps({"error": str(e), "error_type": "RenderingError"})


async def search_async(
    text: str | None,
    queries: list[str],
    filters: list[str],
    collection_name: str | None,
    partition_name: str | None,
    limit: int | None,
    token: str,
) -> list[SearchResult]:
    """Run sync search in executor (db.search is sync and may call embedder)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _get_search_executor(),
        lambda: search(
            text=text,
            queries=queries or None,
            filters=filters,
            collection_name=collection_name,
            partition_name=partition_name,
            limit=limit if isinstance(limit, int) and limit > 0 else 100,
            token=token,
        ),
    )
