"""Milvus search tool implementation for semantic document retrieval."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import ollama
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from src.config import radchat_config


class CollectionNotFoundError(Exception):
    """Raised when a Milvus collection does not exist."""

    def __init__(self, collection_name: str) -> None:
        """Initialize the exception.

        Args:
            collection_name: Name of the collection that was not found.
        """
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' does not exist.")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Pydantic Models for Type Safety ---


class MilvusDocumentMetadata(BaseModel):
    """Metadata for a Milvus document."""

    title: str | None = ""
    author: list[str] | str | None = Field(default_factory=list)
    date: int | None = 0
    keywords: list[str] | None = Field(default_factory=list)
    unique_words: list[str] | None = Field(default_factory=list)

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class MilvusDocument(BaseModel):
    """Milvus document with metadata and content."""

    id: Any | None = -1
    document_id: str
    text: str
    chunk_index: int
    source: str
    security_group: list[str] = Field(default_factory=lambda: ["public"])
    metadata: MilvusDocumentMetadata = Field(default_factory=MilvusDocumentMetadata)
    distance: float | None = 1.0

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MilvusDocument":
        """Create a document from a dict (with validation).

        Args:
            data: Dictionary with document data.

        Returns:
            MilvusDocument instance.
        """
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Dump the model as a dict.

        Returns:
            Dictionary representation of the document.
        """
        return self.model_dump()  # type: ignore[no-any-return]

    def render(self, include_text: bool = True) -> str:
        """Render the document to markdown format.

        Args:
            include_text: If True, includes the full document text. If False, only metadata.

        Returns:
            Markdown-formatted string representation of the document.
        """
        parts = []

        # Title
        if self.metadata.title:
            parts.append(f"### {self.metadata.title}")

        # Authors
        if self.metadata.author:
            authors = self.metadata.author
            if isinstance(authors, list):
                parts.append(f"**Authors:** {', '.join(authors)}")
            else:
                parts.append(f"**Authors:** {authors}")

        # Date
        if self.metadata.date:
            parts.append(f"**Date:** {self.metadata.date}")

        # Source
        if self.source:
            parts.append(f"**Source:** `{self.source}` (Chunk: {self.chunk_index})")

        # Dynamically render other metadata fields, excluding those already handled or empty
        fields_to_ignore = {
            "title",
            "author",
            "date",
            "keywords",
            "unique_words",
        }

        other_metadata = []
        for key, value in self.metadata.model_dump().items():
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
        if include_text and self.text:
            parts.append("\n---\n" + self.text)

        return "\n".join(parts)


# --- Core Functions ---


def connect_milvus(
    token: str | None = None,
    collection_name: str | None = None,
) -> MilvusClient:
    """Connect to Milvus and load the specified collection.

    Args:
        token: Milvus authentication token in format 'username:password' (uses config default if not provided).
        collection_name: Collection name (uses config default if not provided).

    Returns:
        MilvusClient instance.

    Raises:
        CollectionNotFoundError: If the collection does not exist.
        Exception: If connection or loading fails.
    """

    # If no token provided, use config defaults (backward compatibility)
    if token is None:
        username = radchat_config.milvus.username
        password = radchat_config.milvus.password
        token = f"{username}:{password}"

    collection_name = collection_name or radchat_config.milvus.collection_name
    uri = f"http://{radchat_config.milvus.host}:{radchat_config.milvus.port}"
    try:
        client = MilvusClient(uri=uri, token=token)
    except Exception as e:
        error_msg = f"Failed to connect to Milvus at {uri}: {str(e)}. Check if Milvus is running and the connection settings are correct."
        logging.error(error_msg)
        raise ConnectionError(error_msg) from e

    try:
        if not client.has_collection(collection_name=collection_name):
            raise CollectionNotFoundError(collection_name)
    except CollectionNotFoundError:
        raise
    except Exception as e:
        error_msg = f"Failed to check if collection '{collection_name}' exists: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        client.load_collection(collection_name=collection_name)
        logging.info(f"Collection '{collection_name}' loaded successfully.")
        return client
    except Exception as e:
        error_msg = f"Failed to load collection '{collection_name}': {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_embedding(text: str | list[str]) -> list[float]:
    """Get embedding for text using Ollama.

    Args:
        text: Text to embed or list of texts to embed.

    Returns:
        Embedding vector.

    Raises:
        RuntimeError: If embedding generation fails with details about the error.
    """
    try:
        client = ollama.Client(host=radchat_config.ollama.base_url)
        response = client.embed(model=radchat_config.ollama.embedding_model, input=text)
        embeddings = response.get("embeddings")
        if embeddings is None:
            error_msg = f"Ollama embedding service returned no embeddings (model: {radchat_config.ollama.embedding_model}, base_url: {radchat_config.ollama.base_url})"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        if not isinstance(embeddings, list):
            error_msg = f"Ollama embedding service returned unexpected format (expected list, got {type(embeddings).__name__})"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        return list[float](embeddings)  # type: ignore[return-value]
    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"Failed to get embedding from Ollama (model: {radchat_config.ollama.embedding_model}, base_url: {radchat_config.ollama.base_url}): {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def perform_search(
    client: MilvusClient,
    queries: list[str] = [],
    filters: list[str] = [],
    collection_name: str | None = None,
    token: str | None = None,
    partition_name: str | None = None,
) -> list[MilvusDocument]:
    """Perform hybrid search (dense + sparse embeddings) on the Milvus collection.

    Args:
        client: MilvusClient instance.
        queries: List of query strings for semantic search.
        filters: List of filter expressions (will not be mutated).
        collection_name: Name of the collection to search.
        token: Authentication token in format 'username:password' for role-based access control.
        partition_name: Optional partition name to search. If not provided, searches all partitions.

    Returns:
        List of matching MilvusDocument objects.
    """
    from src.auth_utils import extract_username_from_token

    collection_name = collection_name or radchat_config.milvus.collection_name

    # Extract username from token
    if token:
        try:
            username = extract_username_from_token(token)
        except ValueError:
            # Fallback to config default if token format is invalid
            username = radchat_config.milvus.username
    else:
        username = radchat_config.milvus.username

    search_requests = []

    # Build filter expression without mutating the original list
    try:
        user_info = client.describe_user(user_name=username)
        user_roles = list[str](user_info.get("roles", []))
    except Exception as e:
        error_msg = f"Failed to retrieve user roles for '{username}': {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    security_filter = f"array_contains_any(security_group, {user_roles})"

    # Combine security filter with user filters
    all_filters = [security_filter] + filters
    filter_expr = " and ".join(all_filters)

    search_configs: list[dict[str, Any]] = [
        {
            "field": "text_sparse_embedding",
            "param": {"drop_ratio_search": radchat_config.search.drop_ratio},
            "data_transform": lambda q: [q],  # type: ignore[assignment]
        },
        {
            "field": "metadata_sparse_embedding",
            "param": {"drop_ratio_search": radchat_config.search.drop_ratio},
            "data_transform": lambda q: [q],  # type: ignore[assignment]
        },
    ]

    # Get index parameters from collection description
    try:
        collection_info = client.describe_collection(collection_name=collection_name)
        indexes = collection_info.get("indexes", [])
        # Find index for text_embedding field
        embedding_index_params: dict[str, Any] = {
            "metric_type": "COSINE",
            "params": {"nprobe": radchat_config.search.nprobe},
        }
        for index in indexes:
            if index.get("field_name") == "text_embedding":
                index_params = index.get("params", {})
                # Extract metric_type if available
                if "metric_type" in index_params:
                    embedding_index_params["metric_type"] = index_params["metric_type"]
                # Update params with index-specific settings if available
                if "params" in index_params:
                    embedding_index_params["params"].update(index_params["params"])
                break
    except Exception as e:
        warning_msg = f"Could not get index parameters from collection '{collection_name}': {str(e)}. Using default parameters (metric_type: COSINE, nprobe: {radchat_config.search.nprobe})."
        logging.warning(warning_msg)
        embedding_index_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": radchat_config.search.nprobe},
        }

    for query in queries:
        try:
            embeddings = get_embedding(query)
            search_requests.append(
                AnnSearchRequest(
                    data=embeddings,
                    anns_field="text_embedding",
                    param=embedding_index_params,
                    expr=filter_expr,
                    limit=radchat_config.search.search_limit,
                )
            )
        except Exception as e:
            error_msg = f"Failed to generate embeddings for query '{query[:100]}...': {str(e)}"
            logging.error(error_msg)
            # Continue with sparse embeddings even if dense embedding fails
            logging.info(f"Continuing search with sparse embeddings only for query: {query[:100]}...")

        for config in search_configs:
            try:
                search_requests.append(
                    AnnSearchRequest(
                        data=config["data_transform"](query),
                        anns_field=config["field"],
                        param=config["param"],
                        expr=filter_expr,
                        limit=radchat_config.search.search_limit,
                    )
                )
            except Exception as e:
                error_msg = f"Failed to create sparse search request for field '{config['field']}': {str(e)}"
                logging.error(error_msg)
                # Continue with other search requests

    if not search_requests:
        # If no search requests, fall back to query with filters
        # Note: perform_query will add the security filter, so pass only user filters
        if len(filters) > 0 or len(user_roles) > 0:
            return perform_query(filters, client, collection_name, token, partition_name)
        return []

    # Build hybrid_search kwargs, including partition_name if provided
    hybrid_search_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "reqs": search_requests,
        "ranker": RRFRanker(k=radchat_config.search.rrf_k),
        "output_fields": radchat_config.search.output_fields,
        "limit": radchat_config.search.hybrid_limit,
    }
    if partition_name:
        hybrid_search_kwargs["partition_names"] = [partition_name]

    try:
        result = client.hybrid_search(**hybrid_search_kwargs)
    except Exception as e:
        error_msg = f"Failed to execute hybrid search on collection '{collection_name}' (partition: {partition_name or 'all'}): {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        documents = []
        for doc in result[0]:
            try:
                documents.append(MilvusDocument(**doc["entity"], distance=doc["distance"]))
            except Exception as e:
                error_msg = f"Failed to create MilvusDocument from search result: {str(e)}. Document data: {doc.get('entity', {})}"
                logging.error(error_msg)
                # Continue processing other documents
                continue
        return consolidate_documents(documents)
    except Exception as e:
        error_msg = f"Failed to process search results: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def perform_query(
    filters: list[str],
    client: MilvusClient,
    collection_name: str | None = None,
    token: str | None = None,
    partition_name: str | None = None,
) -> list[MilvusDocument]:
    """Perform a query on Milvus with filters.

    Args:
        filters: List of filter expressions.
        client: MilvusClient instance.
        collection_name: Name of the collection to query.
        token: Authentication token in format 'username:password' for role-based access control.
        partition_name: Optional partition name to query. If not provided, queries all partitions.

    Returns:
        List of matching MilvusDocument objects.
    """
    from src.auth_utils import extract_username_from_token

    collection_name = collection_name or radchat_config.milvus.collection_name

    # Extract username from token
    if token:
        try:
            username = extract_username_from_token(token)
        except ValueError:
            # Fallback to config default if token format is invalid
            username = radchat_config.milvus.username
    else:
        username = radchat_config.milvus.username

    # Build security filter for role-based access control
    try:
        user_info = client.describe_user(user_name=username)
        user_roles = list(user_info.get("roles", []))
    except Exception as e:
        error_msg = f"Failed to retrieve user roles for '{username}': {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    security_filter = f"array_contains_any(security_group, {user_roles})"

    # Combine security filter with user filters
    all_filters = [security_filter] + filters

    # Build query kwargs, including partition_name if provided
    query_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "filter": " and ".join(all_filters),
        "output_fields": radchat_config.search.output_fields,
        "limit": 100,
    }
    if partition_name:
        query_kwargs["partition_names"] = [partition_name]

    try:
        query_results = client.query(**query_kwargs)
    except Exception as e:
        error_msg = f"Failed to execute query on collection '{collection_name}' (partition: {partition_name or 'all'}) with filter '{query_kwargs['filter']}': {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        documents = []
        for doc in query_results:
            try:
                documents.append(MilvusDocument(**doc))
            except Exception as e:
                error_msg = f"Failed to create MilvusDocument from query result: {str(e)}. Document data: {doc}"
                logging.error(error_msg)
                # Continue processing other documents
                continue
        return consolidate_documents(documents)
    except Exception as e:
        error_msg = f"Failed to process query results: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


def consolidate_documents(documents: list[MilvusDocument]) -> list[MilvusDocument]:
    """Consolidate documents with the same document_id, combining their text and metadata.

    Groups by document_id and combines:
    - Text chunks in order
    - Unique keywords
    - Unique words
    - Unique authors
    - Uses minimum distance

    Args:
        documents: List of documents to consolidate.

    Returns:
        Consolidated list of documents sorted by distance.
    """
    if not documents:
        return []

    # Group by document_id instead of source
    doc_groups = defaultdict(list)
    for doc in documents:
        doc_groups[doc.document_id].append(doc)

    consolidated_docs = []
    for doc_id, docs in doc_groups.items():
        # Sort chunks by index to maintain reading order
        sorted_chunks = sorted(docs, key=lambda d: d.chunk_index)
        base_doc = sorted_chunks[0]

        # Combine text chunks with separator
        combined_text = "\n\n---\n\n".join([d.text for d in sorted_chunks if d.text])

        # Combine unique keywords
        combined_keywords = sorted(list(set(kw for d in sorted_chunks for kw in (d.metadata.keywords or []))))

        # Combine unique words
        combined_unique_words = sorted(list(set(word for d in sorted_chunks for word in (d.metadata.unique_words or []))))

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
        min_distance = min((d.distance for d in sorted_chunks if d.distance is not None), default=None)

        # Create consolidated document
        consolidated_data = base_doc.model_dump()
        consolidated_data.update(
            {
                "text": combined_text,
                "distance": min_distance,
                "chunk_index": 0,  # Represents the consolidated document
            }
        )

        # Update metadata with combined fields
        updated_metadata = consolidated_data["metadata"].copy()
        updated_metadata["keywords"] = combined_keywords
        updated_metadata["unique_words"] = combined_unique_words
        updated_metadata["author"] = combined_authors if combined_authors else updated_metadata.get("author")
        consolidated_data["metadata"] = updated_metadata

        consolidated_docs.append(MilvusDocument(**consolidated_data))

    # Sort the final list of consolidated documents by distance
    return sorted(
        consolidated_docs,
        key=lambda d: d.distance if d.distance is not None else float("inf"),
    )


def build_citations(documents: list[MilvusDocument]) -> list[dict[str, Any]]:
    """Build citation objects from documents for display in the UI.

    Formats citations according to OpenWebUI (OI) citation format:
    - document: Array of content strings
    - metadata: Array of metadata objects with date_accessed, source, author, etc.
    - source: Primary source info with name and url

    Args:
        documents: List of documents to create citations from.

    Returns:
        List of citation dictionaries in OI format.
    """
    consolidated_docs = consolidate_documents(documents)

    citations = []
    for doc in consolidated_docs:
        # Extract author(s) as a string
        author_str = ""
        if doc.metadata.author:
            if isinstance(doc.metadata.author, list):
                author_str = ", ".join(doc.metadata.author)
            else:
                author_str = str(doc.metadata.author)

        # Format publication date from metadata.date if available
        publication_date = ""
        if doc.metadata.date:
            publication_date = str(doc.metadata.date)

        citations.append(
            {
                "source": {"name": doc.source, "url": ""},
                "document": [doc.render(include_text=False)],
                "metadata": [
                    {
                        "date_accessed": datetime.now().isoformat(),
                        "source": doc.metadata.title or doc.source,
                        "author": author_str,
                        "publication_date": publication_date,
                        "url": "",  # URL not available in document metadata
                    }
                ],
                "distance": doc.distance,
            }
        )
    return citations


# --- Tool Implementation ---


class MilvusSearchTool:
    """Tool for performing semantic search on Milvus document collection."""

    def __init__(
        self,
        client: MilvusClient | None = None,
        token: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the Milvus search tool.

        Args:
            client: Optional MilvusClient instance. If None, will connect on first use.
            token: Optional authentication token in format 'username:password'. Uses config default if None.
            collection_name: Optional collection name. Uses config default if None.
        """
        self._client = client
        self._token = token
        self._collection_name = collection_name or radchat_config.milvus.collection_name

    def get_definition(self) -> ChatCompletionToolParam:
        """Get the OpenAI tool definition for the search tool.

        Returns:
            ChatCompletionToolParam with search function schema.
        """
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": "search",
                "description": ("Performs a semantic search using the given queries and optional filters."),
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
                            "description": ("Name of the Milvus collection to search. " "If not provided, uses the default collection."),
                        },
                        "partition_name": {
                            "type": "string",
                            "description": ("Name of the partition to search within the collection. " "If not provided, searches all partitions."),
                        },
                    },
                },
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the search tool with the given arguments.

        Args:
            arguments: Dictionary with 'queries', optionally 'filters',
                'collection_name', and 'partition_name' keys. May also contain
                '_metadata' with 'milvus_token' for authentication.

        Returns:
            JSON string with search results or error message.
        """
        queries = arguments.get("queries", [])
        filters = arguments.get("filters", [])
        collection_name = arguments.get("collection_name") or self._collection_name
        partition_name = arguments.get("partition_name")

        # Get token from metadata if provided, otherwise use instance token
        metadata = arguments.get("_metadata", {})
        token = metadata.get("milvus_token") if isinstance(metadata, dict) else self._token

        # Get or create client with the requested collection
        try:
            client = self.get_client(collection_name=collection_name, token=token)
        except CollectionNotFoundError as e:
            return json.dumps({"error": str(e), "error_type": "CollectionNotFoundError"})
        except Exception as e:
            logging.error(f"Error connecting to Milvus: {e}")
            return json.dumps({"error": f"Unable to connect to Milvus: {str(e)}"})

        # Perform search
        try:
            results = perform_search(
                client=client,
                queries=queries if queries else [],
                filters=filters if filters else [],
                collection_name=collection_name,
                token=token,
                partition_name=partition_name,
            )
        except CollectionNotFoundError as e:
            error_msg = f"Search failed: Collection '{e.collection_name}' does not exist."
            logging.error(error_msg)
            return json.dumps({"error": error_msg, "error_type": "CollectionNotFoundError"})
        except RuntimeError as e:
            error_msg = f"Search execution failed: {str(e)}"
            logging.error(error_msg)
            return json.dumps({"error": error_msg, "error_type": "RuntimeError"})
        except Exception as e:
            error_msg = f"Unexpected error during search execution: {str(e)}"
            logging.error(error_msg)
            return json.dumps({"error": error_msg, "error_type": type(e).__name__})

        # Return rendered documents as JSON
        try:
            rendered = [doc.render(include_text=True) for doc in results]
            return json.dumps(rendered, indent=2)
        except Exception as e:
            error_msg = f"Failed to render search results: {str(e)}"
            logging.error(error_msg)
            return json.dumps({"error": error_msg, "error_type": "RenderingError"})

    def get_client(self, collection_name: str | None = None, token: str | None = None) -> MilvusClient:
        """Get the Milvus client instance, optionally for a specific collection.

        Args:
            collection_name: Optional collection name. If provided and differs from
                the cached collection, will connect/reconnect to the new collection.
                If None, uses the default collection from initialization.
            token: Optional authentication token. If provided, will use this token
                for authentication. Otherwise uses instance token or config defaults.

        Returns:
            MilvusClient instance.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        target_collection = collection_name or self._collection_name
        auth_token = token or self._token

        # If no client exists, connect to the target collection
        if self._client is None:
            self._client = connect_milvus(
                token=auth_token,
                collection_name=target_collection,
            )
            return self._client

        # Client exists - if collection_name was provided and differs, reconnect
        if collection_name is not None and collection_name != self._collection_name:
            # Switching to a different collection - reconnect with the new collection
            self._client = connect_milvus(
                token=auth_token,
                collection_name=target_collection,
            )
        else:
            # Same collection requested, but ensure it exists and is loaded
            if not self._client.has_collection(collection_name=target_collection):
                raise CollectionNotFoundError(target_collection)
            try:
                self._client.load_collection(collection_name=target_collection)
            except Exception as e:
                logging.warning(f"Error ensuring collection is loaded, reconnecting: {e}")
                self._client = connect_milvus(
                    token=auth_token,
                    collection_name=target_collection,
                )

        return self._client
