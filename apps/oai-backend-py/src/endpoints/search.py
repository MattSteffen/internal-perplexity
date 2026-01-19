"""Search endpoint handler."""

import copy
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, RRFRanker

from src.clients.router import client_router
from src.milvus_client import get_milvus_client

logger = logging.getLogger(__name__)

# Import CollectionDescription for parsing collection descriptions
try:
    from crawler.document import Document
    from crawler.vector_db import CollectionDescription, DatabaseDocument
except ImportError:
    raise ImportError("crawler package not available")

# Output fields for search results - matching milvus_benchmarks.py
OUTPUT_FIELDS = [
    "document_id",
    "source",
    "chunk_index",
    "text",
    "str_metadata",
    "security_group",
    "benchmark_questions",
    "title",
    "author",
    "date",
    "keywords",
    "unique_words",
]


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    collection: str = Field(..., min_length=1, description="Name of the Milvus collection to search")
    text: str = Field(default="", description="Query text to search for")
    filters: list[str] = Field(default_factory=list, description="List of filter strings to apply")
    limit: int = Field(default=100, ge=1, le=10000, description="Maximum number of results to return")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: list[Document]
    total: int


def _parse_search_result_to_database_document(entity: dict[str, Any]) -> DatabaseDocument:
    """
    Convert a search result entity to a DatabaseDocument.

    Parses str_metadata and benchmark_questions from JSON strings and constructs
    a DatabaseDocument object with all required fields.

    Args:
        entity: Dictionary containing entity data from Milvus search result

    Returns:
        DatabaseDocument instance
    """
    # Parse metadata from JSON string
    metadata = {}
    if entity.get("str_metadata"):
        try:
            metadata = json.loads(entity["str_metadata"])
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse str_metadata: {e}")
            metadata = {}

    # Parse benchmark_questions from JSON string
    benchmark_questions: list[str] | None = None
    if entity.get("benchmark_questions"):
        try:
            benchmark_questions = json.loads(entity["benchmark_questions"])
            if not isinstance(benchmark_questions, list):
                benchmark_questions = None
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse benchmark_questions: {e}")
            benchmark_questions = None

    # Get security_group, defaulting to ["public"] if not present
    security_group = entity.get("security_group", ["public"])
    if not isinstance(security_group, list):
        security_group = ["public"]

    # Create DatabaseDocument - note: text_embedding is required but not in search results
    # We'll use an empty list as placeholder since embeddings aren't needed for Document construction
    return DatabaseDocument(
        document_id=entity["document_id"],
        text=entity.get("text", ""),
        chunk_index=entity.get("chunk_index", 0),
        source=entity.get("source", ""),
        metadata=metadata,
        security_group=security_group,
        benchmark_questions=benchmark_questions,
        text_embedding=[],  # Placeholder - not needed for Document.from_database_documents
    )


def _combine_search_results_to_documents(
    database_documents: list[DatabaseDocument],
) -> list[Document]:
    """
    Combine database documents with the same document_id into Document objects.

    Groups database documents by document_id, sorts chunks by chunk_index,
    and creates Document objects with all chunks combined.

    Args:
        database_documents: List of DatabaseDocument objects from search results

    Returns:
        List of Document objects, one per unique document_id
    """
    # Group by document_id
    documents_by_id: dict[str, list[DatabaseDocument]] = defaultdict(list)
    for db_doc in database_documents:
        documents_by_id[db_doc.document_id].append(db_doc)

    # Combine documents with same document_id
    combined_documents = []
    for document_id, db_docs in documents_by_id.items():
        # Sort by chunk_index to ensure correct order
        db_docs.sort(key=lambda x: x.chunk_index)

        # Use Document.from_database_documents to combine chunks
        # Create a temporary Document instance to call the instance method
        temp_doc = Document(
            document_id=db_docs[0].document_id,
            source=db_docs[0].source,
        )
        document = temp_doc.from_database_documents(db_docs, include_chunks=False)
        combined_documents.append(document)

    return combined_documents


async def search(
    request: SearchRequest,
    user: dict[str, Any],
) -> SearchResponse:
    """Handle search requests.

    curl -X POST http://localhost:8000/v1/search \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "collection": "my_collection",
        "text": "query text",
        "filters": ["title == \"example\""],
        "limit": 100
      }'
    """
    token: str = user.get("milvus_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Milvus token is required")

    # Extract username from token for security group filtering
    username = user.get("username", "")
    if not username:
        # Try to extract from token format "username:password"
        if ":" in token:
            username = token.split(":", 1)[0]
        else:
            raise HTTPException(status_code=401, detail="Could not extract username from token")

    try:
        # Get Milvus client
        client = get_milvus_client(token)

        # Check if collection exists
        if not client.has_collection(collection_name=request.collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection}' not found",
            )

        # Load collection
        client.load_collection(collection_name=request.collection)

        # Get collection description to extract embedding model
        collection_info = client.describe_collection(request.collection)
        description_str = collection_info.get("description", "")
        if not description_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Collection '{request.collection}' does not have a description with pipeline config",
            )

        # Parse collection description
        try:
            collection_desc = CollectionDescription.from_json(description_str)
            if not collection_desc:
                raise ValueError("Failed to parse collection description")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse collection description: {str(e)}",
            ) from e

        # Extract embedding model from collection config
        embedding_model = collection_desc.collection_config.embeddings.model

        # Get user roles for security group filtering
        try:
            user_info = client.describe_user(username)
            user_roles = list(user_info.get("roles", []))
        except Exception as e:
            logger.warning(f"Failed to get user roles for {username}: {e}")
            user_roles = []

        # Build filter string with security group filtering
        filters = copy.deepcopy(request.filters)
        filters.insert(
            0,
            f"array_contains_any(security_group, {user_roles})",
        )
        filter_str = " and ".join(filters)

        # Generate embedding for query text
        try:
            embedding_response = await client_router.create_embedding(
                model=embedding_model,
                input=request.text,
            )
            # Extract embedding vector from response
            if not embedding_response.data or len(embedding_response.data) == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate embedding: empty response",
                )
            embedding = embedding_response.data[0].embedding
        except Exception as e:
            logger.exception(f"Failed to generate embedding: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate embedding: {str(e)}",
            ) from e

        # Build search requests for hybrid search
        search_requests = []

        # Add dense vector search request
        search_requests.append(
            AnnSearchRequest(
                data=[embedding],
                anns_field="text_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                expr=filter_str,
                limit=10,
            )
        )

        # Add sparse search requests
        search_requests.append(
            AnnSearchRequest(
                data=[request.text],
                anns_field="text_sparse_embedding",
                param={"drop_ratio_search": 0.2},
                expr=filter_str,
                limit=10,
            )
        )
        search_requests.append(
            AnnSearchRequest(
                data=[request.text],
                anns_field="metadata_sparse_embedding",
                param={"drop_ratio_search": 0.2},
                expr=filter_str,
                limit=10,
            )
        )

        # Perform hybrid search
        try:
            ranker = RRFRanker(k=100)
            results = client.hybrid_search(
                collection_name=request.collection,
                reqs=search_requests,
                ranker=ranker,
                output_fields=OUTPUT_FIELDS,
                limit=request.limit,
            )

            # Process results: convert to DatabaseDocument, then combine into Documents
            database_documents = []
            if results:
                for doc in results[0]:
                    entity = doc.entity.to_dict()
                    try:
                        db_doc = _parse_search_result_to_database_document(entity)
                        database_documents.append(db_doc)
                    except Exception as e:
                        logger.warning(f"Failed to parse search result: {e}")
                        continue

            # Combine database documents with same document_id into Document objects
            combined_documents = _combine_search_results_to_documents(database_documents)

            return SearchResponse(results=combined_documents, total=len(combined_documents))

        except Exception as e:
            logger.exception(f"Failed to perform search: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Search endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e
