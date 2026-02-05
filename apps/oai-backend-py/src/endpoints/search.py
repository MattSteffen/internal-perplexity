"""Search endpoint handler."""

import logging

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from src.milvus_client import MilvusClientContext
from src.tools import milvus_search

logger = logging.getLogger(__name__)

try:
    from crawler.document import Document
except ImportError:
    raise ImportError("crawler package not available")


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


async def search(
    request: SearchRequest,
    context: MilvusClientContext,
) -> SearchResponse:
    """Handle search requests.

    Delegates to the Milvus search tool for both semantic and filter-only queries,
    then consolidates results into crawler Document objects.

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
    try:
        search_results = await milvus_search.search_async(
            text=(request.text or "").strip() or None,
            queries=[],
            filters=request.filters,
            collection_name=request.collection,
            partition_name=None,
            token=context.token,
        )
        combined = milvus_search.consolidate_documents(search_results)
        return SearchResponse(results=combined, total=len(combined))
    except milvus_search.CollectionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        if "has no description" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        logger.exception("Search endpoint error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception("Search endpoint error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e
