"""Tests for the search endpoint handler."""

import asyncio

import pytest
from fastapi import HTTPException

from src.endpoints import search
from src.milvus_client import MilvusClientContext
from src.tools import milvus_search


def test_search_raises_on_missing_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search should return 400 when collection description is missing."""
    request = search.SearchRequest(
        collection="my_collection",
        text="query",
        filters=[],
        limit=10,
    )
    context = MilvusClientContext(
        client=object(),
        pool=object(),
        cache_key="cache",
        token="user:pass",
        user={"username": "user"},
    )

    async def _raise_missing_description(**_: object) -> list[object]:
        raise RuntimeError(
            "Collection 'my_collection' has no description; cannot run semantic search without pipeline config"
        )

    monkeypatch.setattr(milvus_search, "search_async", _raise_missing_description)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(search.search(request, context))

    assert exc_info.value.status_code == 400
    assert "has no description" in exc_info.value.detail
