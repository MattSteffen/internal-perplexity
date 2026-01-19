"""
Vector database module for document storage and retrieval.

This module provides the database abstraction layer for the crawler,
including the DatabaseClient ABC and Milvus implementation.
"""

from .database_client import (
    BenchmarkResult,
    BenchmarkRunResults,
    CollectionDescription,
    DatabaseBenchmark,
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    SearchResult,
    UpsertResult,
)
from .database_utils import get_db, get_db_benchmark
from .milvus_benchmarks import MilvusBenchmark
from .milvus_client import MilvusDB

# Rebuild CollectionDescription to resolve forward references to CrawlerConfig
# This must be done after CrawlerConfig is fully defined
try:
    from ..config import CrawlerConfig

    CollectionDescription.model_rebuild()
except Exception:
    # If CrawlerConfig is not yet imported, model_rebuild will be called later
    pass

__all__ = [
    # Data models
    "BenchmarkResult",
    "BenchmarkRunResults",
    "CollectionDescription",
    "DatabaseClientConfig",
    "DatabaseDocument",
    "SearchResult",
    "UpsertResult",
    # ABC interfaces
    "DatabaseBenchmark",
    "DatabaseClient",
    # Implementations
    "MilvusBenchmark",
    "MilvusDB",
    # Factory functions
    "get_db",
    "get_db_benchmark",
]
