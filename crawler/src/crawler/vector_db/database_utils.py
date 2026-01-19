"""
Database factory utilities.

This module provides factory functions for creating database clients
and benchmark instances based on provider configuration.
"""

from typing import TYPE_CHECKING

from .database_client import DatabaseBenchmark, DatabaseClient, DatabaseClientConfig
from .milvus_benchmarks import MilvusBenchmark
from .milvus_client import MilvusDB

if TYPE_CHECKING:
    from ..config import CrawlerConfig
    from ..llm.embeddings import Embedder, EmbedderConfig


def get_db(
    config: DatabaseClientConfig,
    dimension: int,
    crawler_config: "CrawlerConfig",
    embedder: "Embedder | None" = None,
) -> DatabaseClient:
    """
    Get a database client based on the provider configuration.

    The returned client is NOT connected. Caller must call connect()
    before performing any operations.

    Args:
        config: Configuration for the database client
        dimension: Dimension of the embeddings
        crawler_config: CrawlerConfig containing collection configuration
        embedder: Optional embedder for search operations (can be set later)

    Returns:
        A DatabaseClient instance (not yet connected)

    Raises:
        ValueError: If the database provider is not supported

    Example:
        >>> db = get_db(config, 384, crawler_config, embedder)
        >>> db.connect(create_if_missing=True)
        >>> results = db.search(["query text"])
    """
    if config.provider == "milvus":
        return MilvusDB(config, dimension, crawler_config, embedder)
    raise ValueError(f"Unsupported database provider: {config.provider}")


def get_db_benchmark(
    db_config: "DatabaseClientConfig",
    embed_config: "EmbedderConfig",
    db: "DatabaseClient | None" = None,
) -> "DatabaseBenchmark":
    """
    Get a database benchmark client based on the provider configuration.

    Args:
        db_config: Configuration for the database client
        embed_config: Configuration for the embedding provider
        db: Optional pre-connected DatabaseClient to use for searching

    Returns:
        A DatabaseBenchmark instance

    Raises:
        ValueError: If the database provider is not supported

    Example:
        >>> benchmarker = get_db_benchmark(db_config, embed_config, db)
        >>> results = benchmarker.run_benchmark()
    """
    if db_config.provider == "milvus":
        # If db is provided and is a MilvusDB, use it directly
        milvus_db = db if isinstance(db, MilvusDB) else None
        return MilvusBenchmark(
            db_config=db_config,
            embed_config=embed_config,
            db=milvus_db,
        )
    raise ValueError(f"Unsupported database provider: {db_config.provider}")
