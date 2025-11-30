from typing import TYPE_CHECKING, Any

from .database_client import DatabaseBenchmark, DatabaseClient, DatabaseClientConfig
from .milvus_benchmarks import MilvusBenchmark
from .milvus_client import MilvusDB

if TYPE_CHECKING:
    from ..config import CrawlerConfig
    from ..llm.embeddings import EmbedderConfig


def get_db(
    config: DatabaseClientConfig,
    dimension: int,
    crawler_config: "CrawlerConfig",
) -> DatabaseClient:
    """
    Get a database client based on the provider.
    Args:
        config: Configuration for the database client.
        dimension: Dimension of the embeddings.
        crawler_config: CrawlerConfig containing collection configuration.
    Returns:
        A DatabaseClient object.
    """
    if config.provider == "milvus":
        return MilvusDB(config, dimension, crawler_config)
    raise ValueError(f"unsupported database provider: {config.provider}")


def get_db_benchmark(db_config: "DatabaseClientConfig", embed_config: "EmbedderConfig") -> "DatabaseBenchmark":
    """
    Get a database benchmark client based on the provider.
    Args:
        db_config: Configuration for the database client.
        embed_config: Configuration for the embedding provider.
    Returns:
        A DatabaseBenchmark object.
    """
    if db_config.provider == "milvus":
        return MilvusBenchmark(db_config=db_config, embed_config=embed_config)
    else:
        raise ValueError(f"Unsupported database provider: {db_config.provider}")
