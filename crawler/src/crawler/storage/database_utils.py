from .database_client import DatabaseClient, DatabaseClientConfig, DatabaseBenchmark
from .milvus_client import MilvusDB
from .milvus_benchmarks import MilvusBenchmark
from typing import Dict

from ..processing.embeddings import EmbedderConfig


def get_db(
    config: DatabaseClientConfig, dimension: int, metadata: Dict[str, any]
) -> DatabaseClient:
    if config.provider == "milvus":
        return MilvusDB(config, dimension, metadata)
    raise ValueError(f"unsupported database provider: {config.provider}")


def get_db_benchmark(
    db_config: "DatabaseClientConfig", embed_config: "EmbedderConfig"
) -> "DatabaseBenchmark":
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
