from .database_client import DatabaseClient, DatabaseClientConfig
from .milvus_client import MilvusDB
from typing import Dict


def get_db(
    config: DatabaseClientConfig, dimension: int, metadata: Dict[str, any]
) -> DatabaseClient:
    if config.provider == "milvus":
        return MilvusDB(config, dimension, metadata)
    raise ValueError(f"unsupported database provider: {config.provider}")
