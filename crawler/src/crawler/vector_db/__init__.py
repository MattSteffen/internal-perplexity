from .database_client import (
    CollectionDescription,
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
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
    "CollectionDescription",
    "DatabaseClient",
    "DatabaseClientConfig",
    "DatabaseDocument",
    "MilvusDB",
    "MilvusBenchmark",
    "get_db",
    "get_db_benchmark",
]
