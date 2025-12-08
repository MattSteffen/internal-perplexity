"""
Crawler - Document Processing and Vector Database System

This package provides a modular system for crawling, processing, and indexing documents
into vector databases for retrieval-augmented generation (RAG) applications.
"""

# Re-export main classes from submodules for convenience, based on new file structure

from .converter import (
    Converter,
    ConverterConfig,
    create_converter,
)
from .extractor import (
    MetadataExtractor,
    MetadataExtractorConfig,
)
from .llm.embeddings import (
    Embedder,
    EmbedderConfig,
    get_embedder,
)
from .llm.llm import (
    LLM,
    LLMConfig,
    get_llm,
)
from .config import CrawlerConfig
from .main import Crawler, sanitize_metadata
from .vector_db import (
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    get_db,
    get_db_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Crawler",
    "CrawlerConfig",
    "sanitize_metadata",
    # Processing components
    "Converter",
    "ConverterConfig",
    "create_converter",
    "MetadataExtractor",
    "MetadataExtractorConfig",
    "LLM",
    "LLMConfig",
    "get_llm",
    "Embedder",
    "EmbedderConfig",
    "get_embedder",
    # Storage components
    "DatabaseClient",
    "DatabaseClientConfig",
    "DatabaseDocument",
    "MilvusDB",
    "get_db",
    "get_db_benchmark",
]
