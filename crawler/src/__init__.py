"""
Crawler - Document Processing and Vector Database System

This package provides a modular system for crawling, processing, and indexing documents
into vector databases for retrieval-augmented generation (RAG) applications.
"""

# Re-export main classes from submodules for convenience, based on new file structure

from .crawler.main import Crawler, CrawlerConfig, sanitize_metadata

from .crawler.converter import (
    Converter,
    ConverterConfig,
    create_converter,
)

from .crawler.extractor import (
    MetadataExtractor,
    MetadataExtractorConfig,
)

from .crawler.llm.embeddings import (
    Embedder,
    EmbedderConfig,
    get_embedder,
)
from .crawler.llm.llm import (
    LLM,
    LLMConfig,
    get_llm,
)

from .crawler.vector_db import (
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
    "RESERVED",
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
    # Default configurations
    "DEFAULT_OLLAMA_EMBEDDINGS",
    "DEFAULT_OLLAMA_LLM",
    "DEFAULT_OLLAMA_VISION_LLM",
    "DEFAULT_MILVUS_CONFIG",
    "DEFAULT_CONVERTER_CONFIG",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_TEMP_DIR",
    "DEFAULT_BENCHMARK",
    "DEFAULT_METADATA_SCHEMA",
    "DEFAULT_EXTRACTOR_CONFIG",
]
