"""
Crawler - Document Processing and Vector Database System

This package provides a modular system for crawling, processing, and indexing documents
into vector databases for retrieval-augmented generation (RAG) applications.
"""

# Re-export main classes from submodules for convenience
from .crawler.main import Crawler, CrawlerConfig, RESERVED, sanitize_metadata
from .crawler.processing import (
    Converter, ConverterConfig, create_converter,
    Extractor, BasicExtractor, MultiSchemaExtractor,
    LLM, LLMConfig, get_llm,
    Embedder, EmbedderConfig, get_embedder
)
from .crawler.storage import (
    DatabaseClient, DatabaseClientConfig, DatabaseDocument,
    MilvusDB, get_db, get_db_benchmark
)
from .crawler.config_defaults import *

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
    "Extractor",
    "BasicExtractor",
    "MultiSchemaExtractor",
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
