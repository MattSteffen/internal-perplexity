"""
Centralized configuration defaults for all providers and components.

This module provides default configurations that can be easily imported and used
throughout the crawler system, ensuring consistency and reducing duplication.
"""

from .processing.embeddings import EmbedderConfig
from .processing.llm import LLMConfig
from .storage.database_client import DatabaseClientConfig
from .processing.converter import ConverterConfig
from .processing.extractor import ExtractorConfig

# Ollama provider defaults
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 300.0
DEFAULT_OLLAMA_CTX_LENGTH = 32000

# Embedding model defaults
DEFAULT_OLLAMA_EMBEDDINGS = EmbedderConfig(
    model="all-minilm:v2",
    base_url=DEFAULT_OLLAMA_BASE_URL,
    api_key="",  # Ollama typically doesn't need API key
    provider="ollama"
)

# LLM model defaults
DEFAULT_OLLAMA_LLM = LLMConfig(
    model_name="llama3.2:3b",
    base_url=DEFAULT_OLLAMA_BASE_URL,
    system_prompt=None,
    ctx_length=DEFAULT_OLLAMA_CTX_LENGTH,
    default_timeout=DEFAULT_OLLAMA_TIMEOUT,
    provider="ollama"
)

# Vision LLM defaults (for document processing with images)
DEFAULT_OLLAMA_VISION_LLM = LLMConfig(
    model_name="llava:latest",  # Common vision model
    base_url=DEFAULT_OLLAMA_BASE_URL,
    system_prompt=None,
    ctx_length=DEFAULT_OLLAMA_CTX_LENGTH,
    default_timeout=DEFAULT_OLLAMA_TIMEOUT,
    provider="ollama"
)

# Database defaults
DEFAULT_MILVUS_CONFIG = DatabaseClientConfig(
    provider="milvus",
    collection="documents",
    partition=None,
    recreate=False,
    collection_description="Document collection for RAG system",
    host="localhost",
    port=19530,
    username="root",
    password="Milvus"
)

# Converter defaults
DEFAULT_CONVERTER_CONFIG = ConverterConfig(
    type="markitdown",
    vision_llm=DEFAULT_OLLAMA_VISION_LLM
)

# Extractor defaults
DEFAULT_EXTRACTOR_CONFIG = ExtractorConfig(
    type="basic",
    llm=DEFAULT_OLLAMA_LLM,
    metadata_schema=None
)

# Crawler defaults
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_TEMP_DIR = "tmp/"
DEFAULT_BENCHMARK = False

# Metadata schema defaults
DEFAULT_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 512},
        "author": {"type": "string", "maxLength": 256},
        "date": {"type": "string", "maxLength": 32},
        "keywords": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 20},
        "summary": {"type": "string", "maxLength": 2048},
    }
}


__all__ = [
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
