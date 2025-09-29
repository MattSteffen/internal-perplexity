"""
Configuration package for the document crawler system.

This package provides centralized configuration management, validation,
and default settings for all components of the crawler system.
"""

from .config_defaults import *
from .validator import ConfigValidator, ValidationError
from .loader import load_config_from_file, load_config_from_env

__all__ = [
    # Config defaults
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
    # Validation
    "ConfigValidator",
    "ValidationError",
    # Loading
    "load_config_from_file",
    "load_config_from_env",
]
