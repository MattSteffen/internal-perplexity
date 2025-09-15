from .main import Crawler, CrawlerConfig
from .config_defaults import *


__all__ = [
    # crawler.py
    "Crawler",
    "CrawlerConfig",
    # config defaults
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
