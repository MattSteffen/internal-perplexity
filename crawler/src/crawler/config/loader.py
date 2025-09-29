"""
Configuration loading utilities.

This module provides functions to load configuration from various sources
including files, environment variables, and other formats.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    # When run as part of the crawler package
    from ..processing.embeddings import EmbedderConfig
    from ..processing.llm import LLMConfig
    from ..processing.converter import ConverterConfig
    from ..processing.extractor import ExtractorConfig
    from ..storage.database_client import DatabaseClientConfig
    from ..storage.database_utils import get_db
    from ..main import CrawlerConfig
except ImportError:
    # When run standalone (e.g., for testing)
    from processing.embeddings import EmbedderConfig
    from processing.llm import LLMConfig
    from processing.converter import ConverterConfig
    from processing.extractor import ExtractorConfig
    from storage.database_client import DatabaseClientConfig
    from main import CrawlerConfig


def load_config_from_file(config_path: str) -> CrawlerConfig:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        CrawlerConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    return load_config_from_dict(config_data)


def load_config_from_env() -> CrawlerConfig:
    """
    Load configuration from environment variables.

    Environment variables should be prefixed with 'CRAWLER_':
    - CRAWLER_EMBEDDING_MODEL
    - CRAWLER_EMBEDDING_BASE_URL
    - CRAWLER_LLM_MODEL
    - CRAWLER_LLM_BASE_URL
    - CRAWLER_VISION_MODEL
    - CRAWLER_VISION_BASE_URL
    - CRAWLER_DATABASE_HOST
    - CRAWLER_DATABASE_PORT
    - CRAWLER_DATABASE_USERNAME
    - CRAWLER_DATABASE_PASSWORD
    - CRAWLER_DATABASE_COLLECTION
    - CRAWLER_CHUNK_SIZE
    - etc.

    Returns:
        CrawlerConfig instance
    """
    # Extract embeddings config
    embeddings = EmbedderConfig.ollama(
        model=os.getenv("CRAWLER_EMBEDDING_MODEL", "all-minilm:v2"),
        base_url=os.getenv("CRAWLER_EMBEDDING_BASE_URL", "http://localhost:11434"),
    )

    # Extract LLM config
    llm = LLMConfig.ollama(
        model_name=os.getenv("CRAWLER_LLM_MODEL", "llama3.2:3b"),
        base_url=os.getenv("CRAWLER_LLM_BASE_URL", "http://localhost:11434"),
    )

    # Extract vision LLM config
    vision_llm = LLMConfig.ollama(
        model_name=os.getenv("CRAWLER_VISION_MODEL", "llava:latest"),
        base_url=os.getenv("CRAWLER_VISION_BASE_URL", "http://localhost:11434"),
    )

    # Extract database config
    database = DatabaseClientConfig.milvus(
        collection=os.getenv("CRAWLER_DATABASE_COLLECTION", "documents"),
        host=os.getenv("CRAWLER_DATABASE_HOST", "localhost"),
        port=int(os.getenv("CRAWLER_DATABASE_PORT", "19530")),
        username=os.getenv("CRAWLER_DATABASE_USERNAME", "root"),
        password=os.getenv("CRAWLER_DATABASE_PASSWORD", "Milvus"),
    )

    # Extract optional parameters
    chunk_size = int(os.getenv("CRAWLER_CHUNK_SIZE", "10000"))
    temp_dir = os.getenv("CRAWLER_TEMP_DIR", "tmp/")
    benchmark = os.getenv("CRAWLER_BENCHMARK", "false").lower() == "true"
    log_level = os.getenv("CRAWLER_LOG_LEVEL", "INFO")

    return CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        chunk_size=chunk_size,
        temp_dir=temp_dir,
        benchmark=benchmark,
        log_level=log_level,
    )


def load_config_from_dict(config_dict: Dict[str, Any]) -> CrawlerConfig:
    """
    Load configuration from a dictionary.

    Args:
        config_dict: Dictionary containing configuration data

    Returns:
        CrawlerConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Extract embeddings config
        embeddings_data = config_dict.get("embeddings", {})
        embeddings = EmbedderConfig.ollama(
            model=embeddings_data.get("model", "all-minilm:v2"),
            base_url=embeddings_data.get("base_url", "http://localhost:11434"),
        )

        # Extract LLM config
        llm_data = config_dict.get("llm", {})
        llm = LLMConfig.ollama(
            model_name=llm_data.get("model", "llama3.2:3b"),
            base_url=llm_data.get("base_url", "http://localhost:11434"),
        )

        # Extract vision LLM config
        vision_data = config_dict.get("vision_llm", {})
        vision_llm = LLMConfig.ollama(
            model_name=vision_data.get("model", "llava:latest"),
            base_url=vision_data.get("base_url", "http://localhost:11434"),
        )

        # Extract database config
        db_data = config_dict.get("database", {})
        database = DatabaseClientConfig.milvus(
            collection=db_data.get("collection", "documents"),
            host=db_data.get("host", "localhost"),
            port=db_data.get("port", 19530),
            username=db_data.get("username", "root"),
            password=db_data.get("password", "Milvus"),
        )

        # Extract optional parameters
        chunk_size = config_dict.get("chunk_size", 10000)
        temp_dir = config_dict.get("temp_dir", "tmp/")
        benchmark = config_dict.get("benchmark", False)
        log_level = config_dict.get("log_level", "INFO")

        # Extract converter config if present
        converter_data = config_dict.get("converter")
        converter = None
        if converter_data:
            converter = ConverterConfig.markitdown(vision_llm=vision_llm)

        # Extract extractor config if present
        extractor_data = config_dict.get("extractor")
        extractor = None
        if extractor_data:
            extractor = ExtractorConfig.basic(llm=llm)

        return CrawlerConfig.create(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter,
            extractor=extractor,
            chunk_size=chunk_size,
            temp_dir=temp_dir,
            benchmark=benchmark,
            log_level=log_level,
        )

    except Exception as e:
        raise ValueError(f"Failed to load configuration from dictionary: {e}") from e


def save_config_to_file(config: CrawlerConfig, config_path: str) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: CrawlerConfig instance to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dictionary
    config_dict = {
        "embeddings": {
            "model": config.embeddings.model,
            "base_url": config.embeddings.base_url,
            "provider": config.embeddings.provider,
        },
        "llm": {
            "model": config.llm.model_name,
            "base_url": config.llm.base_url,
            "provider": config.llm.provider,
        },
        "vision_llm": {
            "model": config.vision_llm.model_name,
            "base_url": config.vision_llm.base_url,
            "provider": config.vision_llm.provider,
        },
        "database": {
            "collection": config.database.collection,
            "host": config.database.host,
            "port": config.database.port,
            "username": config.database.username,
            "password": config.database.password,
            "provider": config.database.provider,
        },
        "converter": {
            "type": config.converter.type if config.converter else "markitdown",
        },
        "extractor": {
            "type": config.extractor.type if config.extractor else "basic",
        },
        "chunk_size": config.chunk_size,
        "temp_dir": config.temp_dir,
        "benchmark": config.benchmark,
        "log_level": config.log_level,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def create_example_config(output_path: str = "example_config.json") -> None:
    """
    Create an example configuration file.

    Args:
        output_path: Path where to save the example configuration
    """
    example_config = {
        "embeddings": {
            "model": "all-minilm:v2",
            "base_url": "http://localhost:11434",
            "provider": "ollama",
        },
        "llm": {
            "model": "llama3.2:3b",
            "base_url": "http://localhost:11434",
            "provider": "ollama",
        },
        "vision_llm": {
            "model": "llava:latest",
            "base_url": "http://localhost:11434",
            "provider": "ollama",
        },
        "database": {
            "collection": "documents",
            "host": "localhost",
            "port": 19530,
            "username": "root",
            "password": "Milvus",
            "provider": "milvus",
        },
        "converter": {"type": "markitdown"},
        "extractor": {},
        "chunk_size": 10000,
        "temp_dir": "tmp/",
        "benchmark": False,
        "log_level": "INFO",
    }

    with open(output_path, "w") as f:
        json.dump(example_config, f, indent=2)

    print(f"📄 Created example configuration file: {output_path}")
