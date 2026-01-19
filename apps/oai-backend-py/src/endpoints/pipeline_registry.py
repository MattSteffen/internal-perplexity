"""Pipeline registry for managing predefined crawler configurations.

This module provides template CrawlerConfigs for common document processing use cases.
Templates define default settings for LLMs, embeddings, chunking, and metadata extraction.
Database credentials are not included - they are provided by the user at runtime.
"""

import os
from collections.abc import Callable
from typing import Any

from crawler import CrawlerConfig
from crawler.chunker import ChunkingConfig
from crawler.converter import ConverterConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig

# --- Default Configuration Values (from environment or fallbacks) ---

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))


# --- Metadata Schemas ---

STANDARD_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "author", "date", "keywords"],
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 2550,
            "description": "The title of the document.",
        },
        "author": {
            "type": "array",
            "description": "List of authors.",
            "items": {"type": "string", "maxLength": 255},
        },
        "date": {
            "type": "integer",
            "description": "Publication year (YYYY format).",
            "minimum": 1900,
            "maximum": 2100,
        },
        "keywords": {
            "type": "array",
            "description": "Keywords describing the document content.",
            "items": {"type": "string", "maxLength": 100},
        },
        "description": {
            "type": "string",
            "maxLength": 5000,
            "description": "Brief summary of the document.",
        },
    },
}

ACADEMIC_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "author", "date", "keywords", "unique_words"],
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 2550,
            "description": "The official title of the document.",
        },
        "author": {
            "type": "array",
            "description": "All authors of the document.",
            "items": {"type": "string", "maxLength": 255},
        },
        "date": {
            "type": "integer",
            "description": "Publication year (YYYY format).",
            "minimum": 1900,
            "maximum": 2100,
        },
        "keywords": {
            "type": "array",
            "description": "Terms categorizing the document's subject matter.",
            "items": {"type": "string", "maxLength": 100},
        },
        "unique_words": {
            "type": "array",
            "description": "Domain-specific or technical terms from the document.",
            "items": {"type": "string", "maxLength": 100},
            "minItems": 0,
        },
        "description": {
            "type": "string",
            "maxLength": 15000,
            "description": "Overview of the document's content, arguments, and findings.",
        },
        "summary_item_1": {
            "type": "string",
            "maxLength": 5000,
            "description": "Summary of the primary topic or central argument.",
        },
        "summary_item_2": {
            "type": "string",
            "maxLength": 5000,
            "description": "Summary of a secondary topic if present.",
        },
    },
}


# --- Template Factory Functions ---


def create_standard_template() -> CrawlerConfig:
    """Create a standard template for general document processing.

    This template is suitable for most document types with basic metadata extraction.
    Uses moderate chunk sizes and standard LLM settings.

    Returns:
        CrawlerConfig with standard processing settings
    """
    embeddings = EmbedderConfig(
        provider="ollama",
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    llm = LLMConfig(
        provider="ollama",
        base_url=OLLAMA_BASE_URL,
        model_name=OLLAMA_LLM_MODEL,
        structured_output="tools",
    )

    vision_llm = LLMConfig(
        provider="ollama",
        base_url=OLLAMA_BASE_URL,
        model_name=OLLAMA_VISION_MODEL,
    )

    # Placeholder database config - collection/credentials set at runtime
    database = DatabaseClientConfig(
        provider="milvus",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        username="placeholder",
        password="placeholder",
        collection="placeholder",
        recreate=False,
        collection_description="Standard document collection",
    )

    converter = ConverterConfig(
        max_workers=2,
        vlm_config=vision_llm,
    )

    extractor = MetadataExtractorConfig(
        json_schema=STANDARD_METADATA_SCHEMA,
        context="General document collection",
        llm=llm,
    )

    chunking = ChunkingConfig(
        chunk_size=2000,
        overlap=200,
    )

    return CrawlerConfig(
        name="standard",
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunking=chunking,
        metadata_schema=STANDARD_METADATA_SCHEMA,
        temp_dir="tmp/",
        use_cache=True,
        benchmark=False,
        generate_benchmark_questions=False,
        security_groups=["public"],
    )


def create_academic_template() -> CrawlerConfig:
    """Create an academic template optimized for research papers.

    This template uses larger chunks suitable for academic papers, extracts
    richer metadata including summaries and unique terminology.

    Returns:
        CrawlerConfig with academic paper processing settings
    """
    embeddings = EmbedderConfig(
        provider="ollama",
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    llm = LLMConfig(
        provider="ollama",
        base_url=OLLAMA_BASE_URL,
        model_name=OLLAMA_LLM_MODEL,
        structured_output="tools",
    )

    vision_llm = LLMConfig(
        provider="ollama",
        base_url=OLLAMA_BASE_URL,
        model_name=OLLAMA_VISION_MODEL,
    )

    # Placeholder database config - collection/credentials set at runtime
    database = DatabaseClientConfig(
        provider="milvus",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        username="placeholder",
        password="placeholder",
        collection="placeholder",
        recreate=False,
        collection_description="Academic research paper collection",
    )

    converter = ConverterConfig(
        max_workers=2,
        vlm_config=vision_llm,
    )

    extractor = MetadataExtractorConfig(
        json_schema=ACADEMIC_METADATA_SCHEMA,
        context="Academic research papers and publications",
        llm=llm,
    )

    # Larger chunks for academic content
    chunking = ChunkingConfig(
        chunk_size=10000,
        overlap=500,
    )

    return CrawlerConfig(
        name="academic",
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunking=chunking,
        metadata_schema=ACADEMIC_METADATA_SCHEMA,
        temp_dir="tmp/",
        use_cache=True,
        benchmark=False,
        generate_benchmark_questions=True,
        num_benchmark_questions=3,
        security_groups=["public"],
    )


# --- Pipeline Registry Class ---


class PipelineRegistry:
    """Registry for managing predefined document processing pipelines.

    Pipelines are registered by name with factory functions that return
    CrawlerConfig instances. This allows for easy retrieval and extension
    of predefined processing configurations.
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline registry."""
        self._pipelines: dict[str, Callable[[], CrawlerConfig]] = {}

    def register(self, name: str, config_factory: Callable[[], CrawlerConfig]) -> None:
        """Register a pipeline configuration factory.

        Args:
            name: Unique name identifier for the pipeline
            config_factory: Function that returns a CrawlerConfig instance

        Raises:
            ValueError: If pipeline name already exists
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered")
        self._pipelines[name] = config_factory

    def get_config(self, name: str) -> CrawlerConfig:
        """Get a pipeline configuration by name.

        Args:
            name: Pipeline name identifier

        Returns:
            CrawlerConfig instance for the pipeline

        Raises:
            KeyError: If pipeline name is not found
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found. Available: {list(self._pipelines.keys())}")
        return self._pipelines[name]()

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names.

        Returns:
            List of pipeline name strings
        """
        return list(self._pipelines.keys())

    def has_pipeline(self, name: str) -> bool:
        """Check if a pipeline is registered.

        Args:
            name: Pipeline name identifier

        Returns:
            True if pipeline exists, False otherwise
        """
        return name in self._pipelines

    def get_pipeline_info(self) -> list[dict[str, Any]]:
        """Get information about all registered pipelines.

        Returns:
            List of dicts with pipeline name, description, and metadata schema
        """
        info = []
        for name in self._pipelines:
            config = self._pipelines[name]()
            info.append(
                {
                    "name": name,
                    "description": config.database.collection_description or "",
                    "metadata_schema": config.metadata_schema,
                    "chunk_size": config.chunking.chunk_size,
                    "embedding_model": config.embeddings.model,
                    "llm_model": config.llm.model_name,
                }
            )
        return info


# --- Global Registry Instance ---

_pipeline_registry = PipelineRegistry()

# Register default templates
_pipeline_registry.register("standard", create_standard_template)
_pipeline_registry.register("academic", create_academic_template)


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry instance.

    Returns:
        The global PipelineRegistry instance
    """
    return _pipeline_registry
