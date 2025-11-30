"""Crawler configuration management.

This module contains the main CrawlerConfig class that orchestrates all crawler
subsystem configurations including embeddings, LLMs, database, converter,
extractor, and chunking settings.
"""

from typing import Any

from pydantic import BaseModel, Field

from ..chunker import ChunkingConfig
from ..converter import ConverterConfig
from ..extractor.extractor import MetadataExtractorConfig
from ..llm.embeddings import EmbedderConfig
from ..llm.llm import LLMConfig
from ..vector_db import CollectionDescription, DatabaseClientConfig


class CrawlerConfig(BaseModel):
    """Configuration for the document crawler with Pydantic validation.

    This class provides type-safe configuration management for the crawler system,
    with automatic validation and serialization capabilities.
    """
    name: str = Field(..., description="Name of the crawler pipeline")

    embeddings: EmbedderConfig = Field(..., description="Configuration for the embedding model")
    llm: LLMConfig = Field(..., description="Configuration for the main LLM used for metadata extraction")
    vision_llm: LLMConfig = Field(..., description="Configuration for the vision LLM used for image processing")
    database: DatabaseClientConfig = Field(..., description="Configuration for the vector database")
    converter: ConverterConfig = Field(
        ...,
        description="Configuration for document conversion to markdown",
    )
    extractor: MetadataExtractorConfig = Field(
        ...,
        description="Configuration for metadata extraction",
    )
    chunking: ChunkingConfig = Field(
        ...,
        description="Configuration for text chunking",
    )
    metadata_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for metadata validation")
    temp_dir: str = Field(
        default="tmp/",
        min_length=1,
        description="Temporary directory for caching processed documents",
    )
    benchmark: bool = Field(default=False, description="Whether to run benchmarking after crawling")
    generate_benchmark_questions: bool = Field(
        default=False,
        description="Generate benchmark questions during metadata extraction",
    )
    num_benchmark_questions: int = Field(
        default=3,
        gt=0,
        description="Number of benchmark questions to generate per document",
    )
    security_groups: list[str] | None = Field(
        default=None,
        description="List of security groups for RBAC access control. If provided, the user must have this role to see the documents.",
    )
    model_config = {"validate_assignment": True}

    @classmethod
    def create(
        cls,
        embeddings: EmbedderConfig,
        llm: LLMConfig,
        vision_llm: LLMConfig,
        database: DatabaseClientConfig,
        converter: ConverterConfig | None = None,
        extractor: MetadataExtractorConfig | None = None,
        chunking: ChunkingConfig | None = None,
        metadata_schema: dict[str, Any] | None = None,
        temp_dir: str = "tmp/",
        benchmark: bool = False,
        generate_benchmark_questions: bool = False,
        num_benchmark_questions: int = 3,
        security_groups: list[str] | None = None,
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig with type-safe parameters."""
        return cls(
            embeddings=embeddings,
            llm=llm,
            vision_llm=vision_llm,
            database=database,
            converter=converter,
            extractor=extractor,
            chunking=chunking,
            metadata_schema=metadata_schema or {},
            temp_dir=temp_dir,
            benchmark=benchmark,
            generate_benchmark_questions=generate_benchmark_questions,
            num_benchmark_questions=num_benchmark_questions,
            security_groups=security_groups,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CrawlerConfig":
        """Create a CrawlerConfig from a dictionary configuration.

        This method provides backward compatibility with dictionary-based configurations
        while leveraging Pydantic's validation capabilities.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Validated CrawlerConfig instance

        Example:
            >>> config = CrawlerConfig.from_dict({
            ...     "embeddings": {"provider": "ollama", "model": "all-minilm:v2", ...},
            ...     "llm": {"model_name": "llama3.2:3b", ...},
            ...     "vision_llm": {"model_name": "llava:latest", ...},
            ...     "database": {"provider": "milvus", "collection": "docs", ...},
            ... })
        """
        # Create a copy to avoid mutating the input
        processed_dict = config_dict.copy()

        # Handle backward compatibility: map "utils" dict to top-level fields
        if "utils" in processed_dict:
            utils = processed_dict.pop("utils")
            if isinstance(utils, dict):
                # Map utils keys to top-level fields if not already present
                if "temp_dir" not in processed_dict and "temp_dir" in utils:
                    processed_dict["temp_dir"] = utils["temp_dir"]
                if "benchmark" not in processed_dict and "benchmark" in utils:
                    processed_dict["benchmark"] = utils["benchmark"]

        # Handle extractor's nested llm config if present (needs special handling)
        if "extractor" in processed_dict and isinstance(processed_dict["extractor"], dict):
            extractor_dict = processed_dict["extractor"]
            if "llm" in extractor_dict and isinstance(extractor_dict["llm"], dict):
                # Convert nested llm dict to LLMConfig instance
                extractor_dict = extractor_dict.copy()
                extractor_dict["llm"] = LLMConfig(**extractor_dict["llm"])
                processed_dict["extractor"] = extractor_dict

        # Use Pydantic's model_validate to handle nested configs automatically
        # This will validate and convert nested dicts to their respective Pydantic models
        try:
            return cls.model_validate(processed_dict)
        except Exception as e:
            # Provide more context for validation errors
            raise ValueError(f"Failed to create CrawlerConfig from dictionary: {str(e)}") from e

    @classmethod
    def from_collection_description(
        cls,
        description: CollectionDescription,
        database_config: DatabaseClientConfig,
    ) -> "CrawlerConfig":
        """Create a CrawlerConfig from a CollectionDescription.

        Args:
            description: CollectionDescription instance containing the config
            database_config: Database configuration (collection name will be used)

        Returns:
            CrawlerConfig instance restored from the collection description

        Raises:
            ValueError: If collection_config_json is None or invalid
        """
        return description.to_crawler_config(database_config)

