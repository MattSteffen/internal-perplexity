"""
Comprehensive configuration validation system.

This module provides thorough validation of all configuration aspects including
connectivity tests, schema validation, and compatibility checks.
"""

import asyncio
import logging
import time
import json
import jsonschema
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import httpx
import requests
from urllib.parse import urlparse

try:
    # When run as part of the crawler package
    from ..processing.embeddings import EmbedderConfig, get_embedder
    from ..processing.llm import LLMConfig, get_llm
    from ..processing.converter import ConverterConfig
    from ..storage.database_client import DatabaseClientConfig
    from ..storage.database_utils import get_db
    from .config_defaults import DEFAULT_METADATA_SCHEMA
except ImportError:
    # When run standalone (e.g., for testing)
    from processing.embeddings import EmbedderConfig, get_embedder
    from processing.llm import LLMConfig, get_llm
    from processing.converter import ConverterConfig
    from storage.database_client import DatabaseClientConfig
    from storage.database_utils import get_db

    try:
        from .config_defaults import DEFAULT_METADATA_SCHEMA
    except ImportError:
        import sys
        import os

        # Add the parent directory to the path for standalone imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, parent_dir)
        from config_defaults import DEFAULT_METADATA_SCHEMA


class ValidationError(Exception):
    """Exception raised when configuration validation fails."""

    pass


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    test_name: str
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None


class ConfigValidator:
    """Comprehensive validator for crawler configuration."""

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("ConfigValidator")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.results: List[ValidationResult] = []

    def validate_all(self, config: "CrawlerConfig") -> List[ValidationResult]:
        """Run all validation tests on the configuration."""
        self.logger.info("🚀 Starting comprehensive configuration validation...")

        tests = [
            ("Basic Configuration", self._validate_basic_config),
            ("LLM Connectivity", self._validate_llm_connectivity),
            ("Embedding Model", self._validate_embedding_connectivity),
            ("Vision LLM", self._validate_vision_llm_connectivity),
            ("Database Connection", self._validate_database_connection),
            ("Image Describer", self._validate_image_describer_connectivity),
            ("Metadata Schema", self._validate_metadata_schema),
            ("Converter Configuration", self._validate_converter_config),
            ("Extractor Configuration", self._validate_extractor_config),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                result = test_func(config)
                duration = time.time() - start_time
                result.duration = duration
                self.results.append(result)
                self._log_result(result)
            except Exception as e:
                duration = time.time() - start_time
                result = ValidationResult(
                    test_name=test_name,
                    success=False,
                    message=f"Validation failed with exception: {e}",
                    duration=duration,
                )
                self.results.append(result)
                self._log_result(result)

        return self.results

    def _log_result(self, result: ValidationResult) -> None:
        """Log validation result."""
        if result.success:
            self.logger.info(f"✅ {result.test_name}: {result.message}")
            if result.duration:
                self.logger.debug(f"   Duration: {result.duration:.3f}s")
        else:
            self.logger.error(f"❌ {result.test_name}: {result.message}")

    def _validate_basic_config(self, config: "CrawlerConfig") -> ValidationResult:
        """Validate basic configuration structure."""
        try:
            # Check required fields
            if not hasattr(config, "embeddings"):
                raise ValidationError("Missing embeddings configuration")
            if not hasattr(config, "llm"):
                raise ValidationError("Missing LLM configuration")
            if not hasattr(config, "vision_llm"):
                raise ValidationError("Missing vision LLM configuration")
            if not hasattr(config, "database"):
                raise ValidationError("Missing database configuration")

            # Check chunk size
            if config.chunk_size <= 0:
                raise ValidationError(f"Invalid chunk size: {config.chunk_size}")

            # Check temp directory
            temp_path = Path(config.temp_dir)
            if not temp_path.exists():
                temp_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created temp directory: {temp_path}")

            return ValidationResult(
                test_name="Basic Configuration",
                success=True,
                message="All basic configuration parameters are valid",
                details={
                    "chunk_size": config.chunk_size,
                    "temp_dir": str(temp_path.absolute()),
                    "log_level": config.log_level,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Basic Configuration", success=False, message=str(e)
            )

    def _validate_llm_connectivity(self, config: "CrawlerConfig") -> ValidationResult:
        """Test LLM connectivity and basic functionality."""
        try:
            llm = get_llm(config.llm)

            # Test basic connectivity
            test_prompt = "Hello! Please respond with the word 'OK'."
            response = llm.invoke(test_prompt)

            if not response or "OK" not in response.upper():
                raise ValidationError("LLM did not respond correctly to test prompt")

            return ValidationResult(
                test_name="LLM Connectivity",
                success=True,
                message=f"LLM model '{config.llm.model_name}' is working correctly",
                details={
                    "model": config.llm.model_name,
                    "provider": config.llm.provider,
                    "base_url": config.llm.base_url,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="LLM Connectivity",
                success=False,
                message=f"LLM connectivity test failed: {e}",
            )

    def _validate_embedding_connectivity(
        self, config: "CrawlerConfig"
    ) -> ValidationResult:
        """Test embedding model connectivity and functionality."""
        try:
            embedder = get_embedder(config.embeddings)

            # Test embedding generation
            test_text = "This is a test sentence for embedding."
            embedding = embedder.embed(test_text)

            if not embedding or len(embedding) != embedder.get_dimension():
                raise ValidationError("Embedding dimension mismatch or empty embedding")

            return ValidationResult(
                test_name="Embedding Model",
                success=True,
                message=f"Embedding model '{config.embeddings.model}' is working correctly",
                details={
                    "model": config.embeddings.model,
                    "dimension": len(embedding),
                    "provider": config.embeddings.provider,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Embedding Model",
                success=False,
                message=f"Embedding model test failed: {e}",
            )

    def _validate_vision_llm_connectivity(
        self, config: "CrawlerConfig"
    ) -> ValidationResult:
        """Test vision LLM connectivity for image processing."""
        try:
            llm = get_llm(config.vision_llm)

            # Test vision capability with text-based prompt
            test_prompt = "What colors do you see in this description? Please respond with 'VISION_OK'."
            response = llm.invoke(test_prompt)

            if not response or "VISION_OK" not in response.upper():
                raise ValidationError(
                    "Vision LLM did not respond correctly to test prompt"
                )

            return ValidationResult(
                test_name="Vision LLM",
                success=True,
                message=f"Vision LLM model '{config.vision_llm.model_name}' is working correctly",
                details={
                    "model": config.vision_llm.model_name,
                    "provider": config.vision_llm.provider,
                    "base_url": config.vision_llm.base_url,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Vision LLM",
                success=False,
                message=f"Vision LLM test failed: {e}",
            )

    def _validate_database_connection(
        self, config: "CrawlerConfig"
    ) -> ValidationResult:
        """Test database connection and basic operations."""
        try:
            # Get database client (this will test the connection)
            db = get_db(
                config.database, 384, DEFAULT_METADATA_SCHEMA
            )  # 384 is test dimension

            # Test basic database operations
            test_source = "test_source"
            test_chunk_index = 0

            # Check if collection exists or can be created
            db.create_collection(recreate=False)

            # Test duplicate checking
            is_duplicate = db.check_duplicate(test_source, test_chunk_index)

            return ValidationResult(
                test_name="Database Connection",
                success=True,
                message=f"Database connection to '{config.database.collection}' is working correctly",
                details={
                    "collection": config.database.collection,
                    "host": config.database.host,
                    "port": config.database.port,
                    "provider": config.database.provider,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Database Connection",
                success=False,
                message=f"Database connection test failed: {e}",
            )

    def _validate_image_describer_connectivity(
        self, config: "CrawlerConfig"
    ) -> ValidationResult:
        """Test image describer service connectivity."""
        try:
            # Test if the vision LLM can handle image descriptions
            llm = get_llm(config.vision_llm)

            # This is a basic connectivity test - in practice, you'd test with actual image data
            test_prompt = "If you can process images, please respond with 'IMAGE_OK'."

            # Use tools mode if available for better image processing
            if (
                hasattr(config.vision_llm, "structured_output")
                and config.vision_llm.structured_output == "tools"
            ):
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "describe_image",
                            "description": "Test image description capability",
                            "parameters": {
                                "type": "object",
                                "properties": {"status": {"type": "string"}},
                                "required": ["status"],
                            },
                        },
                    }
                ]
                response = llm.invoke(test_prompt, tools=tools)
            else:
                response = llm.invoke(
                    test_prompt,
                    response_format={
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                )

            if (
                not response
                or not isinstance(response, dict)
                or response.get("status") != "IMAGE_OK"
            ):
                raise ValidationError(
                    "Image describer service not responding correctly"
                )

            return ValidationResult(
                test_name="Image Describer",
                success=True,
                message="Image describer service is accessible",
                details={
                    "vision_model": config.vision_llm.model_name,
                    "base_url": config.vision_llm.base_url,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Image Describer",
                success=False,
                message=f"Image describer test failed: {e}",
            )

    def _validate_metadata_schema(self, config: "CrawlerConfig") -> ValidationResult:
        """Validate metadata schema structure and format."""
        try:
            schema = config.metadata_schema

            if not schema:
                return ValidationResult(
                    test_name="Metadata Schema",
                    success=True,
                    message="No custom metadata schema provided (using defaults)",
                    details={"using_defaults": True},
                )

            # Basic JSON schema validation
            if not isinstance(schema, dict):
                raise ValidationError("Metadata schema must be a dictionary")

            required_keys = ["type", "properties"]
            for key in required_keys:
                if key not in schema:
                    raise ValidationError(
                        f"Metadata schema missing required key: {key}"
                    )

            if schema.get("type") != "object":
                raise ValidationError("Metadata schema root type must be 'object'")

            # Test schema against a sample document
            sample_metadata = {"title": "Test", "author": "Test Author"}
            jsonschema.validate(instance=sample_metadata, schema=schema)

            return ValidationResult(
                test_name="Metadata Schema",
                success=True,
                message="Metadata schema is valid",
                details={
                    "properties_count": len(schema.get("properties", {})),
                    "required_fields": schema.get("required", []),
                },
            )

        except jsonschema.ValidationError as e:
            return ValidationResult(
                test_name="Metadata Schema",
                success=False,
                message=f"Metadata schema validation error: {e.message}",
            )
        except Exception as e:
            return ValidationResult(
                test_name="Metadata Schema",
                success=False,
                message=f"Metadata schema test failed: {e}",
            )

    def _validate_converter_config(self, config: "CrawlerConfig") -> ValidationResult:
        """Validate converter configuration."""
        try:
            if not config.converter:
                return ValidationResult(
                    test_name="Converter Configuration",
                    success=True,
                    message="No converter configuration provided (will use defaults)",
                )

            if config.converter.type not in [
                "markitdown",
                "docling",
                "docling_vlm",
                "pymupdf",
            ]:
                raise ValidationError(
                    f"Unsupported converter type: {config.converter.type}"
                )

            # Check vision LLM requirement
            vision_required = config.converter.type in ["markitdown", "docling"]
            if vision_required and not config.converter.vision_llm:
                raise ValidationError(
                    f"Converter type '{config.converter.type}' requires vision_llm configuration"
                )

            return ValidationResult(
                test_name="Converter Configuration",
                success=True,
                message=f"Converter configuration is valid for type '{config.converter.type}'",
                details={
                    "converter_type": config.converter.type,
                    "requires_vision": vision_required,
                    "has_vision_llm": config.converter.vision_llm is not None,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Converter Configuration",
                success=False,
                message=f"Converter configuration test failed: {e}",
            )

    def _validate_extractor_config(self, config: "CrawlerConfig") -> ValidationResult:
        """Validate extractor configuration."""
        try:
            if not config.extractor:
                return ValidationResult(
                    test_name="Extractor Configuration",
                    success=True,
                    message="No extractor configuration provided (will use defaults)",
                )

            if config.extractor.type not in ["basic", "multi_schema"]:
                raise ValidationError(
                    f"Unsupported extractor type: {config.extractor.type}"
                )

            # Check LLM requirement
            if not config.extractor.llm:
                raise ValidationError(
                    f"Extractor type '{config.extractor.type}' requires LLM configuration"
                )

            return ValidationResult(
                test_name="Extractor Configuration",
                success=True,
                message=f"Extractor configuration is valid for type '{config.extractor.type}'",
                details={
                    "extractor_type": config.extractor.type,
                    "has_llm": config.extractor.llm is not None,
                    "metadata_schema_provided": config.extractor.metadata_schema
                    is not None,
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Extractor Configuration",
                success=False,
                message=f"Extractor configuration test failed: {e}",
            )

    def print_summary(self) -> None:
        """Print a summary of all validation results."""
        successful = sum(1 for r in self.results if r.success)
        total = len(self.results)

        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {successful}/{total}")

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.test_name}: {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"      {key}: {value}")

        if successful == total:
            print(f"\n🎉 All validation tests passed! Configuration is ready to use.")
        else:
            print(
                f"\n⚠️  {total - successful} validation test(s) failed. Please check the errors above."
            )

        print("=" * 60)
