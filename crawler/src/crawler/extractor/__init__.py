"""
Document extraction module.

This module provides functionality for extracting structured metadata from documents
using LLM-based extraction with JSON Schema validation.
"""

from .extractor import (
    MetadataExtractor,
    MetadataExtractorConfig,
)

__all__ = [
    "MetadataExtractorConfig",
    "MetadataExtractor",
]
