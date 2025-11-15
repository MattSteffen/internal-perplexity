"""
Document conversion package with support for multiple conversion backends.

This package provides a unified interface for document conversion with support for
MarkItDown and PyMuPDF backends. It features type-safe configuration,
progress tracking, and rich result objects.
"""

# Core types and interfaces (no external dependencies)
from .base import Converter

# Factory (lazy imports for converters)
from .factory import ConverterConfig, create_converter
from .markitdown import MarkItDownConfig
from .pymupdf4llm import PyMuPDF4LLMConfig
from .types import (
    Capabilities,
    ConversionStats,
    ConvertedDocument,
    ConvertOptions,
    DocumentInput,
    ImageAsset,
    ProgressEvent,
    TableAsset,
)

__all__ = [
    # Core interface
    "Converter",
    # Types
    "DocumentInput",
    "ConvertOptions",
    "ConvertedDocument",
    "ProgressEvent",
    "Capabilities",
    "ImageAsset",
    "TableAsset",
    "ConversionStats",
    # Configs
    "ConverterConfig",
    "MarkItDownConfig",
    "PyMuPDFConfig",
    "PyMuPDF4LLMConfig",
    # Factory
    "create_converter",
]
