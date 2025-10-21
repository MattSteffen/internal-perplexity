"""
Document conversion package with support for multiple conversion backends.

This package provides a unified interface for document conversion with support for
MarkItDown, Docling, PyMuPDF, and Docling API backends. It features type-safe configuration,
progress tracking, and rich result objects.
"""

# Core types and interfaces (no external dependencies)
from .base import Converter, ProgressCallback
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ProgressEvent,
    Capabilities,
    ImageAsset,
    TableAsset,
    ConversionStats,
)
from .factory import ConverterConfig
from .markitdown import MarkItDownConfig
from .docling import DoclingConfig
from .pymupdf import PyMuPDFConfig
from .docling_api import DoclingAPIConfig

# Factory (lazy imports for converters)
from .factory import create_converter

__all__ = [
    # Core interface
    "Converter",
    "ProgressCallback",
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
    "DoclingConfig",
    "PyMuPDFConfig",
    "DoclingAPIConfig",
    # Factory
    "create_converter",
]
