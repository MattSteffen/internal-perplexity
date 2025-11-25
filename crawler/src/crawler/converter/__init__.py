"""
Document conversion package using PyMuPDF4LLM.

This package provides a unified interface for document conversion using
PyMuPDF4LLM backend. It features type-safe configuration,
progress tracking, and rich result objects.
"""

# Core types and interfaces (no external dependencies)
from .base import (
    Converter,
    Capabilities,
    ConversionStats,
    ProgressEvent,
    ConverterConfig,
    create_converter,
)

# Factory
from .pymupdf4llm import PyMuPDF4LLMConfig

__all__ = [
    # Core interface
    "Converter",
    # Types
    "ProgressEvent",
    "Capabilities",
    "ConversionStats",
    # Configs
    "ConverterConfig",
    "PyMuPDF4LLMConfig",
    # Factory
    "create_converter",
]
