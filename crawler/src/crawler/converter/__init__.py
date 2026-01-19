"""
Document conversion package using PyMuPDF4LLM.

This package provides a unified interface for document conversion using
PyMuPDF4LLM backend. It features type-safe configuration,
progress tracking, and rich result objects.
"""

# Core types and interfaces (no external dependencies)
from .base import (
    Converter,
    create_converter,
)

# Factory
from .pymupdf4llm import ConverterConfig

# Alias for backward compatibility (used in examples and documentation)
PyMuPDF4LLMConfig = ConverterConfig

__all__ = [
    # Core interface
    "Converter",
    # Configs
    "ConverterConfig",
    "PyMuPDF4LLMConfig",
    # Factory
    "create_converter",
]

