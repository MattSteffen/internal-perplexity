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

__all__ = [
    # Core interface
    "Converter",
    # Configs
    "ConverterConfig",
    # Factory
    "create_converter",
]

