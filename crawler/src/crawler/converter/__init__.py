"""
Document conversion package using PyMuPDF4LLM.

This package provides a unified interface for document conversion using
PyMuPDF4LLM backend. It features type-safe configuration,
progress tracking, and rich result objects.
"""

# Core types and interfaces (no external dependencies)
from .base import (
    Capabilities,
    ConversionStats,
    Converter,
    ProgressEvent,
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


def create_converter(config: "ConverterConfig") -> "Converter":
    """
    Create a converter instance based on configuration.

    Args:
        config: PyMuPDF4LLMConfig object specifying the converter parameters

    Returns:
        PyMuPDF4LLMConverter instance
    """
    from .pymupdf4llm import PyMuPDF4LLMConverter

    return PyMuPDF4LLMConverter(config)


# ConverterConfig is just an alias for PyMuPDF4LLMConfig
ConverterConfig = PyMuPDF4LLMConfig
