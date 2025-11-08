"""
Factory for creating converter instances.

This module provides a factory function to create converter instances based on
configuration, using a type-safe discriminated union approach.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import Converter

# Import configs from their respective files
from .markitdown import MarkItDownConfig
from .pymupdf4llm import PyMuPDF4LLMConfig

# Discriminated union for all converter configs
ConverterConfig = Annotated[
    MarkItDownConfig | PyMuPDF4LLMConfig,
    Field(discriminator="type"),
]


def create_converter(config: ConverterConfig) -> Converter:
    """
    Create a converter instance based on configuration.

    Args:
        config: ConverterConfig object specifying the converter type and parameters

    Returns:
        Converter instance of the appropriate type

    Raises:
        ValueError: If the converter type is not supported
    """
    converter_type = config.type.lower()

    # Lazy imports to avoid dependency issues
    if converter_type == "markitdown":
        from .markitdown import MarkItDownConverter

        return MarkItDownConverter(config)
    elif converter_type == "pymupdf4llm":
        from .pymupdf4llm import PyMuPDF4LLMConverter

        return PyMuPDF4LLMConverter(config)
    else:
        raise ValueError(f"Unsupported converter type: {config.type}. " f"Supported types: ['markitdown', 'pymupdf4llm']")
