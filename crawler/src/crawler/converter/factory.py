"""
Factory for creating converter instances.

This module provides a factory function to create converter instances based on
configuration, using a type-safe discriminated union approach.
"""

from __future__ import annotations

from .base import Converter
from typing import Union
from pydantic import Field
from typing import Annotated

# Import configs from their respective files
from .markitdown import MarkItDownConfig
from .docling import DoclingConfig
from .pymupdf import PyMuPDFConfig
from .docling_api import DoclingAPIConfig

# Discriminated union for all converter configs
ConverterConfig = Annotated[
    Union[MarkItDownConfig, DoclingConfig, PyMuPDFConfig, DoclingAPIConfig],
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
    elif converter_type == "docling":
        from .docling import DoclingConverter
        return DoclingConverter(config)
    elif converter_type == "pymupdf":
        from .pymupdf import PyMuPDFConverter
        return PyMuPDFConverter(config)
    elif converter_type == "docling_api":
        from .docling_api import DoclingAPIConverter
        return DoclingAPIConverter(config)
    else:
        raise ValueError(
            f"Unsupported converter type: {config.type}. "
            f"Supported types: ['markitdown', 'docling', 'pymupdf', 'docling_api']"
        )
