"""
Base converter interface and abstract base class.

This module defines the core Converter interface that all converter implementations
must follow, providing a consistent API for document conversion operations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..document import Document
    from .pymupdf4llm import ConverterConfig


def create_converter(config: "ConverterConfig") -> "Converter":
    """
    Create a converter instance based on configuration.

    Args:
        config: ConverterConfig object specifying the converter parameters

    Returns:
        Converter instance
    """
    # Import here to avoid circular dependencies
    from .pymupdf4llm import PyMuPDF4LLMConverter
    
    if config.type == "pymupdf4llm":
        return PyMuPDF4LLMConverter(config)
    
    raise ValueError(f"Unknown converter type: {config.type}")



class Converter(ABC):
    """
    Abstract base class for document converters.

    Implementations must be stateless or externally re-entrant enough to allow
    multiple conversions in a row.
    """

    def __init__(self, config: Any):
        """Initialize the converter with configuration."""
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, document: "Document") -> str:
        """
        Convert a Document in markdown, populating converter fields.

        This method modifies the document directly, populating:
        - content: Raw binary content (if not already set)
        - markdown: Converted markdown text
        - stats: Conversion statistics
        - source_name: Source filename (if not already set)
        - warnings: List of warning messages

        Args:
            document: Document instance to convert (modified in place)
        """
        raise NotImplementedError
