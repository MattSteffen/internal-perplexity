"""
Base converter interface and abstract base class.

This module defines the core Converter interface that all converter implementations
must follow, providing a consistent API for document conversion operations.
"""

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..document import Document

from typing import Any

from pydantic import BaseModel, Field

from .pymupdf4llm import PyMuPDF4LLMConfig


class ConversionStats(BaseModel):
    """Statistics about a conversion operation."""

    total_pages: int = 0
    processed_pages: int = 0
    text_blocks: int = 0
    images: int = 0
    images_described: int = 0
    tables: int = 0
    total_time_sec: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProgressEvent(BaseModel):
    """Event emitted during conversion progress."""

    stage: str
    page: int | None = None
    total_pages: int | None = None
    message: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class Capabilities(BaseModel):
    """Describes what a converter can handle."""

    name: str
    supports_pdf: bool = True
    supports_docx: bool = True
    supports_images: bool = True
    supports_tables: bool = True
    requires_vision: bool = False
    supported_mime_types: list[str] = Field(default_factory=list)


class Converter(abc.ABC):
    """
    Abstract base class for document converters.

    Implementations must be stateless or externally re-entrant enough to allow
    multiple conversions in a row.
    """

    def __init__(self, config: Any):
        """Initialize the converter with configuration."""
        self.config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert(self, document: "Document") -> None:
        """
        Convert a Document in place, populating converter fields.

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


# ConverterConfig is just an alias for PyMuPDF4LLMConfig since we only support one converter
ConverterConfig = PyMuPDF4LLMConfig


def create_converter(config: ConverterConfig) -> Converter:
    """
    Create a converter instance based on configuration.

    Args:
        config: PyMuPDF4LLMConfig object specifying the converter parameters

    Returns:
        PyMuPDF4LLMConverter instance
    """
    from .pymupdf4llm import PyMuPDF4LLMConverter

    return PyMuPDF4LLMConverter(config)
