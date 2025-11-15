"""
Base converter interface and abstract base class.

This module defines the core Converter interface that all converter implementations
must follow, providing a consistent API for document conversion operations.
"""

import abc
from typing import TYPE_CHECKING

from .types import (
    ConvertedDocument,
    DocumentInput,
)

if TYPE_CHECKING:
    from ..document import Document


class Converter(abc.ABC):
    """
    Abstract base class for document converters.

    Implementations must be stateless or externally re-entrant enough to allow
    multiple conversions in a row.
    """

    def __init__(self, config: any):
        """Initialize the converter with configuration."""
        self.config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert(
        self,
        doc: DocumentInput,
    ) -> ConvertedDocument:
        """Convert a single document (blocking)."""
        raise NotImplementedError

    def convert_document(self, document: "Document") -> None:
        """
        Convert a Document in place, populating all converter fields.

        This is the preferred method for the unified Document class.
        It modifies the document directly rather than returning a new object.

        Args:
            document: Document instance to convert (modified in place)
        """
        # Create DocumentInput from Document
        doc_input = DocumentInput.from_document(document)

        # Call the converter's convert method
        converted = self.convert(doc_input)

        # Populate document fields from ConvertedDocument
        document.markdown = converted.markdown
        document.source_name = converted.source_name
        document.images = converted.images
        document.tables = converted.tables
        document.stats = converted.stats
        document.warnings = converted.warnings

        # If document doesn't have content yet, try to get it from DocumentInput
        if document.content is None and doc_input.bytes_data:
            document.content = doc_input.bytes_data
