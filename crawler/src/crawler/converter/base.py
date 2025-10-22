"""
Base converter interface and abstract base class.

This module defines the core Converter interface that all converter implementations
must follow, providing a consistent API for document conversion operations.
"""

import abc

from .types import (
    DocumentInput,
    ConvertedDocument,
)


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
