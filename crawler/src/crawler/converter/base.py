"""
Base converter interface and abstract base class.

This module defines the core Converter interface that all converter implementations
must follow, providing a consistent API for document conversion operations.
"""

from __future__ import annotations

import abc
from typing import Iterable, List, Optional, Callable

from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ProgressEvent,
    Capabilities,
)


ProgressCallback = Callable[[ProgressEvent], None]


class Converter(abc.ABC):
    """
    Abstract base class for document converters.

    Implementations must be stateless or externally re-entrant enough to allow
    multiple conversions in a row. Any network handles should be managed and
    closed in close().
    """

    def __init__(self, config: any):
        """Initialize the converter with configuration."""
        self.config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """Describe supported formats and features."""
        raise NotImplementedError

    @abc.abstractmethod
    def supports(self, doc: DocumentInput) -> bool:
        """Return True if this converter can handle the given document."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ConvertedDocument:
        """Convert a single document (blocking)."""
        raise NotImplementedError

    async def aconvert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ConvertedDocument:
        """Async version of convert. Default impl runs in a thread."""
        # Implementations may override for true async I/O
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.convert(doc, options, on_progress)
        )

    def convert_many(
        self,
        docs: Iterable[DocumentInput],
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[ConvertedDocument]:
        """Convert multiple documents sequentially. Override for batching."""
        return [self.convert(d, options, on_progress) for d in docs]

    async def aconvert_many(
        self,
        docs: Iterable[DocumentInput],
        options: Optional[ConvertOptions] = None,
        concurrency: int = 4,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[ConvertedDocument]:
        """Async conversion of multiple documents with bounded concurrency."""
        import asyncio
        sem = asyncio.Semaphore(concurrency)

        async def run(d: DocumentInput) -> ConvertedDocument:
            async with sem:
                return await self.aconvert(d, options, on_progress)

        return await asyncio.gather(*(run(d) for d in docs))

    def close(self) -> None:
        """Release any resources (network pools, temp dirs). Optional."""
        return
