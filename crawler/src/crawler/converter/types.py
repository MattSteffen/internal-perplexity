"""
Core types and data models for the converter package.

This module defines the input/output types, configuration options, and result
structures used throughout the converter system.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, List, Union, IO
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from ..document import Document


BBox = Tuple[float, float, float, float]


class DocumentInput(BaseModel):
    """Represents a document input from various sources."""

    source: str = Field(..., description="Source type: 'path', 'bytes', or 'fileobj'")
    path: Optional[Path] = None
    bytes_data: Optional[bytes] = None
    fileobj: Optional[IO[bytes]] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,  # Allow IO[bytes] type
    }

    @model_validator(mode="after")
    def _validate_source(self) -> "DocumentInput":
        """Validate that required fields are present based on source type."""
        if self.source == "path" and not self.path:
            raise ValueError("path required when source='path'")
        if self.source == "bytes" and self.bytes_data is None:
            raise ValueError("bytes_data required when source='bytes'")
        if self.source == "fileobj" and self.fileobj is None:
            raise ValueError("fileobj required when source='fileobj'")
        return self

    @classmethod
    def from_path(
        cls, p: Union[str, Path], mime_type: Optional[str] = None
    ) -> "DocumentInput":
        """Create DocumentInput from a file path."""
        p = Path(p)
        return cls(source="path", path=p, filename=p.name, mime_type=mime_type)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "DocumentInput":
        """Create DocumentInput from bytes data."""
        return cls(
            source="bytes",
            bytes_data=data,
            filename=filename,
            mime_type=mime_type,
        )

    @classmethod
    def from_fileobj(
        cls,
        f: IO[bytes],
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "DocumentInput":
        """Create DocumentInput from a file-like object."""
        return cls(source="fileobj", fileobj=f, filename=filename, mime_type=mime_type)

    @classmethod
    def from_document(cls, document: Document) -> "DocumentInput":
        """Create DocumentInput from a Document object."""
        # If document has content, use bytes source
        if document.content is not None:
            return cls.from_bytes(
                data=document.content,
                filename=(
                    document.source.split("/")[-1]
                    if "/" in document.source
                    else document.source
                ),
            )
        # Otherwise, assume source is a file path
        else:
            return cls.from_path(document.source)


class ConvertOptions(BaseModel):
    """Options for controlling document conversion behavior."""

    include_metadata: bool = True
    include_page_numbers: bool = True
    include_images: bool = True
    describe_images: bool = False
    image_prompt: Optional[str] = None
    extract_tables: bool = True
    table_strategy: str = "lines_strict"
    reading_order: bool = True
    page_range: Optional[Tuple[int, int]] = None  # inclusive 1-based range
    timeout_sec: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvertOptions":
        """Create ConvertOptions from a dictionary."""
        return cls(**data)


class ImageAsset(BaseModel):
    """Represents an extracted image from a document."""

    page_number: int = Field(..., ge=0)
    bbox: Optional[BBox] = None
    ext: str
    data: bytes
    description: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,  # Allow bytes type
    }


class TableAsset(BaseModel):
    """Represents an extracted table from a document."""

    page_number: int = Field(..., ge=0)
    bbox: Optional[BBox] = None
    rows: int = 0
    cols: int = 0
    markdown: str


class ConversionStats(BaseModel):
    """Statistics about a conversion operation."""

    total_pages: int = 0
    processed_pages: int = 0
    text_blocks: int = 0
    images: int = 0
    images_described: int = 0
    tables: int = 0
    total_time_sec: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ConvertedDocument(BaseModel):
    """Result of a document conversion operation."""

    source_name: Optional[str] = None
    markdown: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    images: List[ImageAsset] = Field(default_factory=list)
    tables: List[TableAsset] = Field(default_factory=list)
    stats: ConversionStats = Field(default_factory=ConversionStats)
    warnings: List[str] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    """Event emitted during conversion progress."""

    stage: str
    page: Optional[int] = None
    total_pages: Optional[int] = None
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class Capabilities(BaseModel):
    """Describes what a converter can handle."""

    name: str
    supports_pdf: bool = True
    supports_docx: bool = True
    supports_images: bool = True
    supports_tables: bool = True
    requires_vision: bool = False
    supported_mime_types: List[str] = Field(default_factory=list)
