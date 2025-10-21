"""
Document class that is imported in converter, chunker, extractor, and vector db.

This module provides a unified Document class that flows through the processing pipeline:
1. Converter: populates content and markdown from raw input
2. Extractor: populates metadata and benchmark_questions from markdown
3. Chunker: populates chunks from markdown
4. Vector DB: converts to DatabaseDocument for storage

Each processing stage modifies the Document in place, allowing for a clean
data flow through the entire pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid
import json

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Unified document class for the processing pipeline.

    This class is mutable and designed to be modified by each processing stage:
    - Converter: Sets content and markdown
    - Extractor: Sets metadata and benchmark_questions
    - Chunker: Sets chunks
    - Vector DB: Reads all fields for storage

    Attributes:
        document_id: Unique identifier for the document (UUID)
        source: Source identifier (file path, URL, etc.)
        content: Raw binary content (set by converter)
        markdown: Markdown representation (set by converter)
        metadata: Extracted metadata (set by extractor)
        benchmark_questions: Generated questions (set by extractor)
        chunks: Text chunks (set by chunker)
        security_group: Access control groups for RBAC
        minio_url: Optional URL to document in object storage
    """

    # Required fields - must be provided at creation
    document_id: str = Field(
        ..., description="Unique identifier for the document (UUID)"
    )
    source: str = Field(..., description="Source identifier (file path, URL, etc.)")

    # Fields populated by converter
    content: Optional[bytes] = Field(
        default=None, description="Raw binary content (set by converter)"
    )
    markdown: Optional[str] = Field(
        default=None, description="Markdown representation (set by converter)"
    )

    # Fields populated by extractor
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Extracted metadata (set by extractor)"
    )
    benchmark_questions: Optional[List[str]] = Field(
        default=None, description="Generated questions (set by extractor)"
    )

    # Fields populated by chunker
    chunks: Optional[List[str]] = Field(
        default=None, description="Text chunks (set by chunker)"
    )

    # Optional fields
    security_group: List[str] = Field(
        default_factory=lambda: ["public"],
        description="Access control groups for RBAC",
    )
    minio_url: Optional[str] = Field(
        default=None, description="Optional URL to document in object storage"
    )

    model_config = {
        "arbitrary_types_allowed": True,  # Allow bytes type
        "validate_assignment": True,  # Validate when fields are assigned
    }

    @classmethod
    def create(
        cls,
        source: str,
        document_id: Optional[str] = None,
        security_group: Optional[List[str]] = None,
    ) -> Document:
        """
        Factory method to create a new Document with generated ID.

        Args:
            source: Source identifier (file path, URL, etc.)
            document_id: Optional document ID (generates UUID if not provided)
            security_group: Optional security groups (defaults to ["public"])

        Returns:
            New Document instance
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        return cls(
            document_id=document_id,
            source=source,
            security_group=security_group or ["public"],
        )

    def is_converted(self) -> bool:
        """Check if document has been converted (has markdown)."""
        return self.markdown is not None

    def is_extracted(self) -> bool:
        """Check if metadata has been extracted."""
        return self.metadata is not None

    def is_chunked(self) -> bool:
        """Check if document has been chunked."""
        return self.chunks is not None and len(self.chunks) > 0

    def is_ready_for_storage(self) -> bool:
        """Check if document is ready for vector DB storage."""
        return self.is_converted() and self.is_extracted() and self.is_chunked()

    def validate(self) -> None:
        """
        Validate document state.

        Raises:
            ValueError: If document is in an invalid state
        """
        if not self.document_id:
            raise ValueError("document_id is required")
        if not self.source:
            raise ValueError("source is required")

        if self.chunks and not self.markdown:
            raise ValueError("Cannot have chunks without markdown")
        if self.metadata and not self.markdown:
            raise ValueError("Cannot extract metadata without markdown")

    def __repr__(self) -> str:
        """Human-readable representation."""
        status_parts = []
        if self.is_converted():
            status_parts.append("converted")
        if self.is_extracted():
            status_parts.append("extracted")
        if self.is_chunked():
            status_parts.append(f"chunked({len(self.chunks)})")

        status = ", ".join(status_parts) if status_parts else "new"
        return f"Document(id={self.document_id[:8]}..., source={self.source}, status=[{status}])"

    def save(self, filepath: str) -> None:
        """
        Save document to a JSON file.

        Binary content is base64-encoded for JSON compatibility.

        Args:
            filepath: Path to save the document
        """
        import base64

        data = {
            "document_id": self.document_id,
            "source": self.source,
            "markdown": self.markdown,
            "metadata": self.metadata,
            "benchmark_questions": self.benchmark_questions,
            "chunks": self.chunks,
            "security_group": self.security_group,
            "minio_url": self.minio_url,
        }

        # Handle binary content by base64 encoding
        if self.content is not None:
            data["content"] = base64.b64encode(self.content).decode("utf-8")
        else:
            data["content"] = None

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load document from a JSON file.

        Updates the current document instance with data from the file.
        Binary content is base64-decoded from JSON.

        Args:
            filepath: Path to load the document from
        """
        import base64

        with open(filepath, "r") as f:
            data = json.load(f)

        # Update document fields
        self.document_id = data.get("document_id", self.document_id)
        self.source = data.get("source", self.source)
        self.markdown = data.get("markdown")
        self.metadata = data.get("metadata")
        self.benchmark_questions = data.get("benchmark_questions")
        self.chunks = data.get("chunks")
        self.security_group = data.get("security_group", ["public"])
        self.minio_url = data.get("minio_url")

        # Handle binary content by base64 decoding
        content_str = data.get("content")
        if content_str is not None:
            self.content = base64.b64decode(content_str)
        else:
            self.content = None

    @classmethod
    def from_file(cls, filepath: str) -> Document:
        """
        Load a document from a JSON file.

        Factory method that creates a new Document instance from a saved file.

        Args:
            filepath: Path to load the document from

        Returns:
            Document instance loaded from file
        """
        import base64

        with open(filepath, "r") as f:
            data = json.load(f)

        # Handle binary content by base64 decoding
        content_str = data.get("content")
        content = None
        if content_str is not None:
            content = base64.b64decode(content_str)

        return cls(
            document_id=data.get("document_id"),
            source=data.get("source"),
            content=content,
            markdown=data.get("markdown"),
            metadata=data.get("metadata"),
            benchmark_questions=data.get("benchmark_questions"),
            chunks=data.get("chunks"),
            security_group=data.get("security_group", ["public"]),
            minio_url=data.get("minio_url"),
        )
