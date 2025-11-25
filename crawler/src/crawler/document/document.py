"""
Document class that is imported in converter, chunker, extractor, and vector db.

This module provides a unified Document class that flows through the processing pipeline:
1. Converter: populates content, markdown, images, tables, stats, and warnings
2. Extractor: populates metadata and benchmark_questions from markdown
3. Chunker: populates chunks from markdown
4. Embeddings: populates text_embeddings and sparse embeddings for chunks
5. Vector DB: uses to_database_entities() to create entities for storage

Each processing stage modifies the Document in place, allowing for a clean
data flow through the entire pipeline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel, Field

# Import types from converter for stats
from ..converter.base import ConversionStats

# Type alias for bounding box
BBox = tuple[float, float, float, float]


class Document(BaseModel):
    """
    Unified document class for the entire processing pipeline.

    This class is mutable and designed to be modified by each processing stage:
    - Converter: Sets content, markdown, stats, warnings, source_name
    - Extractor: Sets metadata and benchmark_questions
    - Chunker: Sets chunks
    - Embedder: Sets text_embeddings, sparse_text_embeddings, sparse_metadata_embeddings
    - Vector DB: Uses to_database_entities() to create storage entities

    Attributes:
        document_id: Unique identifier for the document (UUID)
        source: Source identifier (file path, URL, etc.)
        content: Raw binary content (set by converter)
        markdown: Markdown representation (set by converter)
        source_name: Optional source name from converter
        stats: Conversion statistics (set by converter)
        warnings: Conversion warnings (set by converter)
        metadata: Extracted metadata (set by extractor)
        benchmark_questions: Generated questions (set by extractor)
        chunks: Text chunks (set by chunker)
        text_embeddings: Dense vector embeddings for each chunk (set by embedder)
        sparse_text_embeddings: Sparse embeddings for text chunks (set by embedder)
        sparse_metadata_embeddings: Sparse embeddings for metadata (set by embedder)
        security_group: Access control groups for RBAC
        minio_url: Optional URL to document in object storage
    """

    # Required fields - must be provided at creation
    document_id: str = Field(..., description="Unique identifier for the document (UUID)")
    source: str = Field(..., description="Source identifier (file path, URL, etc.)")

    # Fields populated by converter
    content: bytes | None = Field(default=None, description="Raw binary content (set by converter)")
    markdown: str | None = Field(default=None, description="Markdown representation (set by converter)")
    source_name: str | None = Field(default=None, description="Source name from converter (e.g., filename)")
    stats: ConversionStats = Field(default_factory=ConversionStats, description="Conversion statistics (set by converter)")
    warnings: list[str] = Field(default_factory=list, description="Conversion warnings (set by converter)")

    # Fields populated by extractor
    metadata: dict[str, Any] | None = Field(default=None, description="Extracted metadata (set by extractor)")
    benchmark_questions: list[str] | None = Field(default=None, description="Generated questions (set by extractor)")

    # Fields populated by chunker
    chunks: list[str] | None = Field(default=None, description="Text chunks (set by chunker)")

    # Fields populated by embedder
    text_embeddings: list[list[float]] | None = Field(default=None, description="Dense vector embeddings for each chunk (set by embedder)")
    sparse_text_embeddings: list[list[float]] | None = Field(default=None, description="Sparse embeddings for text chunks (set by embedder)")
    sparse_metadata_embeddings: list[float] | None = Field(default=None, description="Sparse embedding for metadata (set by embedder)")

    # Optional fields
    security_group: list[str] = Field(
        default_factory=lambda: ["public"],
        description="Access control groups for RBAC",
    )
    minio_url: str | None = Field(default=None, description="Optional URL to document in object storage")

    model_config = {
        "arbitrary_types_allowed": True,  # Allow bytes type
        "validate_assignment": True,  # Validate when fields are assigned
    }

    @classmethod
    def create(
        cls,
        source: str,
        document_id: str | None = None,
        security_group: list[str] | None = None,
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
        return self.is_converted() and self.is_extracted() and self.is_chunked() and self.text_embeddings is not None and len(self.text_embeddings) == len(self.chunks)

    def to_database_entities(self) -> list[dict[str, Any]]:
        """
        Convert document to database entities for insertion.

        Creates a list of entity dictionaries, one per chunk, with all fields
        required for database storage. Uses the default_ prefix for system fields
        to match DatabaseDocument schema.

        Returns:
            List of entity dictionaries ready for database insertion

        Raises:
            ValueError: If document is not ready for storage (missing chunks or embeddings)
        """
        if not self.is_ready_for_storage():
            raise ValueError("Document is not ready for storage. Must have chunks and embeddings.")

        if self.chunks is None or self.text_embeddings is None:
            raise ValueError("Document must have chunks and text_embeddings")

        if len(self.chunks) != len(self.text_embeddings):
            raise ValueError(f"Mismatch between chunks ({len(self.chunks)}) and embeddings ({len(self.text_embeddings)})")

        entities = []
        metadata_json = json.dumps(self.metadata or {})

        for i, (chunk, embedding) in enumerate(zip(self.chunks, self.text_embeddings)):
            entity: dict[str, Any] = {
                "default_document_id": self.document_id,
                "default_text": chunk,
                "default_text_embedding": embedding,
                "default_chunk_index": i,
                "default_source": self.source,
                "default_metadata": metadata_json,
                "default_minio": self.minio_url or "",
                "security_group": self.security_group,
                "metadata": self.metadata or {},
            }

            # Add sparse embeddings if available
            if self.sparse_text_embeddings and i < len(self.sparse_text_embeddings):
                entity["default_text_sparse_embedding"] = self.sparse_text_embeddings[i]

            if self.sparse_metadata_embeddings:
                entity["default_metadata_sparse_embedding"] = self.sparse_metadata_embeddings

            # Add benchmark questions if available (only to first chunk)
            if self.benchmark_questions and i == 0:
                entity["default_benchmark_questions"] = self.benchmark_questions

            entities.append(entity)

        return entities

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
        if self.text_embeddings:
            status_parts.append(f"embedded({len(self.text_embeddings)})")

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
            "source_name": self.source_name,
            "metadata": self.metadata,
            "benchmark_questions": self.benchmark_questions,
            "chunks": self.chunks,
            "security_group": self.security_group,
            "minio_url": self.minio_url,
            "warnings": self.warnings,
            # Note: images, tables, stats, and embeddings are not saved to JSON
            # as they can be large. They should be regenerated if needed.
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

        with open(filepath) as f:
            data = json.load(f)

        # Update document fields
        self.document_id = data.get("document_id", self.document_id)
        self.source = data.get("source", self.source)
        self.markdown = data.get("markdown")
        self.source_name = data.get("source_name")
        self.metadata = data.get("metadata")
        self.benchmark_questions = data.get("benchmark_questions")
        self.chunks = data.get("chunks")
        self.security_group = data.get("security_group", ["public"])
        self.minio_url = data.get("minio_url")
        self.warnings = data.get("warnings", [])

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

        with open(filepath) as f:
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
            source_name=data.get("source_name"),
            metadata=data.get("metadata"),
            benchmark_questions=data.get("benchmark_questions"),
            chunks=data.get("chunks"),
            security_group=data.get("security_group", ["public"]),
            minio_url=data.get("minio_url"),
            warnings=data.get("warnings", []),
        )
