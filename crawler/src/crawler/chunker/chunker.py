"""
Text chunking implementation.

This module provides a Chunker class for splitting text into chunks with
configurable chunk sizes and strategies.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(
        default=1000, gt=0, description="Maximum size of each chunk in characters"
    )
    overlap: int = Field(
        default=200,
        ge=0,
        description="Number of characters to overlap between consecutive chunks",
    )
    strategy: str = Field(
        default="naive", description="Chunking strategy to use (naive, semantic, etc.)"
    )
    preserve_paragraphs: bool = Field(
        default=True,
        description="Whether to try to preserve paragraph boundaries when chunking",
    )
    min_chunk_size: int = Field(
        default=100,
        gt=0,
        description="Minimum size of a chunk (chunks smaller than this will be merged with previous chunk)",
    )

    model_config = {"validate_assignment": True}

    @classmethod
    def create(
        cls,
        chunk_size: int = 1000,
        overlap: int = 200,
        strategy: str = "naive",
        preserve_paragraphs: bool = True,
        min_chunk_size: int = 100,
    ) -> "ChunkingConfig":
        """Create a ChunkingConfig with specified parameters."""
        return cls(
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
            preserve_paragraphs=preserve_paragraphs,
            min_chunk_size=min_chunk_size,
        )


class Chunker:
    """Text chunker with configurable chunk sizes and strategies."""

    def __init__(self, config: ChunkingConfig):
        """Initialize the chunker with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on the configured strategy.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        if self.config.strategy == "naive":
            return self._naive_chunk(text)
        else:
            self.logger.warning(
                f"Unknown chunking strategy: {self.config.strategy}, falling back to naive"
            )
            return self._naive_chunk(text)

    def _naive_chunk(self, text: str) -> List[str]:
        """
        Naive chunking strategy that splits text by character count.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.config.chunk_size

            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a good break point if preserving paragraphs
            if self.config.preserve_paragraphs:
                # Look for paragraph breaks (double newlines) within the chunk
                chunk_text = text[start:end]
                last_paragraph_break = chunk_text.rfind("\n\n")

                if last_paragraph_break > 0:
                    # Found a paragraph break, use it
                    end = start + last_paragraph_break
                else:
                    # Look for sentence endings within the chunk
                    last_sentence_end = chunk_text.rfind(". ")
                    if last_sentence_end > self.config.min_chunk_size:
                        end = start + last_sentence_end + 1
                    else:
                        # Look for word boundaries
                        last_space = chunk_text.rfind(" ")
                        if last_space > self.config.min_chunk_size:
                            end = start + last_space

            # Extract the chunk
            chunk = text[start:end].strip()

            # Only add chunk if it meets minimum size requirement
            if len(chunk) >= self.config.min_chunk_size or not chunks:
                chunks.append(chunk)
            elif chunks:
                # Merge small chunk with previous chunk
                chunks[-1] += " " + chunk

            # Move start position with overlap
            # Always advance at least 1 char to avoid infinite loops
            next_start = end - self.config.overlap
            if next_start <= start:
                next_start = end
            if next_start >= len(text) - self.config.min_chunk_size:
                remainder = text[next_start:].strip()
                if remainder:
                    chunks.append(remainder)
                break
            start = next_start

        return chunks

    def get_chunk_count(self, text: str) -> int:
        """
        Get the estimated number of chunks that would be created for the given text.

        Args:
            text: The text to estimate chunks for

        Returns:
            Estimated number of chunks
        """
        if not text or not text.strip():
            return 0

        if len(text) <= self.config.chunk_size:
            return 1

        # Rough estimation based on chunk size and overlap
        effective_chunk_size = self.config.chunk_size - self.config.overlap
        return max(1, (len(text) - self.config.overlap) // effective_chunk_size + 1)
