"""
MarkItDown converter implementation.

This module provides a converter implementation using the MarkItDown library
for AI-powered document conversion with vision model support.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

# MarkItDown imports
from markitdown import (
    FileConversionException,
    MarkItDown,
    UnsupportedFormatException,
)
from openai import OpenAI
from pydantic import BaseModel, Field

from ..llm.llm import LLMConfig
from .base import Converter
from .types import (
    ConvertedDocument,
    DocumentInput,
)


class MarkItDownConfig(BaseModel):
    """Configuration for MarkItDown converter."""

    type: Literal["markitdown"]
    llm_config: LLMConfig = Field(..., description="LLM configuration for vision processing")
    enable_plugins: bool = Field(default=False, description="Enable MarkItDown plugins")


class MarkItDownConverter(Converter):
    """Document converter using the MarkItDown library."""

    def __init__(self, config: MarkItDownConfig):
        """Initialize the MarkItDown converter."""
        super().__init__(config)
        self._client = self._create_client(config)

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "MarkItDown"

    def convert(
        self,
        doc: DocumentInput,
    ) -> ConvertedDocument:
        """Convert a document using MarkItDown."""

        convert_start_time = time.time()

        try:
            # Get file path for MarkItDown
            if doc.source == "path":
                filepath = str(doc.path)
            else:
                # For bytes or fileobj, we need to write to a temporary file
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(doc.filename or "doc").suffix) as tmp:
                    if doc.source == "bytes":
                        tmp.write(doc.bytes_data)
                    else:  # fileobj
                        tmp.write(doc.fileobj.read())
                    filepath = tmp.name

            # Perform conversion
            result = self._client.convert(filepath)

            # Clean up temporary file if created
            if doc.source != "path":
                Path(filepath).unlink(missing_ok=True)

            # Build result
            markdown = result.markdown
            total_time = time.time() - convert_start_time

            return ConvertedDocument(
                source_name=doc.filename,
                markdown=markdown,
                stats={
                    "total_time_sec": total_time,
                    "output_length": len(markdown),
                },
            )

        except UnsupportedFormatException:
            raise
        except FileConversionException:
            raise
        except Exception:
            raise

    def _create_client(self, config: MarkItDownConfig) -> MarkItDown:
        """Create and configure the MarkItDown client."""
        # Adjust URL based on provider
        api_url = config.llm_config.base_url
        if not api_url.endswith("/v1"):
            api_url = f"{api_url}/v1"

        # Initialize the LLM client
        client = OpenAI(base_url=api_url, api_key=config.llm_config.api_key or "ollama")

        return MarkItDown(
            llm_client=client,
            llm_model=config.llm_config.model_name,
            enable_plugins=config.enable_plugins,
        )
