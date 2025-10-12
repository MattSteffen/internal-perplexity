"""
MarkItDown converter implementation.

This module provides a converter implementation using the MarkItDown library
for AI-powered document conversion with vision model support.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ProgressEvent,
    Capabilities,
)
from pydantic import BaseModel, Field
from typing import Literal, Optional


class MarkItDownConfig(BaseModel):
    """Configuration for MarkItDown converter."""
    
    type: Literal["markitdown"]
    llm_base_url: str = Field(..., description="Base URL for the LLM API")
    llm_model: str = Field(..., description="Model name to use for vision processing")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    enable_plugins: bool = Field(default=False, description="Enable MarkItDown plugins")


# MarkItDown imports
from markitdown import (
    MarkItDown,
    UnsupportedFormatException,
    FileConversionException,
    MissingDependencyException,
)


class MarkItDownConverter(Converter):
    """Document converter using the MarkItDown library."""

    def __init__(self, config: MarkItDownConfig):
        """Initialize the MarkItDown converter."""
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = self._create_client(config)

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "MarkItDown"

    @property
    def capabilities(self) -> Capabilities:
        """Describe supported formats and features."""
        return Capabilities(
            name=self.name,
            supports_pdf=True,
            supports_docx=True,
            supports_images=True,
            supports_tables=False,
            requires_vision=True,
            supported_mime_types=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain",
                "text/html",
            ],
        )

    def supports(self, doc: DocumentInput) -> bool:
        """Return True if this converter can handle the given document."""
        # MarkItDown can handle most document types
        return True

    def convert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[callable] = None,
    ) -> ConvertedDocument:
        """Convert a document using MarkItDown."""
        if options is None:
            options = ConvertOptions()

        if on_progress:
            on_progress(
                ProgressEvent(stage="start", message="Starting MarkItDown conversion")
            )

        convert_start_time = time.time()
        doc_name = doc.filename or "unknown"

        try:
            self.logger.info(f"🔄 Starting MarkItDown conversion: {doc_name}")
            self.logger.info("📄 Processing document with AI-powered conversion...")

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

            if on_progress:
                on_progress(
                    ProgressEvent(
                        stage="finish",
                        message="MarkItDown conversion completed",
                        metrics={"total_time_sec": total_time, "output_length": len(markdown)}
                    )
                )

            self.logger.info("=== MarkItDown conversion completed successfully ===")
            self.logger.info("📊 Conversion Statistics:")
            self.logger.info(f"   • Processing time: {total_time:.2f}s")
            self.logger.info(f"   • Output length: {len(markdown)} characters")
            self.logger.info(f"   • Average processing speed: {len(markdown)/total_time:.0f} chars/sec")

            return ConvertedDocument(
                source_name=doc.filename,
                markdown=markdown,
                stats={
                    "total_time_sec": total_time,
                    "output_length": len(markdown),
                }
            )

        except UnsupportedFormatException as e:
            self.logger.error(f"❌ Unsupported file format for '{doc_name}': {e}")
            raise
        except FileConversionException as e:
            self.logger.error(f"❌ Failed to convert '{doc_name}': {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during conversion of '{doc_name}': {e}")
            raise

    def _create_client(self, config: MarkItDownConfig) -> MarkItDown:
        """Create and configure the MarkItDown client."""
        # Adjust URL based on provider
        api_url = config.llm_base_url
        if not api_url.endswith("/v1"):
            api_url = f"{api_url}/v1"

        # Initialize the LLM client
        client = OpenAI(
            base_url=api_url,
            api_key=config.api_key or "ollama"
        )

        return MarkItDown(
            llm_client=client,
            llm_model=config.llm_model,
            enable_plugins=config.enable_plugins,
        )
