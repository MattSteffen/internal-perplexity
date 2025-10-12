"""
Docling converter implementation.

This module provides a converter implementation using the Docling library
for advanced PDF processing with vision model integration.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

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


class DoclingConfig(BaseModel):
    """Configuration for Docling converter."""
    
    type: Literal["docling"]
    use_vlm: bool = Field(default=True, description="Whether to use VLM pipeline")
    vlm_base_url: str = Field(default="http://localhost:11434", description="Base URL for VLM API")
    vlm_model: str = Field(default="granite3.2-vision:latest", description="Vision model name")
    prompt: Optional[str] = Field(default=None, description="Custom prompt for vision processing")
    timeout_sec: float = Field(default=600.0, description="Timeout for API calls")
    scale: float = Field(default=1.0, description="Image scale factor")


# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


class DoclingConverter(Converter):
    """Document converter using the Docling library with optional VLM support."""

    DEFAULT_VISION_PROMPT = (
        "Analyze the page content. Convert all text to Markdown. "
        "For any technical diagrams, provide a detailed description of its components, "
        "connections, and overall structure within the Markdown output. Be brief and concise."
    )

    def __init__(self, config: DoclingConfig):
        """Initialize the Docling converter."""
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.doc_converter = self._create_converter()

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "Docling VLM" if self.config.use_vlm else "Docling"

    @property
    def capabilities(self) -> Capabilities:
        """Describe supported formats and features."""
        return Capabilities(
            name=self.name,
            supports_pdf=True,
            supports_docx=False,
            supports_images=True,
            supports_tables=True,
            requires_vision=self.config.use_vlm,
            supported_mime_types=[
                "application/pdf",
            ],
        )

    def supports(self, doc: DocumentInput) -> bool:
        """Return True if this converter can handle the given document."""
        # Docling primarily supports PDFs
        if doc.mime_type:
            return doc.mime_type == "application/pdf"
        if doc.filename:
            return Path(doc.filename).suffix.lower() == ".pdf"
        return False

    def convert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[callable] = None,
    ) -> ConvertedDocument:
        """Convert a document using Docling."""
        if options is None:
            options = ConvertOptions()

        if on_progress:
            on_progress(
                ProgressEvent(stage="start", message="Starting Docling conversion")
            )

        convert_start_time = time.time()
        doc_name = doc.filename or "unknown"

        try:
            self.logger.info(f"🔄 Starting Docling conversion: {doc_name}")
            self.logger.info("📄 Processing document with advanced vision model...")

            # Get file path for Docling
            if doc.source == "path":
                filepath = doc.path
            else:
                # For bytes or fileobj, we need to write to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    if doc.source == "bytes":
                        tmp.write(doc.bytes_data)
                    else:  # fileobj
                        tmp.write(doc.fileobj.read())
                    filepath = Path(tmp.name)

            # Perform conversion
            result = self.doc_converter.convert(filepath)

            # Clean up temporary file if created
            if doc.source != "path":
                filepath.unlink(missing_ok=True)

            # Export to markdown
            self.logger.info("📝 Exporting to markdown format...")
            markdown = result.document.export_to_markdown()

            total_time = time.time() - convert_start_time

            if on_progress:
                on_progress(
                    ProgressEvent(
                        stage="finish",
                        message="Docling conversion completed",
                        metrics={"total_time_sec": total_time, "output_length": len(markdown)}
                    )
                )

            self.logger.info("=== Docling conversion completed successfully ===")
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

        except Exception as e:
            self.logger.error(f"❌ Failed to convert '{doc_name}': {e}")
            raise

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter."""
        config = self.config  # type: DoclingConfig
        
        if config.use_vlm:
            # Create VLM options
            api_url = f"{config.vlm_base_url}/v1/chat/completions"
            
            vlm_options = ApiVlmOptions(
                url=api_url,
                params=dict(model=config.vlm_model),
                prompt=config.prompt or self.DEFAULT_VISION_PROMPT,
                timeout=config.timeout_sec,
                scale=config.scale,
                response_format=ResponseFormat.MARKDOWN,
            )

            pipeline_options = VlmPipelineOptions(enable_remote_services=True)
            pipeline_options.vlm_options = vlm_options

            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        pipeline_cls=VlmPipeline,
                    )
                }
            )
        else:
            # Use default pipeline without VLM
            return DocumentConverter()
