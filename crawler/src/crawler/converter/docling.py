"""
Docling converter implementation.

This module provides a converter implementation using the Docling library
for advanced PDF processing with vision model integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
)
from pydantic import BaseModel, Field
from typing import Literal, Optional
from ..llm.llm import LLMConfig


class DoclingConfig(BaseModel):
    """Configuration for Docling converter."""

    type: Literal["docling"]
    use_vlm: bool = Field(default=True, description="Whether to use VLM pipeline")
    vlm_config: Optional[LLMConfig] = Field(
        default=None, description="VLM configuration for vision processing"
    )
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
        self.doc_converter = self._create_converter()

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "Docling VLM" if self.config.use_vlm else "Docling"

    def convert(
        self,
        doc: DocumentInput,
    ) -> ConvertedDocument:
        """Convert a document using Docling."""

        convert_start_time = time.time()
        doc_name = doc.filename or "unknown"

        try:
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
            markdown = result.document.export_to_markdown()

            total_time = time.time() - convert_start_time

            return ConvertedDocument(
                source_name=doc.filename,
                markdown=markdown,
                stats={
                    "total_time_sec": total_time,
                    "output_length": len(markdown),
                },
            )

        except Exception as e:
            raise

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter."""
        config = self.config  # type: DoclingConfig

        if config.use_vlm and config.vlm_config:
            # Create VLM options using LLMConfig
            api_url = f"{config.vlm_config.base_url}/v1/chat/completions"

            vlm_options = ApiVlmOptions(
                url=api_url,
                params=dict(model=config.vlm_config.model_name),
                prompt=config.vlm_config.system_prompt or self.DEFAULT_VISION_PROMPT,
                timeout=config.vlm_config.default_timeout,
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
