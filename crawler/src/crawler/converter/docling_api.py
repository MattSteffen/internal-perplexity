"""
Docling API converter implementation.

This module provides a converter implementation that makes REST requests to a
Docling API instance for document conversion with VLM support.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
)
from pydantic import BaseModel, Field
from typing import Literal, Optional
from ..llm.llm import LLMConfig


class DoclingAPIConfig(BaseModel):
    """Configuration for Docling API converter."""

    type: Literal["docling_api"]
    base_url: str = Field(..., description="Base URL for the Docling API")
    vlm_config: Optional[LLMConfig] = Field(
        default=None, description="VLM configuration for image processing"
    )
    timeout: int = Field(default=600, description="Request timeout in seconds")
    do_picture_description: bool = Field(
        default=True, description="Enable image description"
    )
    image_export_mode: str = Field(default="embedded", description="Image export mode")
    include_images: bool = Field(default=True, description="Include images in output")
    abort_on_error: bool = Field(default=True, description="Abort conversion on error")


class DoclingAPIConverter(Converter):
    """Document converter using Docling API with VLM support."""

    def __init__(self, config: DoclingAPIConfig):
        """Initialize the Docling API converter."""
        super().__init__(config)
        self.session = requests.Session()

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "Docling API"

    def convert(
        self,
        doc: DocumentInput,
    ) -> ConvertedDocument:
        """Convert a document using Docling API."""

        convert_start_time = time.time()
        doc_name = doc.filename or "unknown"

        try:
            # Prepare the document data
            document_data = self._prepare_document_data(doc)

            # Prepare the conversion options
            conversion_options = self._prepare_conversion_options(doc)

            # Prepare the API request payload
            payload = {"options": conversion_options, "sources": [document_data]}

            # Make the API request
            api_url = f"{self.config.base_url}/v1/convert/source"
            headers = {"accept": "application/json", "Content-Type": "application/json"}

            response = self.session.post(
                api_url, json=payload, headers=headers, timeout=self.config.timeout
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            # Parse the response
            result = response.json()

            # Extract the markdown content
            markdown = self._extract_markdown_from_response(result)

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

    def _prepare_document_data(self, doc: DocumentInput) -> Dict[str, Any]:
        """Prepare document data for the API request."""
        if doc.source == "path":
            # Read file from path
            with open(doc.path, "rb") as f:
                file_data = f.read()
        elif doc.source == "bytes":
            file_data = doc.bytes_data
        elif doc.source == "fileobj":
            file_data = doc.fileobj.read()
        else:
            raise ValueError(f"Unsupported document source: {doc.source}")

        # Encode file data as base64
        file_base64 = base64.b64encode(file_data).decode("utf-8")

        # Determine the format
        if doc.mime_type:
            if doc.mime_type == "application/pdf":
                format_type = "pdf"
            elif (
                doc.mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                format_type = "docx"
            else:
                format_type = "pdf"  # Default to PDF
        elif doc.filename:
            ext = Path(doc.filename).suffix.lower()
            if ext == ".pdf":
                format_type = "pdf"
            elif ext == ".docx":
                format_type = "docx"
            else:
                format_type = "pdf"  # Default to PDF
        else:
            format_type = "pdf"  # Default to PDF

        return {
            "kind": "file",
            "format": format_type,
            "base64_string": file_base64,
            "filename": doc.filename or "document",
        }

    def _prepare_conversion_options(self, doc: DocumentInput) -> Dict[str, Any]:
        """Prepare conversion options for the API request."""
        config = self.config  # type: DoclingAPIConfig

        # Map from formats
        from_formats = []
        if doc.mime_type == "application/pdf" or (
            doc.filename and Path(doc.filename).suffix.lower() == ".pdf"
        ):
            from_formats.append("pdf")
        elif (
            doc.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or (doc.filename and Path(doc.filename).suffix.lower() == ".docx")
        ):
            from_formats.append("docx")
        else:
            from_formats.append("pdf")  # Default

        conversion_options = {
            "from_formats": from_formats,
            "to_formats": ["md"],
            "abort_on_error": config.abort_on_error,
            "do_picture_description": config.do_picture_description,
            "image_export_mode": config.image_export_mode,
            "include_images": config.include_images,
        }

        # Add VLM configuration if picture description is enabled
        if conversion_options["do_picture_description"] and config.vlm_config:
            conversion_options["picture_description_api"] = {
                "url": f"{config.vlm_config.base_url}/v1/chat/completions",
                "params": {"model": config.vlm_config.model_name},
                "timeout": config.vlm_config.default_timeout,
                "prompt": config.vlm_config.system_prompt
                or "Describe this image in detail for a technical document.",
            }

        return conversion_options

    def _extract_markdown_from_response(self, response: Dict[str, Any]) -> str:
        """Extract markdown content from the API response."""
        # The response structure may vary depending on the Docling API version
        # This is a best-effort extraction

        if "result" in response:
            result = response["result"]
            if isinstance(result, str):
                return result
            elif isinstance(result, dict) and "markdown" in result:
                return result["markdown"]
            elif isinstance(result, list) and len(result) > 0:
                # If result is a list, take the first item
                first_result = result[0]
                if isinstance(first_result, str):
                    return first_result
                elif isinstance(first_result, dict) and "markdown" in first_result:
                    return first_result["markdown"]

        # Fallback: look for any text content in the response
        if "content" in response:
            content = response["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, dict) and "markdown" in content:
                return content["markdown"]

        # If we can't find markdown, return the entire response as JSON
        return json.dumps(response, indent=2)

    def close(self) -> None:
        """Close the requests session."""
        self.session.close()
