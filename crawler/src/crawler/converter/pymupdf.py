"""
PyMuPDF converter implementation.

This module provides a comprehensive converter implementation using PyMuPDF
for PDF processing with image extraction, table detection, and AI-powered descriptions.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, IO
from abc import ABC, abstractmethod

import pymupdf
from pydantic import BaseModel, Field

from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    Capabilities,
    ImageAsset,
    TableAsset,
    ConversionStats,
    BBox,
)
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from ..llm.llm import LLMConfig


class PyMuPDFConfig(BaseModel):
    """Configuration for PyMuPDF converter."""

    type: Literal["pymupdf"]
    vlm_config: Optional[LLMConfig] = Field(
        default=None,
        description="VLM configuration",
    )
    convert_options: Optional[ConvertOptions] = Field(
        default=None,
        description="Conversion options",
    )


class VLMInterface(ABC):
    """Abstract interface for image description services."""

    @abstractmethod
    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: str = None
    ) -> str:
        """
        Describe an image given its binary data and extension.

        Args:
            image_data: Binary image data
            image_ext: Image file extension (e.g., 'png', 'jpg')
            prompt: Custom prompt for image description

        Returns:
            Description of the image as a string
        """
        pass


class OllamaVLM(VLMInterface):
    """Implementation for Ollama API VLM."""

    def __init__(
        self, model_name: str = "llava", base_url: str = "http://localhost:11434"
    ):
        """Initialize Ollama VLM."""
        # Validate inputs early to fail fast
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("OllamaVLM requires a non-empty model_name")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OllamaVLM requires a non-empty base_url")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        # Try to import requests
        try:
            import requests

            self.requests = requests
        except ImportError:
            raise ImportError(
                "requests library not found. Install with: pip install requests"
            )

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: str = None
    ) -> str:
        """Describe image using Ollama API."""
        try:
            # Convert image data to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Use provided prompt or default
            if prompt is None:
                prompt = "Describe this image in detail. Focus on the main content, objects, text, and any relevant information that would be useful in a document context."

            # Prepare the API request
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}

            # Make the API call
            response = self.requests.post(
                url, json=payload, headers=headers, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()

                if description:
                    return description
                else:
                    return (
                        f"[No description returned from Ollama for {image_ext} image]"
                    )
            else:
                return f"[Error getting description from Ollama: HTTP {response.status_code}]"

        except Exception as e:
            return f"[Error describing image: {str(e)}]"


class DummyVLM(VLMInterface):
    """Dummy implementation for testing."""

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: str = None
    ) -> str:
        return f"[Dummy description for {image_ext} image of {len(image_data)} bytes]"


class PyMuPDFConverter(Converter):
    """Document converter using PyMuPDF with comprehensive content extraction."""

    DEFAULT_IMAGE_PROMPT = "Describe this image in detail. Focus on the main content, objects, text, and any relevant information that would be useful in a document context."

    def __init__(self, config: PyMuPDFConfig):
        """Initialize the PyMuPDF converter."""
        super().__init__(config)
        self.config = config

        # Initialize image describer
        self.vlm = self._create_vlm()

    @property
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        return "PyMuPDF"

    @property
    def capabilities(self) -> Capabilities:
        """Describe supported formats and features."""
        return Capabilities(
            name=self.name,
            supports_pdf=True,
            supports_docx=False,
            supports_images=True,
            supports_tables=True,
            requires_vision=False,  # Optional
            supported_mime_types=[
                "application/pdf",
            ],
        )

    def supports(self, doc: DocumentInput) -> bool:
        """Return True if this converter can handle the given document."""
        # PyMuPDF primarily supports PDFs
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
        """Convert a PDF file to markdown with comprehensive content extraction."""
        options = self.config.convert_options

        convert_start_time = time.time()
        doc_name = doc.filename or "unknown"

        try:
            # Get file path for PyMuPDF
            if doc.source == "path":
                filepath = str(doc.path)
            else:
                # For bytes or fileobj, we need to write to a temporary file
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    if doc.source == "bytes":
                        tmp.write(doc.bytes_data)
                    else:  # fileobj
                        tmp.write(doc.fileobj.read())
                    filepath = tmp.name

            with pymupdf.open(filepath) as pdf_doc:
                total_pages = len(pdf_doc)

                markdown_content = []
                stats = ConversionStats(
                    total_pages=total_pages,
                    processed_pages=0,
                    text_blocks=0,
                    images=0,
                    images_described=0,
                    tables=0,
                    total_time_sec=None,
                )

                # Add document header
                markdown_content.append(f"# Document: {doc_name}\n\n")

                # Add metadata if requested
                if options.include_metadata:
                    metadata = pdf_doc.metadata
                    if metadata and any(metadata.values()):
                        markdown_content.append("## Document Metadata\n\n")
                        for key, value in metadata.items():
                            if value:
                                markdown_content.append(f"- **{key}**: {value}\n")
                        markdown_content.append("\n")

                # Process pages
                for page_num in range(total_pages):
                    page_start_time = time.time()
                    page = pdf_doc[page_num]

                    if options.include_page_numbers:
                        markdown_content.append(f"## Page {page_num + 1}\n\n")

                    # Extract all content types
                    text_blocks = self._get_text_blocks_with_positions(page, options)
                    stats.text_blocks += len(text_blocks)

                    images = self._extract_images_from_page(page, options)
                    stats.images += len(images)

                    tables = self._extract_tables_from_page(page, options)
                    stats.tables += len(tables)

                    # Describe images with progress tracking
                    if images and options.describe_images:
                        for i, image in enumerate(images):
                            try:
                                description = self.vlm.describe_image(
                                    image.data,
                                    image.ext,
                                    options.image_prompt or self.DEFAULT_IMAGE_PROMPT,
                                )
                                image.description = description
                                stats.images_described += 1
                            except Exception as e:
                                image.description = (
                                    f"[Error describing image: {str(e)}]"
                                )

                    # Merge and sort all content
                    merged_content = self._merge_content_by_position(
                        text_blocks, images, tables
                    )

                    # Generate markdown for this page
                    page_markdown = []
                    for item in merged_content:
                        if item["type"] == "text":
                            page_markdown.append(item["text"])
                            page_markdown.append("\n\n")
                        elif item["type"] == "image":
                            image_desc = (
                                item["image"].description
                                or "[No description available]"
                            )
                            page_markdown.append(
                                f"![Image Description]({item['image'].ext})\n*{image_desc}*\n\n"
                            )
                        elif item["type"] == "table":
                            page_markdown.append(
                                f"**Table {item['table_index'] + 1}** ({item['rows']} rows × {item['cols']} cols)\n\n"
                            )
                            page_markdown.append(item["markdown"])
                            page_markdown.append("\n\n")

                    markdown_content.extend(page_markdown)
                    markdown_content.append("\n")  # Add space between pages

                    stats.processed_pages += 1

                # Clean up temporary file if created
                if doc.source != "path":
                    Path(filepath).unlink(missing_ok=True)

                # Join all content
                result_markdown = "".join(markdown_content)

                # Log final statistics
                total_time = time.time() - convert_start_time
                stats.total_time_sec = total_time

                return ConvertedDocument(
                    source_name=doc.filename,
                    markdown=result_markdown,
                    images=images,
                    tables=tables,
                    stats=stats,
                )

        except Exception as e:
            raise

    def _create_vlm(self) -> VLMInterface:
        """Create and configure the VLM based on configuration."""
        config = self.config.vlm_config

        if config.provider == "ollama":
            return OllamaVLM(model_name=config.model_name, base_url=config.base_url)

        if config.provider == "dummy":
            return DummyVLM()

        raise ValueError(f"Unsupported vlm provider: {config.provider}")

    def _extract_tables_from_page(
        self, page: pymupdf.Page, options: ConvertOptions
    ) -> List[TableAsset]:
        """Extract tables from a page using PyMuPDF's table detection."""
        tables = []

        if not options.extract_tables:
            return tables

        try:
            # Find tables on the page
            page_tables = page.find_tables(strategy=options.table_strategy)

            for table_index, table in enumerate(page_tables):
                try:
                    # Extract table data
                    table_data = table.extract()

                    # Convert to markdown table format
                    markdown_table = self._convert_table_to_markdown(table_data)

                    tables.append(
                        TableAsset(
                            page_number=page.number,
                            bbox=table.bbox,
                            rows=len(table_data),
                            cols=len(table_data[0]) if table_data else 0,
                            markdown=markdown_table,
                        )
                    )

                except Exception:
                    continue

        except Exception:
            pass

        return tables

    def _convert_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table data to markdown format."""
        if not table_data:
            return ""

        markdown_lines = []

        # Add header row
        if table_data:
            header_row = (
                "| " + " | ".join(str(cell).strip() for cell in table_data[0]) + " |"
            )
            markdown_lines.append(header_row)

            # Add separator row
            separator = "| " + " | ".join("---" for _ in table_data[0]) + " |"
            markdown_lines.append(separator)

            # Add data rows
            for row in table_data[1:]:
                data_row = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
                markdown_lines.append(data_row)

        return "\n".join(markdown_lines)

    def _get_text_blocks_with_positions(
        self, page: pymupdf.Page, options: ConvertOptions
    ) -> List[Dict[str, Any]]:
        """Get text blocks with their positions on the page, excluding table areas."""
        # Get text as blocks to maintain position information
        text_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        blocks = text_dict["blocks"]

        # Get table areas to exclude from text extraction
        table_areas = []
        if options.extract_tables:
            try:
                page_tables = page.find_tables(strategy=options.table_strategy)
                table_areas = [table.bbox for table in page_tables]
            except Exception:
                pass

        text_blocks = []
        for block in blocks:
            if "lines" in block:  # Text block
                block_bbox = block["bbox"]

                # Check if block overlaps significantly with any table
                is_in_table = False
                for table_bbox in table_areas:
                    if self._bbox_overlap_ratio(block_bbox, table_bbox) > 0.7:
                        is_in_table = True
                        break

                # Skip blocks that are part of tables
                if is_in_table:
                    continue

                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    if line_text.strip():  # Only add non-empty lines
                        block_text += line_text.strip() + " "

                if block_text.strip():
                    text_blocks.append(
                        {
                            "type": "text",
                            "bbox": block["bbox"],
                            "text": block_text.strip(),
                            "block_no": block["number"],
                        }
                    )

        return text_blocks

    def _bbox_overlap_ratio(self, bbox1, bbox2):
        """Calculate the overlap ratio between two bounding boxes."""
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        return intersection / area1 if area1 > 0 else 0.0

    def _extract_images_from_page(
        self, page: pymupdf.Page, options: ConvertOptions
    ) -> List[ImageAsset]:
        """Extract all images from a page."""
        extracted_images = []

        if not options.include_images:
            return extracted_images

        # Get list of images on the page
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            # Get image reference
            xref = img[0]

            try:
                # Extract image data
                base_image = page.parent.extract_image(xref)
                image_data = base_image["image"]
                image_ext = base_image["ext"]

                # Get image rectangle (position on page)
                image_rects = page.get_image_rects(xref)
                bbox = image_rects[0] if image_rects else None

                extracted_image = ImageAsset(
                    page_number=page.number,
                    bbox=bbox,
                    data=image_data,
                    ext=image_ext,
                )

                extracted_images.append(extracted_image)

            except Exception:
                continue

        return extracted_images

    def _merge_content_by_position(
        self,
        text_blocks: List[Dict[str, Any]],
        images: List[ImageAsset],
        tables: List[TableAsset],
    ) -> List[Dict[str, Any]]:
        """Merge text blocks, images, and tables based on their positions on the page."""
        all_content = []

        # Add text blocks
        for block in text_blocks:
            all_content.append(block)

        # Add tables
        for table_index, table in enumerate(tables):
            all_content.append(
                {
                    "type": "table",
                    "bbox": table.bbox,
                    "table": table,
                    "table_index": table_index,
                    "rows": table.rows,
                    "cols": table.cols,
                    "markdown": table.markdown,
                    "block_no": 999 + table_index,
                }
            )

        # Add images
        for image_index, image in enumerate(images):
            all_content.append(
                {
                    "type": "image",
                    "bbox": image.bbox,
                    "image": image,
                    "block_no": 1999 + image_index,
                }
            )

        # Sort by vertical position (top to bottom), then horizontal (left to right)
        all_content.sort(
            key=lambda x: (
                x["bbox"][1] if x["bbox"] else 0,
                x["bbox"][0] if x["bbox"] else 0,
            )
        )

        return all_content
