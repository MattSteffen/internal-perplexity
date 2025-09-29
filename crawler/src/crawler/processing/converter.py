"""
Document conversion utilities with support for multiple conversion backends.

This module provides a base converter class and implementations for different
document conversion libraries including MarkItDown and Docling.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
import pathlib

import pymupdf
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Third-party imports
from openai import OpenAI

# MarkItDown imports
from markitdown import (
    MarkItDown,
    UnsupportedFormatException,
    FileConversionException,
    MissingDependencyException,
)

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Local imports
from .llm import LLMConfig


@dataclass
class ConverterConfig:
    """Configuration for document converters."""

    type: str = "markitdown"
    vision_llm: Optional[LLMConfig] = None
    metadata: Optional[Dict[str, Any]] = (
        None  # Converter-specific metadata configuration
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.type:
            raise ValueError("Converter type cannot be empty")

        # Validate that vision_llm is provided for converters that need it
        vision_requiring_types = ["markitdown", "docling"]
        if self.type in vision_requiring_types and self.vision_llm is None:
            raise ValueError(
                f"Converter type '{self.type}' requires vision_llm configuration"
            )

    @classmethod
    def markitdown(
        cls, vision_llm: LLMConfig, metadata: Optional[Dict[str, Any]] = None
    ) -> "ConverterConfig":
        """Create MarkItDown converter configuration."""
        return cls(type="markitdown", vision_llm=vision_llm, metadata=metadata)

    @classmethod
    def docling(
        cls, vision_llm: LLMConfig, metadata: Optional[Dict[str, Any]] = None
    ) -> "ConverterConfig":
        """Create Docling converter configuration."""
        return cls(type="docling", vision_llm=vision_llm, metadata=metadata)

    @classmethod
    def docling_vlm(cls) -> "ConverterConfig":
        """Create Docling VLM converter configuration (uses default VLM)."""
        return cls(
            type="docling_vlm",
            vision_llm=None,  # Docling VLM uses its own default
            metadata=None,
        )

    @classmethod
    def pymupdf(
        cls,
        vision_llm: Optional[LLMConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ConverterConfig":
        """Create PyMuPDF converter configuration."""
        return cls(type="pymupdf", vision_llm=vision_llm, metadata=metadata)


class Converter(ABC):
    """
    Abstract base class for document converters.

    This class defines the interface that all document converters must implement.
    """

    def __init__(self, config: ConverterConfig):
        """
        Initialize the converter with configuration.

        Args:
            config: ConverterConfig object containing configuration options
        """
        self.config = config
        # Use metadata field for converter-specific configuration
        self.metadata_config = config.metadata or {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Get logger for the converter (already configured by main crawler)."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_file_exists(self, filepath: str) -> None:
        """
        Validate that the input file exists.

        Args:
            filepath: Path to the file to validate

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found at {filepath}")

    @abstractmethod
    def convert(self, filepath: str) -> str:
        """
        Convert the given file to markdown.

        Args:
            filepath: Path to the file to be converted.

        Returns:
            Markdown string representation of the document.
        """
        pass


class MarkItDownConverter(Converter):
    """
    Document converter using the MarkItDown library.

    This converter supports various document formats and can use vision models
    for processing images and complex layouts.
    """

    def __init__(self, config: ConverterConfig):
        """
        Initialize the MarkItDown converter.

        Args:
            config: ConverterConfig object with vision_llm and metadata configuration
        """
        super().__init__(config)
        self.markitdown = self._create_converter()

    def _create_converter(self) -> MarkItDown:
        """Create and configure the MarkItDown converter."""
        vision_config = self.config.vision_llm
        if not vision_config:
            raise ValueError(
                "vision_llm configuration is required for MarkItDown converter"
            )

        # Adjust URL based on provider
        api_url = (
            f"{vision_config.base_url}/v1"
            if vision_config.provider == "ollama"
            else vision_config.base_url
        )

        # Initialize the LLM client
        client = OpenAI(base_url=api_url, api_key="ollama")

        return MarkItDown(
            llm_client=client,
            llm_model=vision_config.model_name,
            enable_plugins=False,
        )

    def convert(self, filepath: str) -> str:
        """
        Convert a document to markdown using MarkItDown with progress tracking.

        Args:
            filepath: Path to the document to convert

        Returns:
            Markdown string representation of the document

        Raises:
            FileNotFoundError: If the input file doesn't exist
            UnsupportedFormatException: If the file format is not supported
            FileConversionException: If conversion fails
        """
        self._validate_file_exists(filepath)

        convert_start_time = time.time()
        doc_name = pathlib.Path(filepath).name

        try:
            self.logger.info(f"🔄 Starting MarkItDown conversion: {doc_name}")
            self.logger.info("📄 Processing document with AI-powered conversion...")

            result = self.markitdown.convert(filepath)

            # Log conversion statistics
            total_time = time.time() - convert_start_time
            markdown_length = len(result.markdown)

            self.logger.info("=== MarkItDown conversion completed successfully ===")
            self.logger.info("📊 Conversion Statistics:")
            self.logger.info(f"   • Processing time: {total_time:.2f}s")
            self.logger.info(f"   • Output length: {markdown_length} characters")
            self.logger.info(
                f"   • Average processing speed: {markdown_length/total_time:.0f} chars/sec"
            )

            return result.markdown

        except UnsupportedFormatException as e:
            self.logger.error(f"❌ Unsupported file format for '{doc_name}': {e}")
            raise
        except FileConversionException as e:
            self.logger.error(f"❌ Failed to convert '{doc_name}': {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error during conversion of '{doc_name}': {e}"
            )
            raise


class DoclingConverter(Converter):
    """
    Document converter using the Docling library.

    This converter specializes in PDF processing with advanced vision model
    integration for handling complex layouts and visual elements.
    """

    DEFAULT_VISION_PROMPT = "Analyze the page content. Convert all text to Markdown. For any technical diagrams, provide a detailed description of its components, connections, and overall structure within the Markdown output.  Be brief and concise."

    def __init__(self, config: ConverterConfig):
        """
        Initialize the Docling converter.

        Args:
            config: ConverterConfig object with vision_llm and metadata configuration
        """
        super().__init__(config)
        self.doc_converter = self._create_converter()

    def _ollama_vlm_options(self) -> ApiVlmOptions:
        """Create VLM options for Ollama based on the configuration."""
        vision_config = self.config.vision_llm
        model = (
            vision_config.model_name if vision_config else "granite3.2-vision:latest"
        )
        base_url = vision_config.base_url if vision_config else "http://localhost:8002"

        extractor_config = self.metadata_config.get("extractor", {})
        timeout = extractor_config.get("timeout", 600)
        scale = self.metadata_config.get("scale", 1.0)
        prompt = self.metadata_config.get("prompt", self.DEFAULT_VISION_PROMPT)

        api_url = f"{base_url}/v1/chat/completions"

        return ApiVlmOptions(
            url=api_url,
            params=dict(model=model),
            prompt=prompt,
            timeout=timeout,
            scale=scale,
            response_format=ResponseFormat.MARKDOWN,
        )

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter."""
        pipeline_options = VlmPipelineOptions(enable_remote_services=True)
        pipeline_options.vlm_options = self._ollama_vlm_options()

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )

    def convert(self, filepath: str) -> str:
        """
        Convert a document to markdown using Docling with progress tracking.

        Args:
            filepath: Path to the document to convert

        Returns:
            Markdown string representation of the document

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If conversion fails
        """
        self._validate_file_exists(filepath)

        convert_start_time = time.time()
        doc_name = pathlib.Path(filepath).name

        try:
            self.logger.info(f"🔄 Starting Docling conversion: {doc_name}")
            self.logger.info("📄 Processing document with advanced vision model...")

            result = self.doc_converter.convert(Path(filepath))

            # Export to markdown
            self.logger.info("📝 Exporting to markdown format...")
            markdown_text = result.document.export_to_markdown()

            # Log conversion statistics
            total_time = time.time() - convert_start_time
            markdown_length = len(markdown_text)

            self.logger.info("=== Docling conversion completed successfully ===")
            self.logger.info("📊 Conversion Statistics:")
            self.logger.info(f"   • Processing time: {total_time:.2f}s")
            self.logger.info(f"   • Output length: {markdown_length} characters")
            self.logger.info(
                f"   • Average processing speed: {markdown_length/total_time:.0f} chars/sec"
            )

            return markdown_text

        except Exception as e:
            self.logger.error(f"❌ Failed to convert '{doc_name}': {e}")
            raise


class DoclingVLMConverter(Converter):
    """
    Document converter using the Docling library with its default VLM.

    This converter specializes in PDF processing with advanced vision model
    integration for handling complex layouts and visual elements. It relies on
    Docling's default VLM configuration.
    """

    def __init__(self, config: ConverterConfig):
        """
        Initialize the Docling VLM converter.

        Args:
            config: ConverterConfig object (currently unused, for future compatibility).
        """
        super().__init__(config)
        self.doc_converter = self._create_converter()

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter with default VLM."""
        pipeline_options = VlmPipelineOptions(enable_remote_services=True)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )

    def convert(self, filepath: str) -> str:
        """
        Convert a document to markdown using Docling's default VLM.

        Args:
            filepath: Path to the document to convert

        Returns:
            Markdown string representation of the document

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If conversion fails
        """
        self._validate_file_exists(filepath)

        try:
            self.logger.info(f"Converting {filepath} using Docling with default VLM")
            result = self.doc_converter.convert(Path(filepath))
            markdown_text = result.document.export_to_markdown()
            self.logger.info("Conversion completed successfully")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Failed to convert '{filepath}': {e}")
            raise


@dataclass
class ExtractedImage:
    """Data class for extracted image information"""

    page_number: int
    image_index: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_data: bytes
    image_ext: str
    description: Optional[str] = None


class ImageDescriptionInterface(ABC):
    """Abstract interface for image description services"""

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


class OllamaImageDescriber(ImageDescriptionInterface):
    """Implementation for Ollama API image description"""

    def __init__(
        self, model_name: str = "llava", base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama image describer.

        Args:
            model_name: Name of the Ollama model to use for image description
            base_url: Base URL for Ollama API
        """
        # Validate inputs early to fail fast
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("OllamaImageDescriber requires a non-empty model_name")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OllamaImageDescriber requires a non-empty base_url")

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
        """
        Describe image using Ollama API.

        Args:
            image_data: Binary image data
            image_ext: Image file extension
            prompt: Custom prompt for image description

        Returns:
            Description of the image
        """
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


class DummyImageDescriber(ImageDescriptionInterface):
    """Dummy implementation for testing"""

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: str = None
    ) -> str:
        return f"[Dummy description for {image_ext} image of {len(image_data)} bytes]"


class PyMuPDFConverter(Converter):
    """
    Document converter using PyMuPDF with comprehensive content extraction.

    This converter extracts text, images, tables, and other structured data from PDF documents,
    describes images using AI services, and outputs well-formatted markdown.
    """

    DEFAULT_IMAGE_PROMPT = "Describe this image in detail. Focus on the main content, objects, text, and any relevant information that would be useful in a document context."

    def __init__(self, config: ConverterConfig):
        """
        Initialize the PyMuPDF converter.

        Args:
            config: ConverterConfig object with vision_llm and metadata configuration
        """
        super().__init__(config)

        # Configuration options from metadata
        self.preserve_formatting = self.metadata_config.get(
            "preserve_formatting", False
        )
        self.include_page_numbers = self.metadata_config.get(
            "include_page_numbers", True
        )
        self.include_metadata = self.metadata_config.get("include_metadata", True)
        self.sort_reading_order = self.metadata_config.get("sort_reading_order", True)
        self.extract_tables = self.metadata_config.get("extract_tables", True)
        self.table_strategy = self.metadata_config.get("table_strategy", "lines_strict")
        self.image_description_prompt = self.metadata_config.get(
            "image_description_prompt", self.DEFAULT_IMAGE_PROMPT
        )

        # Initialize image describer
        self.image_describer = self._create_image_describer()

    def _create_image_describer(self) -> ImageDescriptionInterface:
        """Create and configure the image describer based on configuration."""
        describer_config = self.metadata_config.get("image_describer", {})
        if not isinstance(describer_config, dict):
            raise ValueError("image_describer config must be a dictionary if provided")

        describer_type = describer_config.get("type", "ollama")

        if describer_type == "ollama":
            model_name = describer_config.get("model", "granite3.2-vision:latest")
            base_url = describer_config.get("base_url", "http://localhost:11434")
            # Validate essential params
            if not model_name or not isinstance(model_name, str):
                raise ValueError("image_describer.model must be a non-empty string")
            if not base_url or not isinstance(base_url, str):
                raise ValueError("image_describer.base_url must be a non-empty string")
            return OllamaImageDescriber(model_name=model_name, base_url=base_url)

        if describer_type == "dummy":
            return DummyImageDescriber()

        raise ValueError(f"Unsupported image_describer.type: {describer_type}")

    def _extract_tables_from_page(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        """
        Extract tables from a page using PyMuPDF's table detection.

        Args:
            page: PyMuPDF page object

        Returns:
            List of table dictionaries with position and content information
        """
        tables = []

        if not self.extract_tables:
            return tables

        try:
            # Find tables on the page
            page_tables = page.find_tables(strategy=self.table_strategy)

            for table_index, table in enumerate(page_tables):
                try:
                    # Extract table data
                    table_data = table.extract()

                    # Convert to markdown table format
                    markdown_table = self._convert_table_to_markdown(table_data)

                    tables.append(
                        {
                            "type": "table",
                            "bbox": table.bbox,
                            "table_index": table_index,
                            "data": table_data,
                            "markdown": markdown_table,
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Error extracting table {table_index}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Error finding tables on page {page.number}: {e}")

        return tables

    def _convert_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to markdown format.

        Args:
            table_data: List of lists representing table rows and cells

        Returns:
            Markdown formatted table string
        """
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

    def _extract_text_normalized(self, page: pymupdf.Page) -> str:
        """
        Extract text from page in normal reading order without preserving original formatting.

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text with normalized formatting and proper reading order
        """
        # Use block-based extraction to get proper reading order
        text_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        blocks = text_dict["blocks"]

        # Extract text blocks with positions
        text_blocks = []
        for block in blocks:
            if "lines" in block:  # Text block
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
                            "bbox": block["bbox"],
                            "text": block_text.strip(),
                            "block_no": block["number"],
                        }
                    )

        # Sort blocks by reading order (top to bottom, left to right)
        if self.sort_reading_order:
            text_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # Join text blocks with proper spacing
        normalized_text = ""
        for i, block in enumerate(text_blocks):
            normalized_text += block["text"]
            # Add paragraph break between blocks unless it's the last block
            if i < len(text_blocks) - 1:
                normalized_text += "\n\n"

        return normalized_text

    def _extract_images_from_page(self, page: pymupdf.Page) -> List[ExtractedImage]:
        """
        Extract all images from a page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of ExtractedImage objects
        """
        extracted_images = []

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
                bbox = image_rects[0] if image_rects else (0, 0, 0, 0)

                extracted_image = ExtractedImage(
                    page_number=page.number,
                    image_index=img_index,
                    bbox=bbox,
                    image_data=image_data,
                    image_ext=image_ext,
                )

                extracted_images.append(extracted_image)

            except Exception as e:
                self.logger.error(
                    f"Error extracting image {img_index} from page {page.number}: {e}"
                )
                continue

        return extracted_images

    def _get_text_blocks_with_positions(
        self, page: pymupdf.Page
    ) -> List[Dict[str, Any]]:
        """
        Get text blocks with their positions on the page, excluding table areas.

        Args:
            page: PyMuPDF page object

        Returns:
            List of text blocks with position information
        """
        # Get text as blocks to maintain position information
        text_dict = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        blocks = text_dict["blocks"]

        # Get table areas to exclude from text extraction
        table_areas = []
        if self.extract_tables:
            try:
                page_tables = page.find_tables(strategy=self.table_strategy)
                table_areas = [table.bbox for table in page_tables]
            except Exception as e:
                self.logger.warning(f"Error finding table areas: {e}")

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

    def _merge_content_by_position(
        self,
        text_blocks: List[Dict[str, Any]],
        images: List[ExtractedImage],
        tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge text blocks, images, and tables based on their positions on the page.

        Args:
            text_blocks: List of text blocks with positions
            images: List of extracted images
            tables: List of extracted tables

        Returns:
            Merged list of content in reading order
        """
        all_content = []

        # Add text blocks
        for block in text_blocks:
            all_content.append(block)

        # Add tables
        for table in tables:
            all_content.append(table)

        # Add images
        for image in images:
            all_content.append(
                {
                    "type": "image",
                    "bbox": image.bbox,
                    "image": image,
                    "block_no": 999 + image.image_index,
                }
            )

        # Sort by vertical position (top to bottom), then horizontal (left to right)
        if self.sort_reading_order:
            all_content.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        return all_content

    def convert(self, filepath: str) -> str:
        """
        Convert a PDF file to markdown with comprehensive content extraction and progress tracking.

        Args:
            filepath: Path to the PDF file

        Returns:
            Markdown formatted text with image descriptions and tables

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If conversion fails
        """
        self._validate_file_exists(filepath)

        convert_start_time = time.time()
        doc_name = pathlib.Path(filepath).name

        try:
            self.logger.info(f"🔄 Starting PDF conversion: {doc_name}")
            self.logger.info(f"📄 Opening document with PyMuPDF...")

            with pymupdf.open(filepath) as doc:
                total_pages = len(doc)
                self.logger.info(
                    f"📊 Document opened successfully - {total_pages} pages"
                )

                markdown_content = []
                stats = {
                    "total_pages": total_pages,
                    "processed_pages": 0,
                    "total_text_blocks": 0,
                    "total_images": 0,
                    "total_tables": 0,
                    "total_image_descriptions": 0,
                    "failed_image_descriptions": 0,
                }

                # Add document header
                markdown_content.append(f"# Document: {doc_name}\n\n")
                self.logger.debug("📝 Added document header")

                # Add metadata if requested
                if self.include_metadata:
                    metadata = doc.metadata
                    if metadata and any(metadata.values()):
                        self.logger.info("📋 Adding document metadata...")
                        markdown_content.append("## Document Metadata\n\n")
                        for key, value in metadata.items():
                            if value:
                                markdown_content.append(f"- **{key}**: {value}\n")
                        markdown_content.append("\n")
                        self.logger.debug("✅ Document metadata added")

                # Process pages with progress tracking
                with tqdm(
                    total=total_pages, desc="Processing pages", unit="page"
                ) as pbar:
                    for page_num in range(total_pages):
                        page_start_time = time.time()
                        page = doc[page_num]

                        pbar.set_postfix_str(f"Page {page_num + 1}/{total_pages}")
                        self.logger.debug(
                            f"📄 Processing page {page_num + 1}/{total_pages}"
                        )

                        if self.include_page_numbers:
                            markdown_content.append(f"## Page {page_num + 1}\n\n")

                        # Extract all content types
                        self.logger.debug("📝 Extracting text blocks...")
                        text_blocks = self._get_text_blocks_with_positions(page)
                        stats["total_text_blocks"] += len(text_blocks)

                        self.logger.debug("🖼️  Extracting images...")
                        images = self._extract_images_from_page(page)
                        stats["total_images"] += len(images)

                        self.logger.debug("📊 Extracting tables...")
                        tables = self._extract_tables_from_page(page)
                        stats["total_tables"] += len(tables)

                        # Describe images with progress tracking
                        if images:
                            self.logger.info(
                                f"🖼️  Describing {len(images)} images on page {page_num + 1}..."
                            )
                            for i, image in enumerate(images):
                                try:
                                    self.logger.debug(
                                        f"🔍 Describing image {i+1}/{len(images)} (Page {page_num + 1})"
                                    )
                                    description = self.image_describer.describe_image(
                                        image.image_data,
                                        image.image_ext,
                                        self.image_description_prompt,
                                    )
                                    image.description = description
                                    stats["total_image_descriptions"] += 1
                                    self.logger.debug(
                                        f"✅ Image {i+1} described successfully"
                                    )
                                except Exception as e:
                                    self.logger.error(
                                        f"❌ Failed to describe image {i+1} on page {page_num + 1}: {e}"
                                    )
                                    image.description = (
                                        f"[Error describing image: {str(e)}]"
                                    )
                                    stats["failed_image_descriptions"] += 1
                        else:
                            self.logger.debug("ℹ️  No images found on this page")

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
                                    f"![Image Description]({item['image'].image_ext})\n*{image_desc}*\n\n"
                                )
                            elif item["type"] == "table":
                                page_markdown.append(
                                    f"**Table {item['table_index'] + 1}** ({item['rows']} rows × {item['cols']} cols)\n\n"
                                )
                                page_markdown.append(item["markdown"])
                                page_markdown.append("\n\n")

                        markdown_content.extend(page_markdown)
                        markdown_content.append("\n")  # Add space between pages

                        page_time = time.time() - page_start_time
                        stats["processed_pages"] += 1
                        self.logger.debug(
                            f"✅ Page {page_num + 1} processed in {page_time:.2f}s"
                        )
                        pbar.update(1)

                # Join all content
                result = "".join(markdown_content)

                # Log final statistics
                total_time = time.time() - convert_start_time
                self.logger.info("=== PDF Conversion completed successfully ===")
                self.logger.info("📊 Conversion Statistics:")
                self.logger.info(
                    f"   • Total pages processed: {stats['processed_pages']}"
                )
                self.logger.info(
                    f"   • Text blocks extracted: {stats['total_text_blocks']}"
                )
                self.logger.info(f"   • Images found: {stats['total_images']}")
                self.logger.info(
                    f"   • Images described: {stats['total_image_descriptions']}"
                )
                if stats["failed_image_descriptions"] > 0:
                    self.logger.warning(
                        f"   • Failed image descriptions: {stats['failed_image_descriptions']}"
                    )
                self.logger.info(f"   • Tables extracted: {stats['total_tables']}")
                self.logger.info(f"   • Total processing time: {total_time:.2f}s")
                self.logger.info(
                    f"   • Average time per page: {total_time/stats['total_pages']:.2f}s"
                )
                self.logger.info(f"   • Output length: {len(result)} characters")

                return result

        except Exception as e:
            self.logger.error(f"❌ PDF conversion failed for {doc_name}: {e}")
            raise


def create_converter(converter_type: str, config: ConverterConfig) -> Converter:
    """
    Factory function to create converter instances.

    Args:
        converter_type: Type of converter ("markitdown", "docling", "docling_vlm", or "pymupdf")
        config: ConverterConfig object

    Returns:
        Converter instance

    Raises:
        ValueError: If converter_type is not supported
    """

    converters = {
        "markitdown": MarkItDownConverter,
        "docling": DoclingConverter,
        "docling_vlm": DoclingVLMConverter,
        "pymupdf": PyMuPDFConverter,
    }

    if converter_type.lower() not in converters:
        raise ValueError(
            f"Unsupported converter type: {converter_type}. "
            f"Supported types: {list(converters.keys())}"
        )

    return converters[converter_type.lower()](config)


"""
Inline test scaffolds removed; add tests under the dedicated tests/ directory.
"""
