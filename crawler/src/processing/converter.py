"""
Document conversion utilities with support for multiple conversion backends.

This module provides a base converter class and implementations for different 
document conversion libraries including MarkItDown and Docling.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
from openai import OpenAI

# MarkItDown imports
from markitdown import (
    MarkItDown,
    UnsupportedFormatException,
    FileConversionException,
    MissingDependencyException,
)

# # Docling imports
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.document import ConversionResult
# from docling.datamodel.pipeline_options import (
#     ApiVlmOptions,
#     ResponseFormat,
#     VlmPipelineOptions,
# )
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.pipeline.vlm_pipeline import VlmPipeline


class Converter(ABC):
    """
    Abstract base class for document converters.
    
    This class defines the interface that all document converters must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the converter with configuration.
        
        Args:
            config: Dictionary containing configuration options
        """
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the converter."""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarkItDown converter.
        
        Args:
            config: Configuration dictionary with keys:
                - vision_llm: Dict with model, provider, base_url
                - prompt: String prompt for the VLM
                - timeout: Optional timeout in seconds (default: 300)
        """
        super().__init__(config)
        self.markitdown = self._create_converter()
    
    def _create_converter(self) -> MarkItDown:
        """Create and configure the MarkItDown converter."""
        vision_config = self.config.get("vision_llm", {})
        model = vision_config.get("model", "gemma3")
        provider = vision_config.get("provider", "ollama")
        base_url = vision_config.get("base_url", "http://localhost:11434")
        
        # Adjust URL based on provider
        api_url = f"{base_url}/v1" if provider == "ollama" else base_url
        
        # Initialize the LLM client
        client = OpenAI(base_url=api_url, api_key="ollama")
        
        return MarkItDown(
            llm_client=client,
            llm_model=model,
            enable_plugins=False,
        )
    
    def convert(self, filepath: str) -> str:
        """
        Convert a document to markdown using MarkItDown.
        
        Args:
            filepath: Path to the document to convert
            
        Returns:
            Markdown string representation of the document
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            MissingDependencyException: If required dependencies are missing
            UnsupportedFormatException: If the file format is not supported
            FileConversionException: If conversion fails
        """
        self._validate_file_exists(filepath)
        
        try:
            self.logger.info(f"Converting {filepath} using MarkItDown")
            result = self.markitdown.convert(filepath)
            self.logger.info("Conversion completed successfully")
            return result.markdown
            
        except MissingDependencyException as e:
            self.logger.error(f"Missing dependency: {e}")
            raise
        except UnsupportedFormatException as e:
            self.logger.error(f"Unsupported file format for '{filepath}': {e}")
            raise
        except FileConversionException as e:
            self.logger.error(f"Failed to convert '{filepath}': {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during conversion: {e}")
            raise


# class DoclingConverter(Converter):
#     """
#     Document converter using the Docling library.
    
#     This converter specializes in PDF processing with advanced vision model
#     integration for handling complex layouts and visual elements.
#     """
    
#     DEFAULT_VISION_PROMPT = "Describe the image in detail, including any text or objects present."
    
#     def __init__(self, config: Dict[str, Any]):
#         """
#         Initialize the Docling converter.
        
#         Args:
#             config: Configuration dictionary with keys:
#                 - vision_llm: Dict with model, provider, base_url
#                 - prompt: String prompt for the VLM
#                 - extractor: Dict with timeout and other extraction options
#                 - scale: Optional image scale factor (default: 1.0)
#         """
#         super().__init__(config)
#         self.doc_converter = self._create_converter()
    
#     def _create_vlm_options(self) -> ApiVlmOptions:
#         """Create VLM options based on the configuration."""
#         vision_config = self.config.get("vision_llm", {})
#         model = vision_config.get("model", "gemma3")
#         provider = vision_config.get("provider", "ollama")
#         base_url = vision_config.get("base_url", "http://localhost:11434")
        
#         extractor_config = self.config.get("extractor", {})
#         timeout = extractor_config.get("timeout", 300)
#         scale = self.config.get("scale", 1.0)
#         prompt = self.config.get("prompt", self.DEFAULT_VISION_PROMPT)
        
#         # Adjust URL based on provider
#         api_url = f"{base_url}/v1" if provider == "ollama" else base_url
        
#         return ApiVlmOptions(
#             url=f"{api_url}/chat/completions",
#             params={"model": model},
#             prompt=prompt,
#             timeout=timeout,
#             scale=scale,
#             response_format=ResponseFormat.MARKDOWN,
#         )
    
#     def _create_converter(self) -> DocumentConverter:
#         """Create and configure the document converter."""
#         pipeline_options = VlmPipelineOptions(enable_remote_services=True)
#         pipeline_options.vlm_options = self._create_vlm_options()
        
#         return DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options,
#                     pipeline_cls=VlmPipeline,
#                 )
#             }
#         )
    
#     def convert(self, filepath: str) -> str:
#         """
#         Convert a document to markdown using Docling.
        
#         Args:
#             filepath: Path to the document to convert
            
#         Returns:
#             Markdown string representation of the document
            
#         Raises:
#             FileNotFoundError: If the input file doesn't exist
#             Exception: If conversion fails
#         """
#         self._validate_file_exists(filepath)
        
#         try:
#             self.logger.info(f"Converting {filepath} using Docling")
#             result = self.doc_converter.convert(filepath)
#             markdown_text = result.document.export_to_markdown()
#             self.logger.info("Conversion completed successfully")
#             return markdown_text
            
#         except Exception as e:
#             self.logger.error(f"Failed to convert '{filepath}': {e}")
#             raise


def create_converter(converter_type: str, config: Dict[str, Any]) -> Converter:
    """
    Factory function to create converter instances.
    
    Args:
        converter_type: Type of converter ("markitdown" or "docling")
        config: Configuration dictionary
        
    Returns:
        Converter instance
        
    Raises:
        ValueError: If converter_type is not supported
    """
    converters = {
        "markitdown": MarkItDownConverter,
        # "docling": DoclingConverter,
    }
    
    if converter_type.lower() not in converters:
        raise ValueError(f"Unsupported converter type: {converter_type}. "
                        f"Supported types: {list(converters.keys())}")
    
    return converters[converter_type.lower()](config)


# Example usage and testing functions
def test_markitdown_converter():
    """Test the MarkItDown converter with sample configuration."""
    config = {
        "vision_llm": {
            "model": "gemma3",
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        },
        "prompt": "You are a helpful assistant.",
        "timeout": 300
    }
    
    converter = MarkItDownConverter(config)
    # Replace with actual file path for testing
    # filepath = "/path/to/your/document.pdf"
    # markdown = converter.convert(filepath)
    # print(markdown)
    print("MarkItDown converter created successfully")


def test_docling_converter():
    """Test the Docling converter with sample configuration."""
    config = {
        "vision_llm": {
            "model": "gemma3",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        },
        "extractor": {
            "timeout": 300,
        },
    }
    
    # converter = DoclingConverter(config)
    # Replace with actual file path for testing
    # filepath = "/path/to/your/document.pdf"
    # markdown = converter.convert(filepath)
    # print(markdown)
    print("Docling converter created successfully")


def test_factory_function():
    """Test the converter factory function."""
    config = {
        "vision_llm": {
            "model": "gemma3",
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        }
    }
    
    # Test creating different converter types
    markitdown_converter = create_converter("markitdown", config)
    docling_converter = create_converter("docling", config)
    
    print(f"Created MarkItDown converter: {type(markitdown_converter).__name__}")
    print(f"Created Docling converter: {type(docling_converter).__name__}")


if __name__ == "__main__":
    test_markitdown_converter()
    test_docling_converter()
    test_factory_function()