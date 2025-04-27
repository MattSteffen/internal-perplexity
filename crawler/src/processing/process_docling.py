import logging
from pathlib import Path
from typing import Dict, Optional, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.chunking import HybridChunker

vision_prompt = "Describe the image in detail, including any text or objects present."

class DoclingConverter:
    def __init__(self, config: Dict):
        """
        Initialize a document converter with the given configuration.
        
        Args:
            config: Dictionary containing configuration options including:
                - vision_llm: Dict with model, provider, base_url
                - prompt: String prompt for the VLM
                - timeout: Optional timeout in seconds (default: 300)
                - scale: Optional image scale factor (default: 1.0)
        """
        logging.basicConfig(level=logging.INFO)
        
        self.config = config
        self.doc_converter = self._create_converter()
        
    def _create_vlm_options(self) -> ApiVlmOptions:
        """Create VLM options based on the configuration."""
        vision_config = self.config.get("vision_llm", {})
        model = vision_config.get("model")
        provider = vision_config.get("provider")
        base_url = vision_config.get("base_url")
        
        timeout = self.config.get("extractor", {}).get("timeout", 300)
        scale = 1.0
        
        # Adjust URL based on provider
        api_url = f"{base_url}/v1" if provider == "ollama" else base_url
        
        return ApiVlmOptions(
            url=api_url+"/chat/completions",
            params=dict(
                model=model,
            ),
            prompt=vision_prompt,
            timeout=timeout,
            scale=scale,
            response_format=ResponseFormat.MARKDOWN,
        )
    
    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter."""
        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True
        )
        
        pipeline_options.vlm_options = self._create_vlm_options()
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )
    
    def convert(self, filepath: Union[str, Path]) -> ConversionResult:
        """
        Convert a document to markdown.
        
        Args:
            filepath: Path to the document to convert
            
        Returns:
            ConversionResult object containing the converted document
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
            
        result = self.convert_to_document(filepath)
        return result.document.export_to_markdown()
    
    def convert_to_document(self, filepath: Union[str, Path]) -> ConversionResult:
        """
        Convert a document and return the markdown text.
        
        Args:
            filepath: Path to the document to convert
            
        Returns:
            Markdown string representation of the document
        """
        self.document = self.doc_converter.convert(filepath)
        return self.document

    def chunk_smart(self, text: str, chunk_size: int = 5000) -> list[str]:
        """
        Split text into chunks of a specified size.

        Args:
            text: The text to split into chunks
            chunk_size: The maximum size of each chunk

        Returns:
            List of chunks
        """
        chunker = HybridChunker(max_tokens=chunk_size//4)
        c_iter = chunker.chunk(dl_doc=self.document.document)
        return [c_iter.text for c_iter in c_iter]

    
    def chunk_length(self, text: str, chunk_size: int = 5000) -> list[str]:
        """
        Split text into chunks of a specified size.

        Args:
            text: The text to split into chunks
            chunk_size: The maximum size of each chunk

        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks
