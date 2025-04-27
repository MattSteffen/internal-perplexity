"""
Document Metadata Extractor

This module provides a framework for extracting text and metadata from various document formats
using LLMs. It supports multiple file types including TXT, PDF, DOCX, PPTX, MD, HTML, and CSV files.
The system uses Ollama to process document content and extract structured metadata according to
a predefined schema from a YAML configuration file.


Extraction process:
1. Identify files in the directory with the specified extension.
2. For each file, read the content using the appropriate reader.
3. Extract metadata using the LLM. Import this metadata from a YAML file converted to JSON schema.
4. Convert any images to descriptions using the vision LLM if available.
  - There is llm.with_structured_output(schema) and vision_llm to get description of images.
5. Convert tables and make sure they are in a consistent format.
6. Return the extracted text and metadata.
"""

import os
import json
from typing import Dict, Any, Tuple, Generator, List, Optional, Union
import base64

from .process_docling import DoclingConverter
from .process_markitdown import MarkItDownConverter

from langchain.chat_models import init_chat_model
from PIL import Image

from typing import Union
import io
from langchain_core.messages import HumanMessage


format_instructions = "Extract the metadata fields from the text following the schema provided."


class Extractor():
    def __init__(self, llm, config: dict[str, Any] = {}):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(config.get("metadata").get("schema", {}).copy())
        self.config = config
        self.extractor_config = config.get("extractor", {})
        self.metadata_config = config.get("metadata", {})
        self.converter = self._create_converter()

    def _create_converter(self) -> Any:
        """Create a converter based on the configuration."""
        match self.extractor_config.get("engine", "docling"):
            case "docling":
                return DoclingConverter(self.config)
            case "markitdown":
                return MarkItDownConverter(self.config)
            case _: # TODO: Implement markitdown converter
                raise ValueError(f"Unknown engine: {self.extractor_config.get('engine')}")
        
    def _extract_metadata_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to extract metadata from the text."""
        return f"Extract metadata from the following text according to these guidelines:\n{format_instructions}\n\nText excerpt (analyze the full text even if truncated here):\n{text[:10000]}... [text continues]\n\nReturn your analysis in the required JSON format."

    def _extract_metadata_to_embed(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract metadata from the entire document based on configured fields."""
        if not self.extractor_config.get("metadata", {}).get("enabled", False):
            return []
            
        metadata_fields = self.metadata_config.get("metadata", {}).get("semantic_search", [])
        if not metadata_fields:
            return []
            
        # Get the text to be embedded
        text_from_metadata_to_embed = []
        for field in metadata_fields:
            if metadata.get(field, ""):
                text_from_metadata_to_embed.append(metadata[field])
        return text_from_metadata_to_embed

    def extract_metadata_with_llm(self, text: str) -> Dict[str, Any]:
        """Use the LLM to extract metadata from the text."""
        prompt = self._extract_metadata_prompt(text)
        llm_response = None
        try:
            llm_response = self.structured_llm.invoke(prompt)
            if isinstance(llm_response, dict):
                return llm_response
            else:
                return json.loads(llm_response.content.replace("```json", "").replace("```", ""))
        except Exception as e:
            print(f"Error parsing LLM metadata response: {e}, response: {llm_response}")
            return {}

    def extract(self, filepath: str) -> list[Dict[str, Any]]:
        """Returns list of (text, metadata) tuples from all supported document types."""
        print("Extracting: ", filepath)

        results = []
        try:
            document_markdown = self.converter.convert(filepath)
            # metadata
            metadata = {'source': filepath}
            metadata.update(self.extract_metadata_with_llm(document_markdown))

            # chunking
            chunks = []
            if self.extractor_config.get("chunking", {}).get("enabled", False):
                if self.extractor_config.get("chunking", {}).get("engine", "smart") == "smart":
                    chunks = self.converter.chunk_smart(document_markdown, self.extractor_config.get("chunking", {}).get("chunk_size", 5000))
                else:
                    chunks = self.converter.chunk_length(document_markdown, self.extractor_config.get("chunking", {}).get("chunk_size", 5000))
            else:
                chunks = [document_markdown]
            
            chunks.extend(self._extract_metadata_to_embed(metadata))
            
            for i,chunk in enumerate(chunks):
                meta = metadata.copy()
                meta["text"] = chunk
                meta["chunk_index"] = i
                results.append(meta)

            print("got results", len(results))
            return results
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")