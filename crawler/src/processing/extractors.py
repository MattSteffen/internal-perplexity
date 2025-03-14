"""
Document Metadata Extractor

This module provides a framework for extracting text and metadata from various document formats
using LLMs. It supports multiple file types including TXT, PDF, DOCX, PPTX, MD, HTML, and CSV files.
The system uses Ollama to process document content and extract structured metadata according to
a predefined schema from a YAML configuration file.

Key components:
- ExtractorHandler: Abstract base class for document processing
- DocumentReader: Utility class with static methods for reading different file formats
- LLMHandler: Implementation that extracts text and uses an LLM to generate structured metadata

Usage:
    handler = LLMHandler(directory_path, schema_path, ollama_model, ollama_base_url)
    for text, metadata in handler.extract():
        process_document(text, metadata)

Dependencies:
    - langchain
    - ollama
    - PyPDF2
    # - python-docx
    - python-pptx
    - markdown
    - beautifulsoup4
    - pyyaml


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
import yaml
import json
from typing import Dict, Any, Tuple, Generator, List, Optional, Union
import base64
from abc import ABC, abstractmethod
from langchain_ollama import ChatOllama
from langchain_groq import  ChatGroq
from langchain.schema import HumanMessage

from langchain.chat_models import init_chat_model
# import docx
import PyPDF2
import markdown
import pptx
from bs4 import BeautifulSoup
import csv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from typing import Union
import base64
import io
from PIL import Image
from langchain_core.messages import HumanMessage

from document_handlers import *

class LLMHandler():
    def __init__(self, schema_path: str = "crawler/src/storage/document.json", 
                 model_name: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        
        # Initialize Ollama LLM
        self.llm = init_chat_model(model_name, model_provider="groq", base_url=base_url)
        
        # Create structured output parser from YAML schema
        self.structured_llm = self._setup_output_llm(schema_path)

        # vision LLM
        self.vision_llm = None
        
        # Document readers mapping
        self.doc_readers = {
            '.txt': TextReader,
            '.pdf': PDFReader,
            '.md': MarkdownReader,
            '.html': HTMLReader,
            '.csv': CSVReader,
            '.json': JSONReader,
            # '.docx': DocxReader,
            # '.doc': DocxReader,
            # '.pptx': PPTXReader
        }

    def _setup_output_llm(self, filepath: str):
        """Create a structured output parser from the YAML schema at the given file path."""
        schema = dict()
        if filepath.endswith('yaml'):
            with open(filepath, 'r') as f:
                schema = yaml.safe_load(f)
        elif filepath.endswith('json'):
            with open(filepath, 'r') as f:
                schema = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        structured_llm = self.llm.with_structured_output(schema)
        return structured_llm
    
    def _extract_metadata_prompt(self, text: str) -> str:
        format_instructions = "Extract the metadata fields from the text following the schema provided."
        """Create a prompt for the LLM to extract metadata from the text."""
        return f"""Extract metadata from the following text according to these guidelines:

{format_instructions}

Text excerpt (analyze the full text even if truncated here):
{text[:1500]}... [text continues]

Return your analysis in the required JSON format."""

    def extract_metadata_with_llm(self, text: str) -> Dict[str, Any]:
        """Use the LLM to extract metadata from the text."""
        prompt = self._extract_metadata_prompt(text)
        try:
            llm_response = self.structured_llm.invoke(prompt)
            print(f"LLM response: {llm_response}")
            return llm_response # dict matching metadata schema
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Return a basic metadata object if parsing fails
            return {}

    def get_file_extension(self, file_path: str) -> str:
        """Get the file extension from a path."""
        _, ext = os.path.splitext(file_path)
        return ext.lower()

    def extract(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Yields text and metadata from all supported document types."""
        # Get all files in the directory
        all_files = [
            os.path.join(self.dir_path, f) 
            for f in os.listdir(self.dir_path) 
            if os.path.isfile(os.path.join(self.dir_path, f))
        ]
        
        for i, file_path in enumerate(all_files): # can this be the extractor yeild thing?
            ext = self.get_file_extension(file_path)
            
            # Skip unsupported file types
            if ext not in self.doc_readers:
                print(f"Skipping unsupported file type: {file_path}")
                continue
                
            try:
                print(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
                
                # Extract text using the appropriate reader
                doc = self.doc_readers[ext].read(file_path)
                text = doc.get_text()
                
                # Extract metadata using LLM
                metadata = self.extract_metadata_with_llm(text)
                
                # Add file-specific metadata
                metadata.update({
                    'source': file_path,
                    'format': ext[1:],  # Remove the leading dot
                    'chunk_index': i,
                })
                
                yield text, metadata
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


class VisionLLM():
    def __init__(self, model_name: str = "llama3.2:1b", model_provider="ollama", base_url: str = "http://localhost:11434"):
        # Initialize Ollama LLM
        self.llm = init_chat_model(model_name, model_provider=model_provider, base_url=base_url)

    def _encode_image(self, image_path_or_data: Union[str, bytes, Image.Image]) -> str:
        """
        Encode an image to base64 string.
        Args:
            image_path_or_data: Path to image file, bytes of image, or PIL Image
        Returns:
            Base64 encoded string of the image
        """
        if isinstance(image_path_or_data, str):
            with open(image_path_or_data, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_path_or_data, bytes):
            return base64.b64encode(image_path_or_data).decode('utf-8')
        elif isinstance(image_path_or_data, Image.Image):
            buffer = io.BytesIO()
            image_path_or_data.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("Unsupported image input type")

    def invoke(self, prompt: str, image_path_or_data: Union[str, bytes, Image.Image]) -> str:
        """
        Send a prompt with an image to the vision model
        Args:
            prompt: Text prompt to send with the image
            image_path_or_data: Image to analyze (path, bytes or PIL Image)
        Returns:
            Model response as string
        """
        image_base64 = self._encode_image(image_path_or_data)
        
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"},
                {"type": "text", "text": prompt}
            ]
        )
        
        response = self.llm.invoke([message])
        return response.content

