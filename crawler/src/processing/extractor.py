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

from langchain.chat_models import init_chat_model
from PIL import Image

from typing import Union
import io
from langchain_core.messages import HumanMessage

from processing.readers import implemented_doc_readers
vision_prompt = "Describe the image in detail, including any text or objects present."


class VisionLLM():
    def __init__(self, model_name: str, model_provider: str, base_url: str):
        # Initialize vision LLM via langchain
        self.llm = init_chat_model(model_name, model_provider=model_provider, base_url=base_url)

    def _encode_image(self, image_path_or_data: Union[str, bytes, Image.Image]) -> str:
        """
        Encode an image to base64 string.
        Args:
            image_path_or_data: Path to image file, bytes of image, or PIL Image
        Returns:
            Base64 encoded string of the image
        """
        match image_path_or_data:
            case str():
                with open(image_path_or_data, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            case bytes():
                return base64.b64encode(image_path_or_data).decode('utf-8')
            case Image.Image():
                buffer = io.BytesIO()
                image_path_or_data.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            case _:
                raise ValueError("Unsupported image input type")

    def invoke(self, image_path_or_data: Union[str, bytes, Image.Image],  prompt: str = vision_prompt) -> str:
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



class Extractor():
    def __init__(self, llm, vision_llm: VisionLLM, config: dict[str, Any] = {}):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(config.get("metadata").get("schema", {}))
        self.vision_llm = vision_llm
        self.doc_readers = {}
        self.extractor_config = config.get("extractor", {})
        self.metadata_config = config.get("metadata", {})

        # Document readers mapping
        for ext in self.extractor_config.get("document_readers", {}):
            if ext.get("enabled", True) and ext.get("type", "") != "" and ext.get("type", "") in implemented_doc_readers:
                print(f"Loading {ext['type']} reader")
                self.doc_readers[ext["type"]] = implemented_doc_readers[ext["type"]](self.llm, self.vision_llm)
        

    def _extract_metadata_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to extract metadata from the text."""
        format_instructions = "Extract the metadata fields from the text following the schema provided."
        return f"""Extract metadata from the following text according to these guidelines:

{format_instructions}

Text excerpt (analyze the full text even if truncated here):
{text[:10000]}... [text continues]

Return your analysis in the required JSON format."""

    def _extract_document_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract metadata from the entire document based on configured fields."""
        if not self.extractor_config.get("chunking", {}).get("metadata", {}).get("enabled", False):
            return []
            
        metadata_fields = self.metadata_config.get("metadata", {}).get("extra_embeddings", [])
        if not metadata_fields:
            return []
            
        # Get the text to be embedded
        fields = []
        for field in metadata_fields:
            if metadata.get(field, ""):
                fields.append(metadata[field])
        return fields

    def _chunk_by_length(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk text by length with overlap and return list of text chunks with metadata."""
        length_config = self.extractor_config.get("chunking", {}).get("length", {})
        
        if not length_config.get("enabled", False):
            return [(text, metadata)]
            
        chunk_size = length_config.get("chunk_size", 5000)
        overlap = length_config.get("overlap", 100)
        
        # Split text into chunks with overlap
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            print("Segmenting: ", start, end, len(text))
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            
            chunks.append((chunk, chunk_metadata))
            
            # Update for next iteration
            start = end - overlap
            chunk_index += 1
            if end == len(text):
                break
            
        return chunks

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
            print(f"Error parsing LLM response: {e}, response: {llm_response}")
            return {}

    def get_file_extension(self, file_path: str) -> str:
        """Get the file extension from a path."""
        _, ext = os.path.splitext(file_path)
        return ext.lower()[1:]

    def extract(self, all_files: list[str]) -> list[tuple[str, Dict[str, Any]]]:
        """Returns list of (text, metadata) tuples from all supported document types."""
        if isinstance(all_files, str):
            all_files = [all_files]

        print("Extracting: ", all_files)
        
        results = []
        for i, file_path in enumerate(all_files):
            ext = self.get_file_extension(file_path)
            
            # Skip unsupported file types
            if ext not in self.doc_readers:
                print(f"Skipping unsupported file type - {ext}: {file_path}")
                continue
                
            try:
                print(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
                
                # Extract text using the appropriate reader
                doc = self.doc_readers[ext].read(file_path)
                text = doc.get_text()
                print("got text, length", len(text))
                
                # Initialize base metadata
                metadata = {
                    'source': file_path,
                    'format': ext,  # Remove the leading dot
                }
                
                metadata.update(self.extract_metadata_with_llm(text))
                print("got metadata", metadata)
                
                # Chunk by length if enabled and also embed metadata
                results.extend(self._chunk_by_length(text, metadata))
                chunk = len(results)
                for i,new_text in enumerate(self._extract_document_metadata(metadata)):
                    new_metadata = metadata.copy()
                    new_metadata["chunk_index"] = chunk + i
                    results.append((new_text, new_metadata))
                print("got results", len(results))
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return results