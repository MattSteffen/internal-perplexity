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
from typing import Dict, Any, Tuple, Generator, List, Optional, Union
import base64

from langchain.chat_models import init_chat_model
from PIL import Image

from typing import Union
import io
from langchain_core.messages import HumanMessage

from processing.readers import *
vision_prompt = "Describe the image in detail, including any text or objects present."


class VisionLLM():
    def __init__(self, model_name: str = "gemma3", model_provider="ollama", base_url: str = "http://localhost:11434"):
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
    def __init__(self, llm, vision_llm: VisionLLM, schema: str):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(schema)
        self.vision_llm = vision_llm
        
        # Document readers mapping
        # TODO: Enable only those that are enabled in the config
        self.doc_readers = {
            '.txt': TextReader(self.llm, self.vision_llm),
            '.pdf': PDFReader(self.llm, self.vision_llm),
            '.md': MarkdownReader(self.llm, self.vision_llm),
            '.html': HTMLReader(self.llm, self.vision_llm),
            '.csv': CSVReader(self.llm, self.vision_llm),
            '.json': JSONReader(self.llm, self.vision_llm),
            '.docx': DocxReader(self.llm, self.vision_llm),
            '.doc': DocxReader(self.llm, self.vision_llm),
            '.pptx': PptxReader(self.llm, self.vision_llm),
        }
    
    def _extract_metadata_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to extract metadata from the text."""
        format_instructions = "Extract the metadata fields from the text following the schema provided."
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

    def extract(self, all_files: list[str]) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Yields text and metadata from all supported document types."""
        if isinstance(all_files, str):
            all_files = [all_files]
        
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



sample_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Document",
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "maxLength": 1024,
      "description": "Text content of the document chunk."
    },
    "source": {
      "type": "string",
      "maxLength": 1024,
      "description": "Source identifier of the document chunk."
    },
    "title": {
      "type": "string",
      "maxLength": 255,
      "description": "Title of the document."
    },
    "author": {
      "type": "array",
      "maxItems": 255,
      "items": {
        "type": "string",
        "description": "An author of the document."
      },
      "description": "List of authors of the document."
    },
    "author_role": {
      "type": "string",
      "maxLength": 255,
      "description": "Role of the author in the document (e.g., writer, editor)."
    },
    "url": {
      "type": "string",
      "maxLength": 1024,
      "description": "URL associated with the document."
    },
    "chunk_index": {
      "type": "integer",
      "description": "Index of the document chunk."
    }
  }
}


def test():
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq") # Must support structured output
    vision_llm = VisionLLM()
    extractor = Extractor(llm, vision_llm, sample_schema)
    all_files = [
        "/Users/mattsteffen/projects/llm/internal-perplexity/data/sample/c4611_sample_explain.pdf",
        "/Users/mattsteffen/projects/llm/internal-perplexity/data/conference/Simple_Is_the_Doctrine_of_Jesus_Christ.json"
    ]

    print(list(extractor.extract(all_files)))

# if __name__ == "__main__":
#     test()