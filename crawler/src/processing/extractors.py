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
"""

import os
import yaml
import json
from typing import Dict, Any, Tuple, Generator, List, Optional
from abc import ABC, abstractmethod
from langchain_ollama import ChatOllama
from langchain_groq import  ChatGroq
from langchain.chat_models import init_chat_model
# import docx
import PyPDF2
import markdown
import pptx
from bs4 import BeautifulSoup
import csv


class ExtractorHandler(ABC):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
    
    @abstractmethod
    def extract(self):
        """Yields text and metadata from documents."""
        pass
    
    def files(self, extension: str):
        """Yields paths to files with given extension in the directory."""
        for file_name in os.listdir(self.dir_path):
            if file_name.endswith(extension):
                yield os.path.join(self.dir_path, file_name)


class DocumentReader:
    """Handles reading different document types and extracting text."""
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    
    # @staticmethod
    # def read_docx(file_path: str) -> str:
    #     doc = docx.Document(file_path)
    #     return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def read_pptx(file_path: str) -> str:
        prs = pptx.Presentation(file_path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    
    @staticmethod
    def read_md(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            md_content = f.read()
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    
    @staticmethod
    def read_html(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
    
    @staticmethod
    def read_csv(file_path: str) -> str:
        text = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text.append(",".join(row))
        return "\n".join(text)
    
    @staticmethod
    def read_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return json.dumps(json.load(f), indent=2)


class LLMHandler(ExtractorHandler):
    def __init__(self, dir_path: str, schema_path: str = "crawler/src/storage/document.json", 
                 ollama_model: str = "llama3.2:1b", ollama_base_url: str = "http://localhost:11434"):
        super().__init__(dir_path)
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(model=ollama_model, base_url=ollama_base_url)
        self.llm = ChatGroq(model=ollama_model, base_url=ollama_base_url)
        self.llm = init_chat_model(ollama_model, model_provider="groq")
        
        # Create structured output parser from YAML schema
        self.structured_llm = self._setup_output_llm(schema_path)
        
        # Document readers mapping
        self.doc_readers = {
            '.txt': DocumentReader.read_txt,
            '.pdf': DocumentReader.read_pdf,
            # '.docx': DocumentReader.read_docx,
            # '.doc': DocumentReader.read_docx,
            '.pptx': DocumentReader.read_pptx,
            '.ppt': DocumentReader.read_pptx,
            '.md': DocumentReader.read_md,
            '.html': DocumentReader.read_html,
            '.htm': DocumentReader.read_html,
            '.csv': DocumentReader.read_csv,
            '.json': DocumentReader.read_json
        }

    def _setup_output_llm(self, json_schema_path):
        """Create a structured output parser from the YAML schema at the given file path."""
        # Load the schema from the YAML file (already loaded in __init__, but keeping this for clarity)
        with open(json_schema_path, 'r') as file:
            schema = json.load(file)
        
        structured_llm = self.llm.with_structured_output(schema)
        return structured_llm
    
    def _extract_metadata_prompt(self, text: str) -> str:
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
            return llm_response # TODO: Should be a dict
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
        
        for i, file_path in enumerate(all_files):
            ext = self.get_file_extension(file_path)
            
            # Skip unsupported file types
            if ext not in self.doc_readers:
                print(f"Skipping unsupported file type: {file_path}")
                continue
                
            try:
                print(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
                
                # Extract text using the appropriate reader
                text = self.doc_readers[ext](file_path)
                
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


def main():
    dir = "../../../data/conference/"
    schema = "../../src/storage/document.json"
    model = "llama-3.3-70b-versatile"
    output = "output.json"


    print(f"Processing documents in: {dir}")
    print(f"Using schema from: {schema}")
    print(f"Using Ollama model: {model}")
    
    # Initialize the handler
    handler = LLMHandler(
        dir_path=dir,
        schema_path=schema,
        ollama_model=model,
    )
    
    # Extract and collect metadata
    all_metadata = []
    for text, metadata in handler.extract():
        print(f"\nExtracted metadata from {metadata['source']}:")
        print(json.dumps(metadata, indent=2))
        all_metadata.append(metadata)
    
    # Save all metadata to a JSON file
    with open(output, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nProcessed {len(all_metadata)} documents")
    print(f"Saved all metadata to {output}")


format_instructions = """Extract the metadata fields from the text following the schema provided."""



if __name__ == "__main__":
    main()