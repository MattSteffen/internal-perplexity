import logging
import os
from pathlib import Path
from typing import Dict, Union

from markitdown import (
    MarkItDown,
    UnsupportedFormatException,
    FileConversionException,
    MissingDependencyException,
)

from openai import OpenAI

class MarkItDownConverter:
    def __init__(self, config: Dict):
        """
        Initialize a document converter with the given configuration.
        
        Args:
            config: Dictionary containing configuration options including:
                - vision_llm: Dict with model, provider, base_url
                - prompt: String prompt for the VLM
                - timeout: Optional timeout in seconds (default: 300)
        """
        logging.basicConfig(level=logging.INFO)
        
        self.config = config
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
        
        # Create and return the MarkItDown instance
        return MarkItDown(
            llm_client=client,
            llm_model=model,
            enable_plugins=False,
        )
    
    def convert(self, filepath: Union[str, Path]) -> str:
        """
        Convert a document to markdown.
        
        Args:
            filepath: Path to the document to convert
            
        Returns:
            Markdown string representation of the document
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: Input file not found at {filepath}")
        
        try:
            result = self.markitdown.convert(filepath)
            return result.markdown
        except MissingDependencyException as e:
            logging.error(f"Conversion Error: Missing dependency. {e}")
            raise
        except UnsupportedFormatException as e:
            logging.error(f"Conversion Error: Unsupported file format for '{filepath}'. {e}")
            raise
        except FileConversionException as e:
            logging.error(f"Conversion Error: Failed to convert '{filepath}'. {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during conversion: {e}")
            raise

    def chunk_smart(self, text: str, chunk_size: int = 5000) -> list[str]:
        """
        Split text into chunks of a specified size.

        Args:
            text: The text to split into chunks
            chunk_size: The maximum size of each chunk

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if current_size + len(paragraph) > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 2  # +2 for the \n\n
            
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

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