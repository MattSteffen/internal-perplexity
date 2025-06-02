from typing import Dict, Any, List
import logging
from abc import ABC, abstractmethod
from .llm import LLM



"""
# TODO: Make an Abstract Base Class for Extractor
Extractor:
- Accepts:
  - Markdown text
Does:
  - Extracts metadata according to json schema
  - Chunks the text
- Outputs:
  - List of chunks with metadata
"""

class Extractor(ABC):
    """
    Abstract base class for document extractors.
    
    This class defines the interface for extracting metadata and chunking text
    from documents. All extractor implementations should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the extractor with configuration.
        
        Args:
            config: Dictionary containing configuration options
        """
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the extractor."""
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)
    
    
    @abstractmethod
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from the given text.
        
        Args:
            text: Text to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Chunk the text into smaller pieces.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        pass

class BasicExtractor(Extractor):
    def __init__(self, metadata_schema: Dict[str, Any], llm: LLM):
        super().__init__(metadata_schema)
        self.metadata_schema = metadata_schema
        self.llm = llm
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata and chunks from the text."""
        if self.metadata_schema is not None:
            return self.llm.invoke(self._get_prompt(text), response_format=self.metadata_schema)
        else:
            return {}
    
    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Chunk the text into chunks of the specified size."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks
    
    def _get_prompt(self, text: str) -> str:
        return f"Extract the metadata according to the provided schema for the following text:\n{text}"


def test():
    # Test the extractor
    metadata_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "author", "date", "tags"],
    }
    llm = LLM(
        model_name="gemma3",
        ollama_base_url="http://localhost:11434"
    )
    extractor = BasicExtractor(metadata_schema, llm)
    text = "This is a test document written by John Doe on 2023-01-01. It contains some text about the topic signal processing."
    results = extractor.extract_metadata(text)
    print(results)

if __name__ == "__main__":
    test()