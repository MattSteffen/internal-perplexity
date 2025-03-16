"""
Document Processor

This module provides a simplified interface for processing documents,
extracting metadata, and generating embeddings.
"""

import os
from typing import Dict, Any, List, Tuple, Generator, Optional, Union

from config.config_manager import ConfigManager
from .extractor import Extractor, VisionLLM
from .embeddings import LocalEmbedder

class DocumentProcessor:
    """Main interface for document processing."""
    
    def __init__(self, 
                config: ConfigManager,
                llm,
                vision_llm: VisionLLM,
                embedder: LocalEmbedder):
        """Initialize the document processor.
        
        Args:
            config_path: Optional path to configuration file
            schema_path: Path to metadata schema file
            embedding_model: Optional embedding model name (overrides config)
            llm_model: Optional LLM model name (overrides config)
            base_url: Optional base URL for API calls (overrides config)
        """
        # Set up extractor args
        self.llm = llm 
        self.vision_llm = vision_llm
        self.schema = config.get("metadata", {}).get("schema", {})
        self.extractor = Extractor(self.llm, self.vision_llm, self.schema)
        
        # Set up embedder
        self.embedder = embedder
    
    def process_documents(self, 
                         file_paths: List[str]) -> Generator[Tuple[str, Dict[str, Any], List[float]], None, None]:
        """Process documents and generate embeddings.
        
        Args:
            file_paths: List of paths to documents to process
            
        Yields:
            Tuple of (text, metadata, embedding)
        """
        # TODO: What does the extractor return, in the metadata is there something we need to embed too?
        # Process documents
        for text, metadata in self.extractor.extract(file_paths):
            # Generate embedding
            embedding = self.embedder.embed_query(text)
            # TODO: Check config for extra embedding fields (like "summary1", "summary2", etc.), and embed those too
            # then do something like for embedding in embeddings: yield text, metadata, embedding
            
            yield text, metadata, embedding
    
    def process_directory(self, 
                         directory_path: str, 
                         extensions: Optional[List[str]] = None) -> Generator[Tuple[str, Dict[str, Any], List[float]], None, None]:
        """Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            extensions: Optional list of file extensions to process
            
        Yields:
            Tuple of (text, metadata, embedding)
        """
        # Get list of files
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                if extensions is None or ext.lower() in extensions:
                    file_paths.append(file_path)
        
        # Process files
        yield from self.process_documents(file_paths)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        return self.embedder.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        return self.embedder.embed_documents(texts) 