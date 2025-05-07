import json
import os
import argparse
from typing import Dict, Any, Iterator, Tuple, Generator, Optional
# from discovery import find_dirs
from processing.embeddings import LocalEmbedder
from processing.extractor import Extractor
from storage.vector_db import VectorStorage
from config.config_manager import ConfigManager
from langchain.chat_models import init_chat_model
"""
TODO:
  - Simplify config management, should only use a single config file.
  - Retest hybrid search, I think I found a bug in the vector db.
"""

class Crawler:
    """
    Main class for crawling and processing directories of documents.
    Attributes:
        config: Configuration dictionary.
        llm: Language model instance for processing documents.
        embedder: Embedder instance for generating embeddings.
        extractor: Extractor instance for extracting metadata and text from documents.
            Extractor is an interface that has 1 method which returns a list of dicts.
            This list contains the text and metadata for each chunk of the document.
            Embeddings for 'text' are generated using the embedder.
        vector_db: VectorStorage instance for storing and retrieving embeddings.
    """
    
    def __init__(self, config_source: str|Dict[str, any]):
        """
        Initialize the crawler with configuration.
        
        Args:
            config_source: source for configuration, either a path to a YAML file or a dictionary.
            config_manager: Optional ConfigManager instance. If not provided, a new one will be created.
        """
        # Set up config manager if not provided
        self.config_manager = ConfigManager(config_source=config_source)
        self.config = self.config_manager.config
        
        # Set up components based on configuration
        self.embedder = LocalEmbedder(self.config.get("embeddings", {}))    
        self.vector_db = self._setup_vector_db()

        self.llm = None
        self.extractor = None
        self.directory_name = self.config.get("path", None)


    def _setup_vector_db(self) -> VectorStorage:
        """Set up the vector database using configuration."""
        # Initialize vector storage with Milvus connection parameters
        return VectorStorage(self.config)

    def set_llm(self, llm=None) -> Any:
        """Set up the LLM using configuration."""
        # If no LLM is provided, use the one from the config
        if llm is None:
            llm_config = self.config.get("llm", {})
            llm = init_chat_model(
                model=llm_config.get("model"), 
                model_provider=llm_config.get("provider"), 
                base_url=llm_config.get("base_url")
            )

        self.llm = llm

        if llm is None:
            raise ValueError("No LLM model provided in config or as argument.")
        return llm
    
    def set_extractor(self, extractor=None):
        """Set up the extractor using configuration."""
        # If no extractor is provided, use the one from the config
        if extractor is None:
            extractor_config = self.config.get("extractor", {})
            extractor = Extractor(self.llm, extractor_config)
        
        self.extractor = extractor

        if extractor is None:
            extractor_config = self.config.get("extractor", {})
            extractor = Extractor(self.llm, extractor_config)
        
        self.extractor = extractor

        if extractor is None:
            raise ValueError("No extractor provided in config or as argument.")

    def _setup_filepaths(self, directory_path) -> list[str]:
        # Get list of files
        # check if the directory path is a directory or if it is a file
        if not os.path.isdir(directory_path):
            print(f"Path {directory_path} is not a directory, skipping")
            if os.path.isfile(directory_path):
                print(f"Path {directory_path} is a file, adding to list")
                return [directory_path]
            else:
                print(f"Path {directory_path} is not a file, skipping")
                return []
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    def run(self, files: Optional[str|list[str]] = None) -> Generator[list[Dict[str, Any]]]:
        """
        Run the crawler on the configured directory.
        
        Yields:
            list of the data for a whole file split into chunks:
                - Dict of (text, embedding, **metadata) from the document processor.
        """
        if self.llm is None:
            self.set_llm()
        if self.extractor is None:
            self.set_extractor()
        # If no files are provided, use the directory from the config
        if files is None:
            files = self.directory_name
        # If a single file is provided, convert it to a list
        if isinstance(files, str):
            filepaths = [files]
        else:
            filepaths = self._setup_filepaths(files)
        
        print(f"Processing files: {len(filepaths)} starting with {filepaths[0]}")
        
        # Process the directory and yield results
        # for filepath in ["/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv/2408.12236v1.pdf"]:
        for filepath in filepaths:
            print(f"Processing file: {filepath}")
            try:
                chunk_dicts = self.extractor.extract(filepath)
                print("yielding:", chunk_dicts[0])
                for chunk_dict in chunk_dicts:
                    chunk_dict["embedding"] = self.embedder.embed_query(chunk_dict['text'])
                yield chunk_dicts # yields all the chunks for a single file
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                continue
  