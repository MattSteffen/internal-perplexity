import json
import os
import argparse
from typing import Dict, Any, Iterator, Tuple
# from discovery import find_dirs
from processing.embeddings import LocalEmbedder
from processing.processor import DocumentProcessor
from storage.vector_db import VectorStorage
from config.config_manager import ConfigManager
from langchain.chat_models import init_chat_model


class Crawler:
    """Main class for crawling and processing directories of documents."""
    
    def __init__(self, directory_name: str = None, config_manager: ConfigManager = None):
        """
        Initialize the crawler with configuration.
        
        Args:
            directory_name: Name of the directory configuration to use.
            config_manager: Optional ConfigManager instance. If not provided, a new one will be created.
        """
        # Set up config manager if not provided
        if config_manager is None:
            self.config_manager = ConfigManager(
                base_config_path='config/base_config.yaml',
                collection_template_path='config/collection_template.yaml',
                collections_config_dir='config/directories'
            )
        else:
            self.config_manager = config_manager
            
        self.directory_name = directory_name
        
        if directory_name:
            # Load configuration for the specified directory
            self.config = self.config_manager.get_config(directory_name)
            
            # Set up components based on configuration
            self.llm = self._setup_llm()
            self.vision_model = self._setup_vision_model()
            self.embedder = self._setup_embedder()
            
            # Create document processor
            self.processor = DocumentProcessor(
                self.config,
                self.llm,
                self.vision_model,
                self.embedder
            )
        else:
            self.config = None
            self.processor = None
    
    def _setup_vector_db(self) -> VectorStorage:
        """Set up the vector database using configuration."""
        # Initialize vector storage with Milvus connection parameters
        return VectorStorage(self.config)

    def _setup_embedder(self) -> LocalEmbedder:
        """Set up the embedder using configuration."""
        # Initialize embedder with configuration
        return LocalEmbedder(self.config)

    def _setup_llm(self) -> Any:
        """Set up the LLM using configuration."""
        llm_config = self.config.get('llm', {})
        # Initialize LLM with configuration
        llm = init_chat_model(
            model=llm_config.get("model"), 
            model_provider=llm_config.get("provider"), 
            base_url=llm_config.get("base_url")
        )
        return llm

    def _setup_vision_model(self) -> Any:
        """Initialize a vision model for image analysis."""
        vision_llm_config = self.config.get('vision_llm', {})
        source = vision_llm_config.get('source', 'ollama')
        model_name = vision_llm_config.get('model_name', 'gemma3')
        base_url = vision_llm_config.get('base_url', 'http://localhost:11434/')
        
        try:
            if source == "ollama":
                return init_chat_model(
                    model_name,
                    model_provider="ollama",
                    base_url=base_url,
                    vision=True
                )
            else:
                print(f"Vision model source {source} not supported")
                return None
        except Exception as e:
            print(f"Failed to initialize vision model: {e}")
            return None
    
    def _setup_filepaths(self, directory_path) -> list[str]:
        # Get list of files
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    def run(self) -> Iterator[Tuple[str, Dict[str, Any], Any]]:
        """
        Run the crawler on the configured directory.
        
        Yields:
            Tuples of (text, metadata, embedding) from the document processor.
        """
        if not self.directory_name or not self.config:
            raise ValueError("No directory configuration specified. Initialize with a directory name first.")
        
        # Get directory path from config
        dir_path = self.config.get('path')
        if not dir_path:
            print(f"No path specified for directory {self.directory_name}, skipping")
            return
        
        filepaths = self._setup_filepaths(dir_path)
        
        print(f"Processing directory: {self.directory_name} at path {dir_path}")
        
        # Process the directory and yield results
        for filepath in filepaths:
            print(f"Processing file: {filepath}")
            yield self.processor.process_document(filepath)
    
    def set_directory(self, directory_name: str):
        """
        Change the directory configuration to use.
        
        Args:
            directory_name: Name of the directory configuration to use.
        """
        self.directory_name = directory_name
        self.config = self.config_manager.get_config(directory_name)
        
        # Re-initialize components with new configuration
        self.llm = self._setup_llm()
        self.vision_model = self._setup_vision_model()
        self.embedder = self._setup_embedder()
        
        # Create new document processor
        self.processor = DocumentProcessor(
            self.config,
            self.llm,
            self.vision_model,
            self.embedder
        )


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--directory', '-d', type=str, help='Directory configuration to use')
    args = parser.parse_args()
    
    # Create and run crawler
    crawler = Crawler(args.directory)
    
    # Process all documents
    # for text, metadata, embedding in crawler.run():
    a,b,c = [], [], []
    for x,y,z in crawler.run():
        a.append(x)
        b.append(z)
        c.append(y)

    with crawler._setup_vector_db() as db:
        db.insert_data(a,b,c)
    print("complete")

if __name__ == "__main__":
    main()