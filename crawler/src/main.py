import json
import os
import argparse
from typing import Dict, Any, Iterator, Tuple, Generator
# from discovery import find_dirs
from processing.embeddings import LocalEmbedder
from processing.extractor import Extractor
from storage.vector_db import VectorStorage
from config.config_manager import ConfigManager
from langchain.chat_models import init_chat_model


class Crawler:
    """Main class for crawling and processing directories of documents."""
    
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
        self.directory_name = self.config["path"]
        
        # Set up components based on configuration
        self.llm = self._setup_llm()
        self.embedder = LocalEmbedder(self.config.get("embeddings", {}))
        self.extractor = Extractor(self.llm, self.config)
    
    def _setup_vector_db(self) -> VectorStorage:
        """Set up the vector database using configuration."""
        # Initialize vector storage with Milvus connection parameters
        return VectorStorage(self.config)

    def _setup_llm(self) -> Any:
        """Set up the LLM using configuration."""
        llm_config = self.config.get('llm', {})
        # Initialize LLM with configuration
        print("LLM Config", llm_config)
        llm = init_chat_model(
            model=llm_config.get("model"), 
            model_provider=llm_config.get("provider"), 
            base_url=llm_config.get("base_url")
        )
        return llm

    def _setup_filepaths(self, directory_path) -> list[str]:
        # Get list of files
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    def run(self) -> Generator[list[Dict[str, Any]]]:
        """
        Run the crawler on the configured directory.
        
        Yields:
            list of the data for a whole file split into chunks:
                - Dict of (text, embedding, **metadata) from the document processor.
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
    

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--config', '-c', type=str, help='Path to your configuration file')
    parser.add_argument('--directory', '-d', type=str, help='Directory of files to import')
    args = parser.parse_args()
    
    # Load config file if provided
    config_dict = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Override path with directory arg if provided
    if args.directory:
        config_dict["path"] = args.directory
    
    # Validate we have a path
    if "path" not in config_dict:
        parser.error("Must specify directory path either via --directory arg or in config file")
    
    # Create and run crawler
    crawler = Crawler(config_dict)
    
    # Process all documents
    with crawler._setup_vector_db() as db:
        for data in crawler.run():
            db.insert_data(data)

    print("complete")

if __name__ == "__main__":
    main()