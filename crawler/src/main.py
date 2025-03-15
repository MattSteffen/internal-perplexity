import json
import os
import argparse
from typing import Dict, Any
# from discovery import find_dirs
from crawler.src.processing.extractor import JSONHandler
from processing.embeddings import LocalEmbedder
from processing.processor import DocumentProcessor
from storage.vector_db import VectorStorage
from config.config_manager import ConfigManager
from langchain.chat_models import init_chat_model


"""
What this file should do:

it is the main entry point for the crawler.

- it should take a directory name as an argument and whether to connect to the vector_db
- it should load the configuration for that directory
- it should load the configuration for the each component (crawler, vector db, embedder)
- crawler should crawl the directory
  - create processor 
  - call process.directory
  - processor yields [text, metadata, embedding]
"""


def setup_vector_db(config: Dict[str, Any]) -> VectorStorage:
    """Set up the vector database using configuration."""
    milvus_config = config.get('milvus', {})
    collection_config = config.get('collection', {})
    
    # Initialize vector storage with Milvus connection parameters
    vector_db = VectorStorage(
        host=milvus_config.get('host', 'localhost'),
        port=milvus_config.get('port', 19530),
        user=milvus_config.get('user', ''),
        password=milvus_config.get('password', ''),
        secure=milvus_config.get('secure', False),
        collection_name=collection_config.get('name', 'default_collection')
    )
    
    return vector_db

def setup_embedder(config: Dict[str, Any]) -> LocalEmbedder:
    """Set up the embedder using configuration."""
    embedding_config = config.get('embeddings', {})
    
    # Initialize embedder with configuration
    embedder = LocalEmbedder(
        model=embedding_config.get('model', 'all-MiniLM-L6-v2'),
        dimension=embedding_config.get('dimension', 384),
    )
    
    return embedder

def setup_llm(config: Dict[str, Any]) -> Any:
    """Set up the LLM using configuration."""
    llm_config = config.get('llm', {})

    # Initialize LLM with configuration
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq") # Must support structured output


    return llm

def setup_vision_model(config: Dict[str, Any]) -> Any:

    """Initialize a vision model for image analysis.
    
    Args:
        model_name: Name of the vision model to use
        base_url: Base URL for API calls
        source: Model provider ("ollama" or other supported providers)
        
    Returns:
        Initialized vision model or None if initialization fails
    """
    vision_llm_config = config.get('vision_llm', {})
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

def run_crawler(directory_name: str = None):
    """
    Run the crawler with the specified directory configuration.
    
    Args:
        directory_name: Name of the directory configuration to use.
    """
    # Load configuration
    config_manager = ConfigManager()
    
    if directory_name is None:
        # If no directory specified, process all configured directories
        directory_names = config_manager.get_all_directory_names()
    else:
        directory_names = [directory_name]
    
    for dir_name in directory_names:
        config = config_manager.get_config_for_directory(dir_name)
        
        # Set up components with configuration
        if config.get('vector_db', {}):
            use_vector_db = True
            vector_db = setup_vector_db(config)
        else:
            use_vector_db = False

        embedder = setup_embedder(config)
        
        # Get directory path from config
        dir_path = config.get('path')
        if not dir_path:
            print(f"No path specified for directory {dir_name}, skipping")
            continue
        
        print(f"Processing directory: {dir_name} at path {dir_path}")
        
        # Process the directory
        processor = DocumentProcessor(
            dir_path,
            vector_db,
            embedder,
            config
        )
        processor.process_directory(dir_path)
        
        if use_vector_db:
            vector_db.close()

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--directory', '-d', type=str, help='Directory configuration to use')
    # TODO: add argument for config files to include so it doesn't have to look in the config directories.
    args = parser.parse_args()
    
    run_crawler(args.directory)

if __name__ == "__main__":
    main()
