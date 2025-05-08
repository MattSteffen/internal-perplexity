from openai import timeout
import yaml
import os
from typing import Dict, Any, Iterator, Tuple, Generator, Optional, Union, List
from processing.embeddings import LocalEmbedder
from processing.extractor import Extractor
from storage.vector_db import VectorStorage
from langchain.chat_models import init_chat_model


"""
TODO:
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
        self.config = self.load_config(config_source)
        
        # Set up components based on configuration
        self.embedder = LocalEmbedder(self.config.get("embeddings", {}))    

        self.llm = None
        self.extractor = None
        self.directory_name = self.config.get("path", None)
    
    def load_config(self, config_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file or return the provided dict.
        
        Args:
            config_source: Either a filepath string to a YAML file or a dictionary
            
        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary
            
        Raises:
            FileNotFoundError: If the provided filepath doesn't exist
            ValueError: If the filepath isn't a YAML file or can't be loaded
        """
        # If config_source is already a dictionary, return it directly
        if isinstance(config_source, dict):
            self.config = config_source
            return config_source
        
        # If config_source is a string, treat it as a filepath
        elif isinstance(config_source, str):
            # Check if file exists
            if not os.path.isfile(config_source):
                raise FileNotFoundError(f"Config file not found: {config_source}")
            
            # Check if file has yaml extension
            file_ext = os.path.splitext(config_source)[1].lower()
            if file_ext not in ['.yaml', '.yml']:
                raise ValueError(f"File is not a YAML file: {config_source}")
            
            # Try to load the YAML file
            try:
                with open(config_source, 'r') as file:
                    config = yaml.safe_load(file)
                    
                # Ensure loaded content is a dictionary
                if not isinstance(config, dict):
                    raise ValueError(f"YAML file does not contain a dictionary: {config_source}")
                
                self.config = config
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {config_source}: {str(e)}")
        
        # If config_source is neither a dict nor a string
        else:
            raise TypeError(f"Expected dict or filepath string, got {type(config_source).__name__}")

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
                base_url=llm_config.get("base_url"),
                timeout=llm_config.get("timeout", 60),
                num_ctx=llm_config.get("num_ctx", 32000),
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

    def get_filepaths(self, path: str) -> List[str]:
        """
        Returns a list of file paths based on the input path.
        
        Args:
            path: Either a file path or a directory path
            
        Returns:
            List[str]: 
                - If path is a file: a list containing only that file path
                - If path is a directory: a list of all file paths in that directory (including subdirectories)
                - If path is neither: an empty list
        """
        # Check if the path exists
        if not os.path.exists(path):
            print(f"Path {path} does not exist, returning empty list")
            return []
        
        # If path is a file, return a list with just that file
        if os.path.isfile(path):
            print(f"Path {path} is a file, adding to list")
            return [path]
        
        # If path is a directory, collect all files within it
        elif os.path.isdir(path):
            print(f"Path {path} is a directory, collecting all files")
            file_paths = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
            return file_paths
        
        # If path is neither a file nor a directory
        else:
            print(f"Path {path} is neither a file nor a directory, returning empty list")
            return []

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
            files = self.get_filepaths(files)
        
        print(f"Processing files: {len(files)} starting with {files[0]}")
        
        # Process the directory and yield results
        for filepath in files:
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
  