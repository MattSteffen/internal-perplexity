import os
import uuid
from typing import Dict, Any, Union, List
import json
import hashlib
from pathlib import Path

from processing.embeddings import LocalEmbedder
from processing.extractor import Extractor, BasicExtractor
from processing.llm import LLM
from processing.converter import Converter, MarkItDownConverter
from storage.milvus import MilvusStorage


"""
Crawler takes a directory or list of filepaths, extracts the markdown and metadata from each, chunks them, and stores them in a vector database.
The crawler class is a base class.

# TODO: Invlidate metadata that contains any of the following:
    - "document_id"
    - "chunk_index"
    - "source"
    - "text"
    - "text_embedding"
    - "text_sparse_embedding"
    - "metadata"
    - "metadata_sparse_embedding"

# TODO: Make a backend server for OI and have radchat model available.

Example config:
{
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model": "gemma3",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "milvus": {
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "123456",
        "collection": "test_collection",
        "partition": "test_partition",
        "recreate": False,
        "collection_description": "descriptions of the collection",
    },
    "llm": {
        "model": "gemma3",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "utils": {
        "chunk_size": 1000,
    }
}
"""
class Crawler:
    def __init__(
                self,
                config: Dict[str, Any],
                metadata_schema: Dict[str, Any],
                converter: Converter = None,
                extractor: Extractor = None,
                vector_db: MilvusStorage = None,
                embedder: LocalEmbedder = None,
                llm: LLM = None,
            ) -> None:
        self.config = config
        self.metadata_schema = metadata_schema
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db

        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        # initialize defaults if provided as none
        if self.llm is None:
            if self.config.get("llm") is None:
                raise ValueError("LLM config is required")
            self.llm = LLM(
                model_name=self.config.get("llm", {}).get("model"),
                base_url=self.config.get("llm", {}).get("base_url")
            )
        
        if self.embedder is None:
            if self.config.get("embeddings") is None:
                raise ValueError("Embeddings config is required")
            self.embedder = LocalEmbedder(self.config.get("embeddings", {}))
        
        if self.extractor is None:
            if self.llm is None:
                raise ValueError("LLM is required prior to creating extractor")
            self.extractor = BasicExtractor(self.metadata_schema, self.llm)
        
        if self.vector_db is None:
            if self.config.get("milvus") is None:
                raise ValueError("Milvus config is required")
            self.vector_db = MilvusStorage(
                self.config.get("milvus", {}),
                dimension=self.embedder.dimension,
                metadata_schema=self.metadata_schema,
            )
        
        if self.converter is None:
            if self.config.get("vision_llm") is None:
                raise ValueError("Vision LLM config is required")
            self.converter = MarkItDownConverter(self.config)
    
    def _get_filepaths(self, path: Union[str, list[str]]) -> List[str]:
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
        if isinstance(path, list):
            return path
        
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

    def crawl(self, path: Union[str, List[str]]) -> None:
        """
        Crawl the given path(s) and process the documents.
        """
        filepaths = self._get_filepaths(path)
        temp_dir_str = self.config.get("utils", {}).get("temp_dir")
        temp_dir = Path(temp_dir_str) if temp_dir_str else None

        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        for filepath in filepaths:
            if self.vector_db._check_duplicate(filepath, 0):
                print(f"Document {filepath} already in Milvus. Skipping.")
                continue

            markdown = None
            metadata = None
            temp_filepath = None

            if temp_dir:
                filename = os.path.basename(filepath) + '.json'
                temp_filepath = temp_dir / filename
                if temp_filepath.exists():
                    print(f"Loading processed document from {temp_filepath}")
                    with open(temp_filepath, 'r') as f:
                        data = json.load(f)
                        markdown = data['text']
                        metadata = data['metadata']

            if markdown is None or metadata is None:
                # Convert to markdown and extract metadata and chunks
                markdown = self.converter.convert(filepath)
                metadata = self.extractor.extract_metadata(markdown)
                if temp_filepath:
                    print(f"Saving processed document to {temp_filepath}")
                    with open(temp_filepath, 'w') as f:
                        json.dump({'text': markdown, 'metadata': metadata}, f)

            chunks = self.extractor.chunk_text(markdown, self.config.get("utils", {}).get("chunk_size", 1000))

            entities = []
            for i, chunk in enumerate(chunks):
                embeddings = self.embedder.embed(chunk)

                # Create entities for milvus
                entity = {
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "source": filepath,
                    "text": chunk,
                    "text_embedding": embeddings,
                    **metadata,
                }

                entities.append(entity)

            # Save the document to the vector database
            self.vector_db.insert_data(entities)
