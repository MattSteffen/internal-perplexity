from dataclasses import dataclass, field
import os
import uuid
from typing import Dict, Union, List
import json
from pathlib import Path

from src.processing import (
    Embedder,
    EmbedderConfig,
    get_embedder,
    Extractor,
    BasicExtractor,
    LLM,
    LLMConfig,
    get_llm,
    Converter,
    MarkItDownConverter,
)
from src.storage import (
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    get_db,
)


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
    "database": {
        "provider": "milvus",
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
        "temp_dir": "tmp/"
    },
    "extractor": {},
    "converter": {}
}
"""


@dataclass
class CrawlerConfig:
    embeddings: EmbedderConfig
    llm: LLMConfig
    vision_llm: LLMConfig
    database: DatabaseClientConfig
    extractor: Dict[str, any] = field(default_factory=dict)
    converter: Dict[str, any] = field(default_factory=dict)
    chunk_size: int = 10000  # treated as maximum if using semantic chunking
    metadata_schema: Dict[str, any] = field(default_factory=dict)
    temp_dir: str = "tmp/"

    @classmethod
    def from_dict(cls, config: Dict[str, any]):
        return cls(
            embeddings=EmbedderConfig.from_dict(config.get("embeddings", {})),
            llm=LLMConfig.from_dict(config.get("llm", {})),
            vision_llm=LLMConfig.from_dict(config.get("vision_llm", {})),
            extractor=config.get("extractor"),
            converter=config.get("converter"),
            database=DatabaseClientConfig.from_dict(config.get("database", {})),
            metadata_schema=config.get("metadata_schema", {}),
            chunk_size=config.get("utils", {}).get("chunk_size", 10000),
            temp_dir=config.get("utils", {}).get("temp_dir", "tmp/"),
        )


class Crawler:
    def __init__(
        self,
        config: CrawlerConfig,
        converter: Converter = None,
        extractor: Extractor = None,
        vector_db: DatabaseClient = None,
        embedder: Embedder = None,
        llm: LLM = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.converter = converter
        self.extractor = extractor
        self.vector_db = vector_db

        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        # initialize defaults if provided as none
        if self.llm is None:
            self.llm = get_llm(self.config.llm)

        if self.embedder is None:
            self.embedder = get_embedder(self.config.embeddings)

        if self.extractor is None:
            # self.extractor = get_extractor(self.config.extractor, self.config.metadata_schema, self.llm)
            self.extractor = BasicExtractor(self.config.metadata_schema, self.llm)

        if self.vector_db is None:
            self.vector_db = get_db(
                self.config.database,
                self.embedder.get_dimension(),
                self.config.metadata_schema,
            )

        if self.converter is None:
            # self.converter = get_converter(self.config.converter)
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
            print(
                f"Path {path} is neither a file nor a directory, returning empty list"
            )
            return []

    def crawl(self, path: Union[str, List[str]]) -> None:
        """
        Crawl the given path(s) and process the documents.
        """
        filepaths = self._get_filepaths(path)
        temp_dir = Path(self.config.temp_dir)

        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)

        for i, filepath in enumerate(filepaths):
            if self.vector_db.check_duplicate(filepath, 0):
                print(f"Document {filepath} already in Milvus. Skipping.")
                continue

            markdown = None
            metadata = None
            temp_filepath = None

            if temp_dir:
                filename = os.path.splitext(os.path.basename(filepath))[0] + ".json"
                temp_filepath = temp_dir / filename
                if temp_filepath.exists():
                    print(f"Loading processed document from {temp_filepath}")
                    with open(temp_filepath, "r") as f:
                        data = json.load(f)
                        markdown = data["text"]
                        metadata = data["metadata"]

            if markdown is None or metadata is None:
                # Convert to markdown and extract metadata and chunks
                markdown = self.converter.convert(filepath)
                metadata = self.extractor.extract_metadata(markdown)
                if temp_filepath:
                    with open(temp_filepath, "w") as f:
                        json.dump({"text": markdown, "metadata": metadata}, f)

            chunks = self.extractor.chunk_text(
                markdown, self.config.chunk_size
            )  # TODO: remove chunk_size, make that part of the init config.

            entities: List[DatabaseDocument] = []
            for i, chunk in enumerate(chunks):
                embeddings = self.embedder.embed(chunk)

                # Create entities for database
                try:
                    doc = DatabaseDocument.from_dict(
                        {
                            "document_id": str(uuid.uuid4()),
                            "chunk_index": i,
                            "source": filepath,
                            "text": chunk,
                            "text_embedding": embeddings,
                            **metadata,
                        }
                    )

                    entities.append(doc)
                except ValueError:
                    continue

            # Save the document to the vector database
            self.vector_db.insert_data(entities)
