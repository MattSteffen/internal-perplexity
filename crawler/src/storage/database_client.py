from abc import ABC, abstractmethod
from curses import meta
from pydoc import text
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum

from typing import Protocol, Any, List, Dict, runtime_checkable
from abc import abstractmethod


@dataclass
class DatabaseDocument:
    """
    Protocol defining the minimum interface for document data.

    Any object that has these attributes/methods can be used as document data.
    This includes regular dicts, custom classes, dataclasses, etc.
    """

    # Required attributes - these must exist
    text: str
    text_embedding: List[float]
    chunk_index: int
    source: str

    metadata: Dict[str, any] = field(default_factory=dict)

    # Required methods for dict-like access
    def __getitem__(self, key: str) -> Any:
        match key:
            case "text":
                return self.text
            case "text_embedding":
                return self.text_embedding
            case "chunk_index":
                return self.chunk_index
            case "source":
                return self.source
            case _:
                return self.metadata.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        val = self.__getitem__(key)
        if val is not None:
            return val
        else:
            return default

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "DatabaseDocument":
        text = data.get("text")
        text_embedding = data.get("text_embedding")
        chunk_index = data.get("chunk_index")
        source = data.get("source")
        metadata = {
            k: v
            for k, v in data.items()
            if k not in ["text", "text_embedding", "chunk_index", "source"]
        }
        if any(
            [text is None, text_embedding is None, chunk_index is None, source is None]
        ):
            raise ValueError("missing data")
        return cls(
            text=text,
            text_embedding=text_embedding,
            chunk_index=chunk_index,
            source=source,
            metadata=metadata,
        )


@dataclass
class DatabaseClientConfig:
    """Base configuration for database clients."""

    provider: str
    collection: str
    partition: Optional[str] = None
    recreate: bool = False
    collection_description: Optional[str] = None

    host: str = "localhost"
    port: int = 19530
    username: str = "root"
    password: str = "Milvus"

    @property
    def uri(self) -> str:
        """Get the connection URI."""
        return f"http://{self.host}:{self.port}"

    @property
    def token(self) -> str:
        """Get the authentication token."""
        return f"{self.username}:{self.password}"

    @classmethod
    def from_dict(cls, config: Dict[str, any]):
        return cls(
            provider=config.get("provider"),
            host=config.get("host"),
            port=config.get("port"),
            username=config.get("username"),
            password=config.get("password"),
            collection=config.get("collection"),
            partition=config.get("partition"),
            recreate=config.get("recreate"),
            collection_description=config.get("collection_description"),
        )


class DatabaseClient(ABC):
    """
    Abstract base class for vector database clients.

    This interface defines the contract that all database implementations
    must follow to be compatible with the document processing pipeline.
    """

    @abstractmethod
    def __init__(
        self,
        config: DatabaseClientConfig,
        embedding_dimension: int,
        metadata_schema: Dict[str, Any],
    ):
        """
        Initialize the database client.

        Args:
            config: Database-specific configuration parameters
            embedding_dimension: Vector embedding dimensionality
            metadata_schema: JSON schema defining user metadata fields
        """
        pass

    @abstractmethod
    def create_collection(self, recreate=False) -> None:
        """
        Create a collection/table with the specified schema.
        Handles the logic for checking if collection exists and recreating if needed.
        This can be a prefix to a bucket in s3. Some differenciator between groups of documents.

        Must have set:
            embedding_dimension: Dimension of vector embeddings
            metadata_schema: Metadata schema definition

        Raises:
            DatabaseError: If collection creation fails
        """
        pass

    @abstractmethod
    def insert_data(self, data: List[DatabaseDocument]) -> None:
        """
        Insert data into the collection with duplicate detection.

        Expected data format:
        [
            {
                "text": "content text",
                "text_embedding": [0.1, 0.2, ...],
                "chunk_index": 0,
                "source": "filename",
                "document_id": "uuid",
                "minio": "optional_url",
                # ... other user-defined fields
            },
            ...
        ]

        Args:
            data: List of document chunks to insert

        Raises:
            DatabaseError: If insertion fails
        """
        pass

    @abstractmethod
    def check_duplicate(self, source: str, chunk_index: int) -> bool:
        """
        Check if a document chunk already exists.

        Args:
            source: Source identifier (file path)
            chunk_index: Chunk index within the document

        Returns:
            bool: True if duplicate exists, False otherwise
        """
        pass
