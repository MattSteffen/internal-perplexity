from abc import ABC, abstractmethod
from curses import meta
from pydoc import text
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import os
import json
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum

from typing import Protocol, Any, List, Dict, runtime_checkable
from abc import abstractmethod

from src.processing.embeddings import EmbedderConfig


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


from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Data for a single query in a benchmark run."""

    query: str
    expected_source: str
    placement_order: Optional[int] = None
    distance: Optional[float] = None
    time_to_search: float = 0.0
    found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        return {
            "query": self.query,
            "expected_source": self.expected_source,
            "placement_order": self.placement_order,
            "distance": self.distance,
            "time_to_search": self.time_to_search,
            "found": self.found,
        }


@dataclass
class BenchmarkRunResults:
    """Aggregated results from a benchmark run."""

    results_by_doc: Dict[str, List[BenchmarkResult]]
    placement_distribution: Dict[int, int]
    distance_distribution: List[float]
    percent_in_top_k: Dict[int, float]
    search_time_distribution: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a JSON-serializable dictionary."""
        # Convert integer keys in distributions to strings for JSON compatibility
        return {
            "results_by_doc": {
                source: [result.to_dict() for result in results]
                for source, results in self.results_by_doc.items()
            },
            "placement_distribution": {
                str(k): v for k, v in self.placement_distribution.items()
            },
            "distance_distribution": self.distance_distribution,
            "percent_in_top_k": {str(k): v for k, v in self.percent_in_top_k.items()},
            "search_time_distribution": self.search_time_distribution,
        }


class DatabaseBenchmark(ABC):
    """
    Abstract base class for database benchmark clients.
    This interface defines the contract that all database benchmark implementations
    must follow to be compatible with the document processing pipeline.
    """

    @abstractmethod
    def __init__(
        self, db_config: "DatabaseClientConfig", embed_config: "EmbedderConfig"
    ) -> None:
        """
        Initialize the database benchmark client.
        Args:
            db_config: Configuration for the database client.
            embed_config: Configuration for the embedding provider.
        """
        pass

    @abstractmethod
    def search(
        self, queries: List[str], filters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a search against the database.
        Args:
            queries: A list of search queries.
            filters: An optional list of filters to apply to the search.
        Returns:
            A list of search results.
        """
        pass

    @abstractmethod
    def run_benchmark(self, generate_queries: bool = False) -> BenchmarkRunResults:
        """
        Run a benchmark for a set of documents and queries.
        Args:
            queries_by_doc: A dictionary where keys are document source identifiers
                            and values are lists of queries to run for that document.
            top_k_values: A list of integers for calculating 'percent in top k'.
                          Defaults to [1, 5, 10, 20, 50, 100].
        Returns:
            A BenchmarkRunResults object containing detailed and aggregated results.
        """
        pass

    def plot_results(self, results: BenchmarkRunResults, output_dir: str) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Placement distribution
        plt.figure()
        placements = sorted(results.placement_distribution.keys())
        counts = [results.placement_distribution[p] for p in placements]
        plt.bar(placements, counts)
        plt.xlabel("Placement Order")
        plt.ylabel("Frequency")
        plt.title("Frequency by Placement Order")
        plt.savefig(os.path.join(output_dir, "placement_distribution.png"))
        plt.close()

        # Distance distribution
        plt.figure()
        plt.hist(results.distance_distribution, bins=20)
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title("Frequency by Distance")
        plt.savefig(os.path.join(output_dir, "distance_distribution.png"))
        plt.close()

        # Percent in top k
        plt.figure()
        k_values = sorted(results.percent_in_top_k.keys())
        percentages = [results.percent_in_top_k[k] for k in k_values]
        plt.plot(k_values, percentages, marker="o")
        plt.xlabel("k")
        plt.ylabel("Percent in Top k")
        plt.title("Percent in Top k by k")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "percent_in_top_k.png"))
        plt.close()

        # Search time distribution
        plt.figure()
        plt.hist(results.search_time_distribution, bins=20)
        plt.xlabel("Time to Search (s)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Time to Search")
        plt.savefig(os.path.join(output_dir, "search_time_distribution.png"))
        plt.close()

    def save_results(self, results: BenchmarkRunResults, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(results.to_dict(), f, indent=4)
