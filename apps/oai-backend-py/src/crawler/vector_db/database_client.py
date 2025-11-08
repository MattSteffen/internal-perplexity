import json
import os
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from ..llm.embeddings import EmbedderConfig


class DatabaseDocument(BaseModel):
    """
    Pydantic model defining the interface for document data.

    All system fields use default_ prefix to avoid conflicts with user metadata.
    This model provides automatic validation and serialization for document chunks.

    Attributes:
        id: Database-assigned primary key (default: -1 before insertion)
        metadata: Additional user-defined metadata as key-value pairs
        security_group: List of security groups for RBAC access control
        default_document_id: Unique identifier for the document chunk (UUID)
        default_text: The text content of the document chunk
        default_text_embedding: Dense vector embedding of the text
        default_chunk_index: Index of this chunk within the document (0-based)
        default_source: Source identifier (e.g., file path, URL)
        default_metadata: JSON string representation of metadata
        default_minio: URL to the original document in MinIO storage
        default_text_sparse_embedding: Sparse embedding for BM25 full-text search
        default_metadata_sparse_embedding: Sparse embedding of metadata for BM25 search
        default_benchmark_questions: List of benchmark questions for testing
    """

    # Required attributes - these must exist (using prefixed field names)
    default_document_id: str = Field(..., description="Unique identifier for the document chunk (UUID format)")
    default_text: str = Field(..., description="The text content of the document chunk")
    default_text_embedding: list[float] = Field(..., description="Dense vector embedding of the text content")
    default_chunk_index: int = Field(..., ge=0, description="Zero-based index of this chunk within the parent document")
    default_source: str = Field(..., description="Source identifier (file path, URL, etc.)")
    security_group: list[str] = Field(default_factory=lambda: ["public"], description="List of security groups for RBAC row-level access control")
    metadata: dict[str, Any] = Field(default_factory=dict, description="User-defined metadata as key-value pairs")

    # Optional attributes with defaults
    id: int | None = Field(default=-1, description="Database-assigned primary key (auto-generated on insert)")
    default_metadata: str | None = Field(default="", description="JSON string representation of the metadata field")
    default_minio: str | None = Field(default="", description="URL to the original document in MinIO object storage")
    default_text_sparse_embedding: list[float] | None = Field(default_factory=list, description="Sparse vector embedding for BM25 full-text search on text")
    default_metadata_sparse_embedding: list[float] | None = Field(default_factory=list, description="Sparse vector embedding for BM25 full-text search on metadata")
    default_benchmark_questions: list[str] | None = Field(default_factory=list, description="List of benchmark questions generated for testing retrieval")

    model_config = {
        "extra": "forbid",  # Prevent extra fields to maintain schema integrity
        "validate_assignment": True,  # Validate when fields are assigned after initialization
    }

    # Dict-like access methods for backward compatibility
    def __getitem__(self, key: str) -> Any:
        """
        Enable dict-like access to fields.
        Falls back to metadata for unknown keys.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return self.metadata.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a field value with a default fallback.

        Args:
            key: Field name to retrieve
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        try:
            val = self.__getitem__(key)
            return val if val is not None else default
        except Exception:
            return default

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.
        Uses Pydantic's model_dump for serialization.

        Returns:
            Dictionary representation of the document
        """
        return self.model_dump()  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseDocument:
        """
        Create a DatabaseDocument from a dictionary.
        Uses Pydantic's validation for type checking.

        Args:
            data: Dictionary containing document data

        Returns:
            DatabaseDocument instance

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def to_string(self) -> str:
        """
        Convert to formatted JSON string.
        Useful for LLM input or debugging.

        Returns:
            Pretty-printed JSON string
        """
        return self.model_dump_json(indent=4)  # type: ignore[no-any-return]


class DatabaseClientConfig(BaseModel):
    """
    Base configuration for database clients.

    This model provides type-safe configuration for connecting to vector databases.
    All connection parameters are validated at creation time.

    Attributes:
        provider: Database provider name (e.g., "milvus")
        collection: Name of the database collection/table
        partition: Optional partition name within the collection
        recreate: Whether to drop and recreate the collection if it exists
        collection_description: Optional description of the collection
        host: Database host address
        port: Database port number (1-65535)
        username: Authentication username
        password: Authentication password
    """

    provider: str = Field(..., min_length=1, description="Database provider name (e.g., 'milvus')")
    collection: str = Field(..., min_length=1, description="Name of the database collection or table")
    partition: str | None = Field(default=None, description="Optional partition name for data organization within the collection")
    recreate: bool = Field(default=False, description="If True, drop and recreate the collection if it already exists")
    collection_description: str | None = Field(default=None, description="Optional human-readable description of the collection")
    host: str = Field(default="localhost", description="Database server hostname or IP address")
    port: int = Field(default=19530, ge=1, le=65535, description="Database server port number (must be between 1 and 65535)")
    username: str = Field(default="root", description="Database authentication username")
    password: str = Field(default="Milvus", description="Database authentication password")

    model_config = {
        "validate_assignment": True,  # Validate when fields are assigned after initialization
    }

    @property
    def uri(self) -> str:
        """
        Get the connection URI for the database.

        Returns:
            Connection URI in format http://host:port
        """
        return f"http://{self.host}:{self.port}"

    @property
    def token(self) -> str:
        """
        Get the authentication token.

        Returns:
            Authentication token in format username:password
        """
        return f"{self.username}:{self.password}"

    @classmethod
    def milvus(
        cls,
        collection: str,
        host: str = "localhost",
        port: int = 19530,
        username: str = "root",
        password: str = "Milvus",
        partition: str | None = None,
        recreate: bool = False,
        collection_description: str | None = None,
    ) -> DatabaseClientConfig:
        """
        Create a Milvus-specific database configuration.

        This is a convenience factory method for creating Milvus configurations
        with sensible defaults.

        Args:
            collection: Name of the Milvus collection
            host: Milvus server hostname (default: localhost)
            port: Milvus server port (default: 19530)
            username: Authentication username (default: root)
            password: Authentication password (default: Milvus)
            partition: Optional partition name
            recreate: Whether to recreate the collection
            collection_description: Optional collection description

        Returns:
            DatabaseClientConfig configured for Milvus
        """
        return cls(
            provider="milvus",
            collection=collection,
            host=host,
            port=port,
            username=username,
            password=password,
            partition=partition,
            recreate=recreate,
            collection_description=collection_description,
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
        metadata_schema: dict[str, Any],
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
    def create_collection(self, recreate: bool = False) -> None:
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
    def insert_data(self, data: list[DatabaseDocument]) -> None:
        # TODO: All locations where minio is used, update to use default_url
        """
        Insert data into the collection with duplicate detection.

        Expected data format:
        [
            {
                "default_text": "content text",
                "default_text_embedding": [0.1, 0.2, ...],
                "default_chunk_index": 0,
                "default_source": "filename",
                "default_document_id": "uuid",
                "default_minio": "optional_url",
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


class BenchmarkResult(BaseModel):
    """
    Results for a single query in a benchmark run.

    This model captures the performance metrics for a single search query,
    including whether the expected document was found and where it ranked.

    Attributes:
        query: The search query text
        expected_source: The expected document source that should be found
        placement_order: Position where the expected document was found (1-indexed, None if not found)
        distance: Similarity distance/score for the result
        time_to_search: Time taken to execute the search in seconds
        found: Whether the expected document was found in the results
    """

    query: str = Field(..., description="The search query text that was executed")
    expected_source: str = Field(..., description="The expected document source identifier that should be retrieved")
    placement_order: int | None = Field(default=None, ge=1, description="1-indexed position where the expected document was found (None if not found)")
    distance: float | None = Field(default=None, description="Similarity distance or score for the retrieved result")
    time_to_search: float = Field(default=0.0, ge=0, description="Time taken to execute the search query in seconds")
    found: bool = Field(default=False, description="Whether the expected document was found in the search results")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary containing all benchmark result fields
        """
        return self.model_dump()  # type: ignore[no-any-return]


class BenchmarkRunResults(BaseModel):
    """
    Aggregated results from a complete benchmark run.

    This model contains all the metrics and statistics collected during
    a benchmark run across multiple documents and queries.

    Attributes:
        results_by_doc: Mapping of document source to list of benchmark results
        placement_distribution: Histogram of placement positions (position -> count)
        distance_distribution: List of all similarity distances from results
        percent_in_top_k: Percentage of queries found in top-k results (k -> percentage)
        search_time_distribution: List of all search times in seconds
    """

    results_by_doc: dict[str, list[BenchmarkResult]] = Field(..., description="Mapping of document source identifiers to their benchmark results")
    placement_distribution: dict[int, int] = Field(..., description="Distribution of placement positions: position (1-indexed) -> frequency count")
    distance_distribution: list[float] = Field(..., description="List of all similarity distances/scores from the benchmark")
    percent_in_top_k: dict[int, float] = Field(..., description="Percentage of queries found in top-k results: k -> percentage (0-100)")
    search_time_distribution: list[float] = Field(..., description="List of all search execution times in seconds")

    model_config = {
        "validate_assignment": True,
    }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary representation.

        Integer dictionary keys are converted to strings for JSON compatibility.
        BenchmarkResult objects are also converted to dictionaries.

        Returns:
            Dictionary representation with all data serialized
        """
        return {
            "results_by_doc": {source: [result.to_dict() for result in results] for source, results in self.results_by_doc.items()},
            "placement_distribution": {str(k): v for k, v in self.placement_distribution.items()},
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
    def __init__(self, db_config: DatabaseClientConfig, embed_config: EmbedderConfig) -> None:
        """
        Initialize the database benchmark client.
        Args:
            db_config: Configuration for the database client.
            embed_config: Configuration for the embedding provider.
        """
        pass

    @abstractmethod
    def search(self, queries: list[str], filters: list[str] | None = None) -> list[dict[str, Any]]:
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
            queries_by_doc: An optional dictionary where keys are document source
                            identifiers and values are lists of queries to run. If not
                            provided, queries will be generated automatically.
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
