import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from ..llm.embeddings import EmbedderConfig

if TYPE_CHECKING:
    from ..config import CrawlerConfig


class DatabaseDocument(BaseModel):
    """
    Pydantic model defining the interface for document data.

    System fields use descriptive names for internal fields to avoid conflicts with user metadata.
    This model provides automatic validation and serialization for document chunks.

    Attributes:
        id: Database-assigned primary key (default: -1 before insertion)
        metadata: Additional user-defined metadata as key-value pairs
        security_group: List of security groups for RBAC access control
        document_id: Unique identifier for the document chunk (UUID)
        text: The text content of the document chunk
        text_embedding: Dense vector embedding of the text
        chunk_index: Index of this chunk within the document (0-based)
        source: Source identifier (e.g., file path, URL)
        minio: URL to the original document in MinIO storage
        text_sparse_embedding: Sparse embedding for BM25 full-text search (internal field)
        metadata_sparse_embedding: Sparse embedding of metadata for BM25 search (internal field)
        benchmark_questions: List of benchmark questions for testing
    """
    id: int = Field(default=-1, description="Database-assigned primary key (auto-generated on insert)")

    # Required attributes - these must exist
    document_id: str = Field(description="Unique identifier for the document chunk (UUID format)")
    text: str = Field(description="The text content of the document chunk")
    chunk_index: int = Field(ge=0, description="Zero-based index of this chunk within the parent document")
    source: str = Field(description="Source identifier (file path, URL, etc.)")
    security_group: list[str] = Field(description="List of security groups for RBAC row-level access control")
    metadata: dict[str, Any] = Field(default_factory=dict, description="User-defined metadata as key-value pairs")

    # Computed fields - these are computed by the embedder or by the vector db on insert
    text_embedding: list[float] = Field(description="Dense vector embedding of the text content")
    text_sparse_embedding: list[float] | None = Field(default=None, description="Sparse vector embedding for BM25 full-text search on text (computed by DB)")
    metadata_sparse_embedding: list[float] | None = Field(default=None, description="Sparse vector embedding for BM25 full-text search on metadata (computed by DB)")
    
    # Optional fields - these are optional and can be set by the user
    benchmark_questions: list[str] | None = Field(default_factory=list, description="List of benchmark questions generated for testing retrieval")

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
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatabaseDocument":
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
        return cls.model_validate(data)

    def to_string(self) -> str:
        """
        Convert to formatted JSON string.
        Useful for LLM input or debugging.

        Returns:
            Pretty-printed JSON string
        """
        return self.model_dump_json(indent=4)


class SearchResult(BaseModel):
    """
    Result from a search operation with distance/score metadata.

    This model wraps a DatabaseDocument with additional search-specific
    information like distance and normalized score.

    Attributes:
        document: The matched DatabaseDocument
        distance: Distance from query vector (lower is more similar for cosine)
        score: Normalized similarity score (0-1, higher is better)
    """

    document: DatabaseDocument = Field(..., description="The matched database document")
    distance: float = Field(..., description="Distance from query vector (lower is more similar)")
    score: float = Field(default=0.0, description="Normalized similarity score (0-1, higher is better)")

    model_config = {
        "validate_assignment": True,
    }

    @classmethod
    def from_milvus_hit(cls, hit: dict[str, Any], output_fields: list[str]) -> "SearchResult":
        """
        Create a SearchResult from a Milvus search hit.

        Args:
            hit: Raw hit dictionary from Milvus search
            output_fields: List of fields that were requested

        Returns:
            SearchResult instance
        """
        distance = hit.get("distance", 0.0)
        # Convert distance to score (for cosine, score = 1 - distance for normalized vectors)
        # Milvus cosine distance is already 1 - cosine_similarity for COSINE metric
        score = 1.0 - distance if distance <= 1.0 else 0.0

        # Build document from hit data
        doc_data = {
            "id": hit.get("id", -1),
            "document_id": hit.get("document_id", ""),
            "text": hit.get("text", ""),
            "chunk_index": hit.get("chunk_index", 0),
            "source": hit.get("source", ""),
            "security_group": hit.get("security_group", ["public"]),
            "metadata": hit.get("metadata", {}),
            "text_embedding": hit.get("text_embedding", []),
        }

        # Add optional fields if present
        if "benchmark_questions" in hit:
            bq = hit["benchmark_questions"]
            if isinstance(bq, str):
                import json
                try:
                    doc_data["benchmark_questions"] = json.loads(bq)
                except json.JSONDecodeError:
                    doc_data["benchmark_questions"] = []
            else:
                doc_data["benchmark_questions"] = bq or []

        document = DatabaseDocument.from_dict(doc_data)
        return cls(document=document, distance=distance, score=score)


class UpsertResult(BaseModel):
    """
    Result from an upsert operation.

    Provides counts of inserted vs updated documents and any failures.

    Attributes:
        inserted_count: Number of new documents inserted
        updated_count: Number of existing documents updated
        failed_ids: List of document IDs that failed to upsert
    """

    inserted_count: int = Field(default=0, ge=0, description="Number of new documents inserted")
    updated_count: int = Field(default=0, ge=0, description="Number of existing documents updated")
    failed_ids: list[str] = Field(default_factory=list, description="List of document IDs that failed to upsert")

    model_config = {
        "validate_assignment": True,
    }

    @property
    def total_count(self) -> int:
        """Total number of successfully upserted documents."""
        return self.inserted_count + self.updated_count

    @property
    def has_failures(self) -> bool:
        """Check if any documents failed to upsert."""
        return len(self.failed_ids) > 0


class CollectionDescription(BaseModel):
    """
    Typed model for Milvus collection descriptions.

    This model stores all the metadata needed to restore a crawler configuration
    from an existing collection, including the crawler config, metadata schema,
    library context, and LLM prompt.

    Attributes:
        collection_config: Full crawler config dictionary
        llm_prompt: Generated prompt text with metadata filtering instructions
        collection_security_groups: List of security groups that are required to access the collection
        metadata_schema: User-provided metadata JSON schema
        description: Human-readable description of collection data
        columns: List of columns in the collection that searching can return
    """
    collection_config: "CrawlerConfig" = Field(..., description="Full crawler config object")
    llm_prompt: str = Field(default="", description="Generated prompt text with metadata filtering instructions")
    columns: list[str] = Field(default=["id", "text", "source", "document_id", "chunk_index", "metadata", "text_embedding", "text_sparse_embedding", "metadata_sparse_embedding", "benchmark_questions"], description="List of columns in the collection that searching can return")
    

    model_config = {
        "validate_assignment": True,
    }

    @property
    def collection_security_groups(self) -> list[str]:
        """
        Get the security groups that are required to access the collection.
        Computed from collection_config.security_groups.
        """
        return self.collection_config.security_groups
        
    @property
    def metadata_schema(self) -> dict[str, Any]:
        """
        Get the metadata schema that is used to validate the metadata.
        Computed from collection_config.metadata_schema.
        """
        return self.collection_config.metadata_schema
    
    @property
    def description(self) -> str:
        """
        Get the human-readable description of the collection.
        Computed from collection_config.database.collection_description.
        """
        return self.collection_config.database.collection_description or ""

    def to_json(self) -> str:
        """
        Convert to JSON string for storage in Milvus collection description.

        Returns:
            JSON string representation of the collection description
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, description: str) -> "CollectionDescription | None":
        """
        Restore CollectionDescription from a JSON string.

        Args:
            description: JSON string from Milvus collection description

        Returns:
            CollectionDescription instance, or None if parsing fails
        """
        if not description:
            return None

        try:
            data = json.loads(description.strip())
            if isinstance(data, dict):
                # Handle backward compatibility: if collection_config_json exists, convert it
                if "collection_config_json" in data and "collection_config" not in data:
                    from ..config import CrawlerConfig
                    data["collection_config"] = CrawlerConfig.from_dict(data.pop("collection_config_json"))
                # Ensure collection_config is a CrawlerConfig object
                if "collection_config" in data and isinstance(data["collection_config"], dict):
                    from ..config import CrawlerConfig
                    data["collection_config"] = CrawlerConfig.from_dict(data["collection_config"])
                return cls.model_validate(data)
            return None
        except (json.JSONDecodeError, ValueError, Exception):
            return None

    def to_crawler_config(self, database_config: "DatabaseClientConfig") -> "CrawlerConfig":
        """
        Create a CrawlerConfig from the stored collection_config.

        This method delegates to CrawlerConfig.from_collection_description() which is
        the canonical implementation for restoring configs from collection descriptions.

        Args:
            database_config: Database configuration to use (collection name will be overridden)

        Returns:
            CrawlerConfig instance restored from collection_config

        Raises:
            ValueError: If collection_config is None or invalid
        """
        from ..config import CrawlerConfig

        return CrawlerConfig.from_collection_description(self, database_config)


class DatabaseClientConfig(BaseModel):
    """
    Base configuration for database clients.

    This model provides type-safe configuration for connecting to vector databases.
    All connection parameters are validated at creation time.

    Attributes:
        provider: Database provider name (e.g., "milvus")
        collection: Name of the database collection/table
        partition: Optional partition name within the collection
        access_level: Access level of the collection: public, private, admin
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
    access_level: str = Field(default="public", description="Access level of the collection: public, private, admin")
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

    def copy_with_overrides(self, **overrides) -> "DatabaseClientConfig":
        """
        Create a copy with specified field overrides.

        Args:
            **overrides: Field values to override in the copy

        Returns:
            New DatabaseClientConfig instance with overridden fields
        """
        return self.model_copy(update=overrides)

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
    ) -> "DatabaseClientConfig":
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

    Connection Lifecycle:
        1. Create client with __init__()
        2. Call connect() to establish connection
        3. Use CRUD methods (search, get, upsert, delete)
        4. Call disconnect() when done

    All methods except __init__, connect, is_connected, and disconnect
    require an active connection (will raise RuntimeError if not connected).
    """

    @abstractmethod
    def __init__(
        self,
        config: DatabaseClientConfig,
        embedding_dimension: int,
        crawler_config: "CrawlerConfig",
    ):
        """
        Initialize the database client without connecting.

        Args:
            config: Database-specific configuration parameters
            embedding_dimension: Vector embedding dimensionality
            crawler_config: CrawlerConfig containing collection configuration
        """
        pass

    @abstractmethod
    def connect(self, create_if_missing: bool = False) -> "DatabaseClient":
        """
        Establish connection to the database.

        Must be called before any other operation except is_connected().
        Can be called multiple times safely (idempotent).

        Args:
            create_if_missing: If True, create collection if it doesn't exist

        Returns:
            Self for method chaining

        Raises:
            ConnectionError: If connection fails
            RuntimeError: If collection doesn't exist and create_if_missing is False
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection.

        Safe to call multiple times. After disconnect, connect() must be
        called again before using other methods.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if client is currently connected to the database.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    def search(
        self,
        texts: list[str],
        filters: list[str] | None = None,
        limit: int = 10,
    ) -> list["SearchResult"]:
        """
        Hybrid search combining dense and sparse vectors.

        Performs a multi-vector search using:
        - Dense embeddings (text_embedding) for semantic similarity
        - Sparse embeddings (text_sparse_embedding) for keyword matching
        - Sparse metadata embeddings for metadata-based search

        Results are ranked using Reciprocal Rank Fusion (RRF).

        Args:
            texts: Query texts to search for (will be embedded)
            filters: Optional Milvus filter expressions (e.g., ["source == 'file.pdf'"])
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects sorted by relevance

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def get_chunk(self, id: int) -> DatabaseDocument | None:
        """
        Get a single chunk by its database ID.

        Args:
            id: The database-assigned primary key (auto-generated on insert)

        Returns:
            DatabaseDocument if found, None otherwise

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> list[DatabaseDocument]:
        """
        Get all chunks for a document by its document_id (UUID).

        Chunks are returned sorted by chunk_index.

        Args:
            document_id: The document's unique identifier (UUID format)

        Returns:
            List of DatabaseDocument chunks, empty list if not found

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def upsert(self, documents: list[DatabaseDocument]) -> "UpsertResult":
        """
        Insert or update documents.

        Uses (source, chunk_index) as the unique key for determining
        whether to insert or update. If a document with the same
        source and chunk_index exists, it will be updated.

        Args:
            documents: List of DatabaseDocument objects to upsert

        Returns:
            UpsertResult with counts of inserted, updated, and failed documents

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def delete_chunk(self, id: int) -> bool:
        """
        Delete a single chunk by its database ID.

        Args:
            id: The database-assigned primary key

        Returns:
            True if chunk was deleted, False if not found

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document by document_id.

        Args:
            document_id: The document's unique identifier (UUID format)

        Returns:
            Number of chunks deleted

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def create_collection(self, recreate: bool = False) -> None:
        """
        Create collection with the defined schema.

        Only callable when connected. The collection schema includes:
        - System fields (id, document_id, text, text_embedding, etc.)
        - Security group for RBAC
        - BM25 sparse indexes for full-text search

        Args:
            recreate: If True, drop existing collection and recreate

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def get_collection(self) -> CollectionDescription | None:
        """
        Get collection description with full config for pipeline reconstruction.

        The description contains the complete CrawlerConfig used to create
        the collection, allowing the pipeline to be restored from the
        collection metadata.

        Returns:
            CollectionDescription if collection exists and has valid description,
            None otherwise

        Raises:
            RuntimeError: If not connected
        """
        pass

    @abstractmethod
    def exists(self, source: str, chunk_index: int) -> bool:
        """
        Check if a document chunk already exists.

        This is a convenience method that can be used for duplicate detection
        before processing. Equivalent to checking if get_document returns
        non-empty results with the given source and chunk_index.

        Args:
            source: Source identifier (file path, URL, etc.)
            chunk_index: Chunk index within the document

        Returns:
            True if chunk exists, False otherwise

        Raises:
            RuntimeError: If not connected
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
        return self.model_dump()


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
        mrr: Mean Reciprocal Rank - average of 1/rank for found documents
        recall_at_k: Recall@K metrics - fraction of relevant docs retrieved in top-k
        precision_at_k: Precision@K metrics - fraction of top-k results that are relevant
        ndcg_at_k: Normalized Discounted Cumulative Gain@K metrics
        hit_rate_at_k: Hit rate@K - percentage of queries with at least one relevant result in top-k
        mean_placement: Mean placement position of found documents
        median_placement: Median placement position of found documents
        std_placement: Standard deviation of placement positions
        total_queries: Total number of queries executed
        queries_found: Number of queries where expected document was found
        queries_not_found: Number of queries where expected document was not found
    """

    results_by_doc: dict[str, list[BenchmarkResult]] = Field(..., description="Mapping of document source identifiers to their benchmark results")
    placement_distribution: dict[int, int] = Field(..., description="Distribution of placement positions: position (1-indexed) -> frequency count")
    distance_distribution: list[float] = Field(..., description="List of all similarity distances/scores from the benchmark")
    percent_in_top_k: dict[int, float] = Field(..., description="Percentage of queries found in top-k results: k -> percentage (0-100)")
    search_time_distribution: list[float] = Field(..., description="List of all search execution times in seconds")
    mrr: float = Field(..., description="Mean Reciprocal Rank - average of 1/rank for found documents")
    recall_at_k: dict[int, float] = Field(..., description="Recall@K metrics - fraction of relevant docs retrieved in top-k")
    precision_at_k: dict[int, float] = Field(..., description="Precision@K metrics - fraction of top-k results that are relevant")
    ndcg_at_k: dict[int, float] = Field(..., description="Normalized Discounted Cumulative Gain@K metrics")
    hit_rate_at_k: dict[int, float] = Field(..., description="Hit rate@K - percentage of queries with at least one relevant result in top-k")
    mean_placement: float = Field(..., description="Mean placement position of found documents")
    median_placement: float = Field(..., description="Median placement position of found documents")
    std_placement: float = Field(..., description="Standard deviation of placement positions")
    total_queries: int = Field(..., description="Total number of queries executed")
    queries_found: int = Field(..., description="Number of queries where expected document was found")
    queries_not_found: int = Field(..., description="Number of queries where expected document was not found")

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
            "mrr": self.mrr,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
            "hit_rate_at_k": {str(k): v for k, v in self.hit_rate_at_k.items()},
            "mean_placement": self.mean_placement,
            "median_placement": self.median_placement,
            "std_placement": self.std_placement,
            "total_queries": self.total_queries,
            "queries_found": self.queries_found,
            "queries_not_found": self.queries_not_found,
        }


class DatabaseBenchmark(ABC):
    """
    Abstract base class for database benchmark clients.
    This interface defines the contract that all database benchmark implementations
    must follow to be compatible with the document processing pipeline.
    """

    @abstractmethod
    def __init__(self, db_config: "DatabaseClientConfig", embed_config: "EmbedderConfig") -> None:
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
    def run_benchmark(self, generate_queries: bool = False, k_values: list[int] | None = None, skip_docs_without_questions: bool = True) -> BenchmarkRunResults:
        """
        Run a benchmark for a set of documents and queries.
        
        Args:
            generate_queries: If True, generate queries using LLM (implementation-specific)
            k_values: List of k values for @K metrics (default: [1, 5, 10, 25, 50, 100])
            skip_docs_without_questions: If True, skip documents without benchmark_questions (default: True)
            
        Returns:
            A BenchmarkRunResults object containing detailed and aggregated results with IR metrics.
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

        # Recall@K curve
        plt.figure()
        k_values = sorted(results.recall_at_k.keys())
        recall_values = [results.recall_at_k[k] * 100 for k in k_values]  # Convert to percentage
        plt.plot(k_values, recall_values, marker="o", label="Recall@K", linewidth=2)
        plt.xlabel("k")
        plt.ylabel("Recall (%)")
        plt.title("Recall@K Performance")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "recall_at_k.png"))
        plt.close()

        # NDCG@K curve
        plt.figure()
        k_values = sorted(results.ndcg_at_k.keys())
        ndcg_values = [results.ndcg_at_k[k] for k in k_values]
        plt.plot(k_values, ndcg_values, marker="o", label="NDCG@K", linewidth=2, color="green")
        plt.xlabel("k")
        plt.ylabel("NDCG")
        plt.title("Normalized Discounted Cumulative Gain@K")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ndcg_at_k.png"))
        plt.close()

        # Combined metrics comparison
        plt.figure(figsize=(10, 6))
        k_values = sorted(results.recall_at_k.keys())
        recall_values = [results.recall_at_k[k] * 100 for k in k_values]
        precision_values = [results.precision_at_k[k] * 100 for k in k_values]
        hit_rate_values = [results.hit_rate_at_k[k] * 100 for k in k_values]
        
        plt.plot(k_values, recall_values, marker="o", label="Recall@K", linewidth=2)
        plt.plot(k_values, precision_values, marker="s", label="Precision@K", linewidth=2)
        plt.plot(k_values, hit_rate_values, marker="^", label="Hit Rate@K", linewidth=2)
        plt.xlabel("k")
        plt.ylabel("Percentage (%)")
        plt.title("IR Metrics Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()

        # MRR and summary statistics bar chart
        plt.figure(figsize=(10, 6))
        metrics = ["MRR", "Mean Placement", "Median Placement"]
        values = [results.mrr, results.mean_placement, results.median_placement]
        colors = ["blue", "orange", "green"]
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel("Value")
        plt.title("Summary Statistics")
        plt.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, "summary_statistics.png"))
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
