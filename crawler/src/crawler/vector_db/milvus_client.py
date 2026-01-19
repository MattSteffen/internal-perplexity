"""
Milvus implementation of the DatabaseClient interface.

This module provides a concrete implementation of the DatabaseClient ABC
for the Milvus vector database. It supports hybrid search (dense + sparse),
CRUD operations, and collection management.
"""

import json
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pymilvus import (
    AnnSearchRequest,
    MilvusClient,
    MilvusException,
    RRFRanker,
)
from tqdm import tqdm

from .database_client import (
    CollectionDescription,
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
    SearchResult,
    UpsertResult,
)
from .milvus_utils import (
    DEFAULT_SECURITY_GROUP,
    create_index,
    create_schema,
    extract_collection_description,
)

if TYPE_CHECKING:
    from ..config import CrawlerConfig
    from ..llm.embeddings import Embedder

# Default output fields for search and query operations
DEFAULT_OUTPUT_FIELDS = [
    "id",
    "document_id",
    "source",
    "chunk_index",
    "text",
    "text_embedding",
    "metadata",
    "security_group",
    "benchmark_questions",
]


class MilvusDB(DatabaseClient):
    """
    Milvus implementation of the DatabaseClient interface.

    Manages interaction with a Milvus vector database collection for storing
    document chunks and their embeddings. Supports hybrid search combining
    dense embeddings with BM25 sparse vectors.

    Example:
        >>> config = DatabaseClientConfig.milvus("my_collection")
        >>> db = MilvusDB(config, embedding_dimension=384, crawler_config=crawler_config)
        >>> db.connect(create_if_missing=True)
        >>> results = db.search(["What is machine learning?"], limit=10)
        >>> db.disconnect()
    """

    def __init__(
        self,
        config: DatabaseClientConfig,
        embedding_dimension: int,
        crawler_config: "CrawlerConfig",
        embedder: "Embedder | None" = None,
    ):
        """
        Initialize the Milvus database client without connecting.

        Args:
            config: DatabaseClientConfig instance with connection parameters
            embedding_dimension: Vector embedding dimensionality
            crawler_config: CrawlerConfig containing collection configuration
            embedder: Optional embedder for search operations (can be set later)
        """
        self.config = config
        self.embedding_dimension = embedding_dimension
        self.crawler_config = crawler_config
        self.embedder = embedder

        # Connection state
        self._client: MilvusClient | None = None
        self._connected: bool = False
        self._user_security_groups: list[str] = []

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self, create_if_missing: bool = False) -> "MilvusDB":
        """
        Establish connection to the Milvus database.

        Args:
            create_if_missing: If True, create collection if it doesn't exist

        Returns:
            Self for method chaining

        Raises:
            ConnectionError: If connection fails
            RuntimeError: If collection doesn't exist and create_if_missing is False
        """
        if self._connected and self._client is not None:
            return self

        try:
            self._client = MilvusClient(uri=self.config.uri, token=self.config.token)
            # Test the connection
            self._client.list_collections()
            self._connected = True
        except MilvusException as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Milvus: {e}") from e

        # Handle collection creation/validation
        collection_exists = self._client.has_collection(self.config.collection)

        if self.config.recreate and collection_exists:
            self._client.drop_collection(self.config.collection)
            collection_exists = False

        if not collection_exists:
            if create_if_missing or self.config.recreate:
                self.create_collection(recreate=False)
            else:
                self._connected = False
                raise RuntimeError(
                    f"Collection '{self.config.collection}' does not exist. "
                    "Set create_if_missing=True to create it."
                )
        else:
            # Load collection for searching
            self._client.load_collection(self.config.collection)

        # Get user security groups
        user_info = self._client.describe_user(self.config.username)
        if user_info:
            user_roles = user_info.get("roles", [])
            if user_roles:
                self._user_security_groups = list[str](user_roles)

        return self

    def disconnect(self) -> None:
        """
        Close the database connection.

        Safe to call multiple times.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore errors during disconnect
            finally:
                self._client = None
                self._connected = False

    def is_connected(self) -> bool:
        """
        Check if client is currently connected to the database.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._client is not None

    def _require_connected(self) -> None:
        """
        Internal guard to ensure connection before operations.

        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected():
            raise RuntimeError(
                "Database not connected. Call connect() before performing operations."
            )

    def set_embedder(self, embedder: "Embedder") -> None:
        """
        Set the embedder for search operations.

        Args:
            embedder: Embedder instance for generating query embeddings
        """
        self.embedder = embedder

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create a collection with the specified schema.

        Args:
            recreate: If True, drop existing collection and recreate

        Raises:
            RuntimeError: If not connected
        """
        self._require_connected()

        try:
            collection_exists = self._client.has_collection(self.config.collection)

            if collection_exists and recreate:
                self._client.drop_collection(self.config.collection)
                collection_exists = False

            if not collection_exists:
                self._create_collection_internal()

            # Handle partition creation
            if self.config.partition:
                partition_exists = self._client.has_partition(
                    collection_name=self.config.collection,
                    partition_name=self.config.partition,
                )
                if recreate and partition_exists:
                    self._client.drop_partition(
                        collection_name=self.config.collection,
                        partition_name=self.config.partition,
                    )
                    partition_exists = False

                if not partition_exists:
                    self._client.create_partition(
                        collection_name=self.config.collection,
                        partition_name=self.config.partition,
                    )

            # Load collection for searching
            self._client.load_collection(self.config.collection)

        except MilvusException as e:
            raise RuntimeError(f"Failed to create collection: {e}") from e

    def _create_collection_internal(self) -> None:
        """
        Internal method to create a new collection with the specified schema.
        """
        collection_schema = create_schema(
            self.embedding_dimension,
            self.crawler_config,
        )
        index = create_index(self._client)

        self._client.create_collection(
            collection_name=self.config.collection,
            dimension=self.embedding_dimension,
            schema=collection_schema,
            index_params=index,
            vector_field_name="text_embedding",
            auto_id=True,
        )

        if self.config.partition:
            self._client.create_partition(self.config.collection, self.config.partition)

    def get_collection(self) -> CollectionDescription | None:
        """
        Get collection description with full config for pipeline reconstruction.

        Returns:
            CollectionDescription if collection exists and has valid description,
            None otherwise

        Raises:
            RuntimeError: If not connected
        """
        self._require_connected()

        try:
            if not self._client.has_collection(self.config.collection):
                return None

            collection_info = self._client.describe_collection(self.config.collection)
            description_str = collection_info.get("description", "")

            if not description_str:
                return None

            return extract_collection_description(description_str)

        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        texts: list[str],
        filters: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors.

        Performs a multi-vector search using:
        - Dense embeddings (text_embedding) for semantic similarity
        - Sparse embeddings (text_sparse_embedding) for keyword matching
        - Sparse metadata embeddings for metadata-based search

        Results are ranked using Reciprocal Rank Fusion (RRF).

        Args:
            texts: Query texts to search for (will be embedded)
            filters: Optional Milvus filter expressions
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by relevance

        Raises:
            RuntimeError: If not connected or embedder not set
        """
        self._require_connected()

        if not texts:
            return []

        # Build filter string with security groups
        all_filters = list(filters or [])

        # Add security group filter
        all_filters.insert(
            0, f"array_contains_any(security_group, {list[str](self._user_security_groups)})"
        )

        filter_str = " and ".join(all_filters) if all_filters else ""

        # Build search requests
        search_requests = []

        for text in texts:
            # Dense vector search (requires embedder)
            if self.embedder is not None:
                try:
                    embedding = self.embedder.embed(text)
                    if embedding:
                        search_requests.append(
                            AnnSearchRequest(
                                data=[embedding],
                                anns_field="text_embedding",
                                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                                expr=filter_str if filter_str else None,
                                limit=limit,
                            )
                        )
                except Exception:
                    pass  # Skip if embedding fails

            # Sparse text search (BM25)
            search_requests.append(
                AnnSearchRequest(
                    data=[text],
                    anns_field="text_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str if filter_str else None,
                    limit=limit,
                )
            )

            # Sparse metadata search (BM25)
            search_requests.append(
                AnnSearchRequest(
                    data=[text],
                    anns_field="metadata_sparse_embedding",
                    param={"drop_ratio_search": 0.2},
                    expr=filter_str if filter_str else None,
                    limit=limit,
                )
            )

        if not search_requests:
            return []

        try:
            # Perform hybrid search with RRF ranking
            ranker = RRFRanker(k=100)
            results = self._client.hybrid_search(
                collection_name=self.config.collection,
                reqs=search_requests,
                ranker=ranker,
                output_fields=DEFAULT_OUTPUT_FIELDS,
                limit=limit,
            )

            # Process results into SearchResult objects
            search_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else hit.entity
                    entity["distance"] = hit.distance
                    entity["id"] = hit.id

                    try:
                        search_result = SearchResult.from_milvus_hit(
                            entity, DEFAULT_OUTPUT_FIELDS
                        )
                        search_results.append(search_result)
                    except Exception:
                        continue

            return search_results

        except MilvusException as e:
            raise RuntimeError(f"Search failed: {e}") from e

    # -------------------------------------------------------------------------
    # Get Operations
    # -------------------------------------------------------------------------

    def get_chunk(self, id: int) -> DatabaseDocument | None:
        """
        Get a single chunk by its database ID.

        Args:
            id: The database-assigned primary key

        Returns:
            DatabaseDocument if found, None otherwise

        Raises:
            RuntimeError: If not connected
        """
        self._require_connected()

        try:
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"id == {id} AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=DEFAULT_OUTPUT_FIELDS,
                limit=1,
            )

            if not results:
                return None

            return self._result_to_document(results[0])

        except MilvusException as e:
            raise RuntimeError(f"Failed to get chunk: {e}") from e

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
        self._require_connected()

        try:
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"document_id == '{document_id}' AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=DEFAULT_OUTPUT_FIELDS,
                limit=10000,  # Support large documents
            )

            if not results:
                return []

            documents = [self._result_to_document(r) for r in results]
            return sorted(documents, key=lambda d: d.chunk_index)

        except MilvusException as e:
            raise RuntimeError(f"Failed to get document: {e}") from e

    def _result_to_document(self, result: dict[str, Any]) -> DatabaseDocument:
        """
        Convert a Milvus query result to a DatabaseDocument.

        Args:
            result: Dictionary from Milvus query

        Returns:
            DatabaseDocument instance
        """
        # Handle benchmark_questions (stored as JSON string)
        benchmark_questions = result.get("benchmark_questions")
        if isinstance(benchmark_questions, str):
            try:
                benchmark_questions = json.loads(benchmark_questions)
            except json.JSONDecodeError:
                benchmark_questions = []

        return DatabaseDocument(
            id=result.get("id", -1),
            document_id=result.get("document_id", ""),
            text=result.get("text", ""),
            text_embedding=result.get("text_embedding", []),
            chunk_index=result.get("chunk_index", 0),
            source=result.get("source", ""),
            security_group=result.get("security_group", ["public"]),
            metadata=result.get("metadata", {}),
            benchmark_questions=benchmark_questions or [],
        )

    # -------------------------------------------------------------------------
    # Upsert Operation
    # -------------------------------------------------------------------------

    def upsert(self, documents: list[DatabaseDocument]) -> UpsertResult:
        """
        Insert or update documents.

        Uses (source, chunk_index) as the unique key for determining
        whether to insert or update.

        Args:
            documents: List of DatabaseDocument objects to upsert

        Returns:
            UpsertResult with counts of inserted, updated, and failed documents

        Raises:
            RuntimeError: If not connected
        """
        self._require_connected()

        if not documents:
            return UpsertResult()

        # Group by source for batch duplicate checking
        by_source: dict[str, list[DatabaseDocument]] = defaultdict(list)
        for doc in documents:
            by_source[doc.source].append(doc)

        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        failed_ids: list[str] = []

        # Process by source with progress tracking
        with tqdm(total=len(documents), desc="Processing documents", unit="doc") as pbar:
            for source, source_docs in by_source.items():
                try:
                    # Fetch existing chunk indexes for this source
                    existing_indexes = self._get_existing_chunk_indexes(source)

                    for doc in source_docs:
                        try:
                            prepared = self._prepare_document_for_insert(doc)

                            if doc.chunk_index in existing_indexes:
                                to_update.append(prepared)
                            else:
                                to_insert.append(prepared)

                            pbar.update(1)

                        except Exception as e:
                            failed_ids.append(doc.document_id)
                            pbar.update(1)

                except Exception:
                    for doc in source_docs:
                        failed_ids.append(doc.document_id)
                    pbar.update(len(source_docs))

        # Insert new documents
        inserted_count = 0
        if to_insert:
            try:
                self._client.insert(
                    collection_name=self.config.collection,
                    partition_name=self.config.partition,
                    data=to_insert,
                )
                inserted_count = len(to_insert)
            except MilvusException as e:
                # All inserts failed
                for item in to_insert:
                    failed_ids.append(item.get("document_id", "unknown"))

        # Update existing documents (delete + insert for Milvus)
        updated_count = 0
        if to_update:
            for item in to_update:
                try:
                    # Delete existing
                    self._client.delete(
                        collection_name=self.config.collection,
                        filter=f"source == '{item['source']}' AND chunk_index == {item['chunk_index']}",
                    )
                except Exception:
                    failed_ids.append(item.get("document_id", "unknown"))
                    continue

            # Insert updated documents
            try:
                self._client.insert(
                    collection_name=self.config.collection,
                    partition_name=self.config.partition,
                    data=to_update,
                )
                updated_count = len(to_update)
            except MilvusException:
                for item in to_update:
                    failed_ids.append(item.get("document_id", "unknown"))

        # Flush to persist
        try:
            self._client.flush(self.config.collection)
        except Exception:
            pass

        return UpsertResult(
            inserted_count=inserted_count,
            updated_count=updated_count,
            failed_ids=failed_ids,
        )

    def _get_existing_chunk_indexes(self, source: str) -> set[int]:
        """
        Get all existing chunk indexes for a given source.

        Args:
            source: Source identifier (file path)

        Returns:
            Set of existing chunk indexes
        """
        try:
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"source == '{source}' AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=["chunk_index"],
                limit=10000,
            )
            return {result["chunk_index"] for result in results}
        except Exception:
            return set()

    def _prepare_document_for_insert(self, doc: DatabaseDocument) -> dict[str, Any]:
        """
        Prepare a DatabaseDocument for Milvus insertion.

        Args:
            doc: DatabaseDocument to prepare

        Returns:
            Dictionary ready for Milvus insert
        """
        prepared = doc.to_dict()

        # Generate document_id if not set
        if not prepared.get("document_id"):
            prepared["document_id"] = str(uuid.uuid4())

        # Serialize metadata as JSON string for str_metadata field
        prepared["str_metadata"] = json.dumps(doc.metadata, separators=(",", ":"))

        # Serialize benchmark_questions as JSON string
        benchmark_questions = prepared.get("benchmark_questions")
        if benchmark_questions is None:
            prepared["benchmark_questions"] = "[]"
        elif isinstance(benchmark_questions, list):
            prepared["benchmark_questions"] = json.dumps(
                benchmark_questions, separators=(",", ":")
            )
        elif not isinstance(benchmark_questions, str):
            prepared["benchmark_questions"] = "[]"

        # Ensure security group is set
        if "security_group" not in prepared or not prepared["security_group"]:
            prepared["security_group"] = DEFAULT_SECURITY_GROUP

        # Remove computed fields (Milvus generates these)
        prepared.pop("text_sparse_embedding", None)
        prepared.pop("metadata_sparse_embedding", None)

        # Remove id (auto-generated)
        prepared.pop("id", None)

        return prepared

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

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
        self._require_connected()

        try:
            # Check if exists first
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"id == {id} AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=["id"],
                limit=1,
            )

            if not results:
                return False

            self._client.delete(
                collection_name=self.config.collection,
                filter=f"id == {id} AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
            )
            return True

        except MilvusException as e:
            raise RuntimeError(f"Failed to delete chunk: {e}") from e

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
        self._require_connected()

        try:
            # Count existing chunks
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"document_id == '{document_id}' AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=["id"],
                limit=10000,
            )

            count = len(results)
            if count == 0:
                return 0

            self._client.delete(
                collection_name=self.config.collection,
                filter=f"document_id == '{document_id}' AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
            )
            return count

        except MilvusException as e:
            raise RuntimeError(f"Failed to delete document: {e}") from e

    # -------------------------------------------------------------------------
    # Exists Check (replaces check_duplicate)
    # -------------------------------------------------------------------------

    def exists(self, source: str, chunk_index: int) -> bool:
        """
        Check if a document chunk already exists.

        Args:
            source: Source identifier (file path, URL, etc.)
            chunk_index: Chunk index within the document

        Returns:
            True if chunk exists, False otherwise

        Raises:
            RuntimeError: If not connected
        """
        self._require_connected()

        try:
            results = self._client.query(
                collection_name=self.config.collection,
                filter=f"source == '{source}' AND chunk_index == {chunk_index} AND array_contains_any(security_group, {list[str](self._user_security_groups)})",
                output_fields=["id"],
                limit=1,
            )
            return len(results) > 0

        except MilvusException as e:
            raise RuntimeError(f"Failed to check existence: {e}") from e
