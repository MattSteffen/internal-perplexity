import json
import uuid
from typing import TYPE_CHECKING, Any

from pymilvus import (
    MilvusClient,
    MilvusException,
)
from tqdm import tqdm

from .database_client import (
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
)
from .milvus_utils import DEFAULT_SECURITY_GROUP, create_index, create_schema, extract_collection_description

if TYPE_CHECKING:
    from ..config import CrawlerConfig


class MilvusDB(DatabaseClient):
    """
    Milvus implementation of the DatabaseClient interface.

    Manages interaction with a Milvus vector database collection for storing
    document chunks and their embeddings.
    """

    def __init__(
        self,
        config: DatabaseClientConfig,
        embedding_dimension: int,
        crawler_config: "CrawlerConfig",
    ):
        """
        Initialize the Milvus database client.

        Args:
            config: DatabaesClientConfig instance with connection parameters
            embedding_dimension: Vector embedding dimensionality
            crawler_config: CrawlerConfig containing collection configuration
        """

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.crawler_config = crawler_config
        # Initialize Milvus client
        try:
            self.client = MilvusClient(uri=self.config.uri, token=self.config.token)
            # Test the connection
            self.client.list_collections()
        except MilvusException as e:
            raise e

        # Create collection if needed
        self.create_collection(recreate=self.config.recreate)

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create a collection/table with the specified schema.

        Args:
            recreate: If True, drop existing collection and recreate

        Raises:
            DatabaseError: If collection creation fails
        """
        try:
            collection_exists = self.client.has_collection(self.config.collection)

            if collection_exists and recreate:
                self.client.drop_collection(self.config.collection)
                collection_exists = False

            if not collection_exists:
                self._create_collection()

            if self.config.partition:
                partition_exists = self.client.has_partition(
                    collection_name=self.config.collection,
                    partition_name=self.config.partition,
                )
                if recreate:
                    self.client.drop_partition(
                        collection_name=self.config.collection,
                        partition_name=self.config.partition,
                    )
                    partition_exists = False

                if not partition_exists:
                    self.client.create_partition(
                        collection_name=self.config.collection,
                        partition_name=self.config.partition,
                    )

        except MilvusException as e:
            raise e

    def _create_collection(self) -> None:
        """
        Internal method to create a new collection with the specified schema.
        """
        collection_schema = create_schema(
            self.embedding_dimension,
            self.crawler_config,
        )
        index = create_index(self.client)

        self.client.create_collection(
            collection_name=self.config.collection,
            dimension=self.embedding_dimension,
            schema=collection_schema,
            index_params=index,
            vector_field_name="text_embedding",
            auto_id=True,
        )

        # Create partition if specified
        if self.config.partition:
            self.client.create_partition(self.config.collection, self.config.partition)

    def _existing_chunk_indexes(self, source: str) -> set[int]:
        """
        Get all existing chunk indexes for a given source in a single query.

        Args:
            source: Source identifier (file path)

        Returns:
            Set[int]: Set of existing chunk indexes for the source
        """
        try:
            results = self.client.query(
                collection_name=self.config.collection,
                filter=f"source == '{source}'",
                output_fields=["chunk_index"],
                limit=10000,  # Adjust limit as needed
            )
            return {result["chunk_index"] for result in results}

        except MilvusException as e:
            raise e
        except Exception as e:
            raise e

    def check_duplicate(self, source: str, chunk_index: int) -> bool:
        """
        Check if a document chunk already exists.

        Args:
            source: Source identifier (file path)
            chunk_index: Chunk index within the document

        Returns:
            bool: True if duplicate exists, False otherwise
        """
        try:
            results = self.client.query(
                collection_name=self.config.collection,
                filter=f"source == '{source}' AND chunk_index == {chunk_index}",
                output_fields=["source"],
                limit=1,
            )
            return len(results) > 0

        except MilvusException as e:
            raise e
        except Exception:
            raise

    def insert_data(self, data: list[DatabaseDocument]) -> None:
        """
        Insert data into the collection with optimized duplicate detection and progress tracking.

        Expected data format matches DatabaseDocument protocol:
        - text: str
        - text_embedding: List[float]
        - chunk_index: int
        - source: str
        - Additional fields accessible via dict-like interface

        Args:
            data: List of document chunks to insert

        Raises:
            DatabaseError: If insertion fails
        """
        # insert_start_time = time.time()

        if not data:
            return

        # Group items by source for optimized duplicate detection
        items_by_source: dict[str, list[DatabaseDocument]] = {}
        for item in data:
            source = item.source
            if source not in items_by_source:
                items_by_source[source] = []
            items_by_source[source].append(item)

        # Filter out duplicates and prepare data for insertion
        data_to_insert = []
        duplicates_found = 0
        processing_errors = 0

        # Process items by source with progress tracking
        with tqdm(total=len(data), desc="Processing items", unit="item") as pbar:
            for source, source_items in items_by_source.items():
                try:
                    # Fetch all existing chunk indexes for this source in one query
                    existing_indexes = self._existing_chunk_indexes(source)

                    for item in source_items:
                        try:
                            # Validate required fields exist (these should be guaranteed by protocol)
                            chunk_index = item.chunk_index

                            # Check for duplicates using the cached indexes
                            if chunk_index in existing_indexes:
                                duplicates_found += 1
                                pbar.update(1)
                                continue

                            # Prepare the data for insertion
                            prepared_item = item.to_dict()
                            if prepared_item["document_id"] is None:
                                prepared_item["document_id"] = str(uuid.uuid4())

                            # Add metadata as JSON string
                            prepared_item["str_metadata"] = json.dumps(item.metadata, separators=(",", ":"))

                            # Serialize benchmark_questions as JSON string (schema expects VARCHAR, not list)
                            benchmark_questions = prepared_item.get("benchmark_questions")
                            if benchmark_questions is None:
                                prepared_item["benchmark_questions"] = "[]"
                            elif isinstance(benchmark_questions, list):
                                prepared_item["benchmark_questions"] = json.dumps(benchmark_questions, separators=(",", ":"))
                            elif not isinstance(benchmark_questions, str):
                                # If it's not a list or string, convert to empty array string
                                prepared_item["benchmark_questions"] = "[]"
                            # If it's already a string, leave it as is

                            # Add security group to the item
                            # TODO: Should require a security group to be set not set a default, this should be checked earlier
                            if "security_group" not in prepared_item:
                                prepared_item["security_group"] = DEFAULT_SECURITY_GROUP

                            # Remove function output fields - these are computed by Milvus functions
                            # and should NOT be provided in insert data
                            prepared_item.pop("text_sparse_embedding", None)
                            prepared_item.pop("metadata_sparse_embedding", None)

                            # Remove the id field - it's auto-generated by Milvus
                            prepared_item.pop("id", None)

                            data_to_insert.append(prepared_item)
                            pbar.set_postfix_str(f"Source: {source}")
                            pbar.update(1)

                        except AttributeError:
                            processing_errors += 1
                            pbar.update(1)
                            continue
                        except Exception:
                            processing_errors += 1
                            pbar.update(1)
                            continue

                except Exception:
                    processing_errors += len(source_items)
                    pbar.update(len(source_items))
                    continue

        # Log processing statistics
        # processing_time = time.time() - insert_start_time

        # if duplicates_found > 0:

        # if processing_errors > 0:

        if not data_to_insert:
            return

        # Insert the data with progress tracking
        try:
            # insert_api_start = time.time()

            self.client.insert(
                collection_name=self.config.collection,
                partition_name=self.config.partition,
                data=data_to_insert,
            )

            # insert_api_time = time.time() - insert_api_start
            # insert_count = result.get("insert_count", len(data_to_insert))

            # Flush to ensure data is persisted
            self.client.flush(self.config.collection)
            # flush_time = time.time() - flush_start

            # total_insert_time = time.time() - insert_start_time

        except MilvusException as e:
            raise e
        except Exception as e:
            raise e

    def get_collection_description(self, collection_name: str) -> str | None:
        """
        Retrieve the collection description from Milvus.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection description string, or None if collection doesn't exist or has no description
        """
        try:
            if not self.client.has_collection(collection_name):
                return None

            # Get collection schema which contains the description
            collection_info = self.client.describe_collection(collection_name)
            return extract_collection_description(collection_info)
        except MilvusException:
            return None
        except Exception:
            return None
