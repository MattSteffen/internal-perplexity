import logging
import json
import uuid
import time
from typing import List, Dict, Any, Optional, Set, Tuple

from numpy import partition
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
from .milvus_utils import create_schema, create_index, DEFAULT_SECURITY_GROUP


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
        metadata_schema: Dict[str, Any],
        library_context: str,
    ):
        """
        Initialize the Milvus database client.

        Args:
            config: DatabaesClientConfig instance with connection parameters
            embedding_dimension: Vector embedding dimensionality
            metadata_schema: JSON schema defining user metadata fields
            library_context: Library context for the database client
        """

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.metadata_schema = metadata_schema if metadata_schema is not None else {}
        self.library_context = library_context
        # Initialize Milvus client
        try:
            self.client = MilvusClient(uri=self.config.uri, token=self.config.token)
            # Test the connection
            self.client.list_collections()
            logging.info("Connected to Milvus successfully!")
        except MilvusException as e:
            logging.error(f"Error connecting to Milvus: {e}")
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
                logging.info(f"Dropping existing collection: {self.config.collection}")
                self.client.drop_collection(self.config.collection)
                collection_exists = False

            if not collection_exists:
                self._create_collection()
                logging.info(f"Created collection: {self.config.collection}")
            else:
                logging.info(f"Collection already exists: {self.config.collection}")

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
            logging.error(
                f"Failed to create collection '{self.config.collection}': {e}"
            )
            raise

    def _create_collection(self) -> None:
        """
        Internal method to create a new collection with the specified schema.
        """
        collection_schema = create_schema(
            self.embedding_dimension, self.metadata_schema, self.library_context
        )
        index = create_index(self.client)

        self.client.create_collection(
            collection_name=self.config.collection,
            dimension=self.embedding_dimension,
            schema=collection_schema,
            index_params=index,
            vector_field_name="default_text_embedding",
            auto_id=True,
        )

        # Create partition if specified
        if self.config.partition:
            self.client.create_partition(self.config.collection, self.config.partition)
            logging.info(f"Created partition: {self.config.partition}")

    def _existing_chunk_indexes(self, source: str) -> Set[int]:
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
                filter=f"default_source == '{source}'",
                output_fields=["default_chunk_index"],
                limit=10000,  # Adjust limit as needed
            )
            return {result["default_chunk_index"] for result in results}

        except MilvusException as e:
            logging.error(
                f"Failed to fetch existing chunk indexes for source '{source}': {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error fetching chunk indexes: {e}")
            raise

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
                filter=f"default_source == '{source}' AND default_chunk_index == {chunk_index}",
                output_fields=["default_source"],
                limit=1,
            )
            return len(results) > 0

        except MilvusException as e:
            logging.error(
                f"Failed to check for duplicate (source: {source}, chunk_index: {chunk_index}): {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error checking for duplicate: {e}")
            raise

    def insert_data(self, data: List[DatabaseDocument]) -> None:
        """
        Insert data into the collection with optimized duplicate detection and progress tracking.

        Expected data format matches DatabaseDocument protocol:
        - default_text: str
        - default_text_embedding: List[float]
        - default_chunk_index: int
        - default_source: str
        - Additional fields accessible via dict-like interface

        Args:
            data: List of document chunks to insert

        Raises:
            DatabaseError: If insertion fails
        """
        insert_start_time = time.time()

        if not data:
            logging.info("Received empty data list. No data to insert.")
            return

        logging.info(
            f"💾 Starting data insertion for {len(data)} items into collection '{self.config.collection}'"
        )

        # Group items by source for optimized duplicate detection
        items_by_source: Dict[str, List[DatabaseDocument]] = {}
        for item in data:
            source = item.default_source
            if source not in items_by_source:
                items_by_source[source] = []
            items_by_source[source].append(item)

        logging.info(
            f"📊 Grouped {len(data)} items into {len(items_by_source)} sources"
        )

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
                            chunk_index = item.default_chunk_index

                            # Check for duplicates using the cached indexes
                            if chunk_index in existing_indexes:
                                duplicates_found += 1
                                logging.debug(
                                    f"⏭️  Duplicate found for source: {source}, chunk_index: {chunk_index}. Skipping."
                                )
                                pbar.update(1)
                                continue

                            # Prepare the data for insertion
                            prepared_item = item.to_dict()
                            if prepared_item["default_document_id"] is None:
                                prepared_item["default_document_id"] = str(uuid.uuid4())

                            # Add metadata as JSON string (include benchmark_questions in all chunks)
                            prepared_item["default_metadata"] = json.dumps(
                                item.metadata, separators=(",", ":")
                            )

                            # Add security group to the item
                            # TODO: Should require a security group to be set not set a default, this should be checked earlier
                            if "security_group" not in prepared_item:
                                prepared_item["security_group"] = DEFAULT_SECURITY_GROUP

                            # Remove function output fields - these are computed by Milvus functions
                            # and should NOT be provided in insert data
                            prepared_item.pop("default_text_sparse_embedding", None)
                            prepared_item.pop("default_metadata_sparse_embedding", None)

                            # Remove the id field - it's auto-generated by Milvus
                            prepared_item.pop("id", None)

                            data_to_insert.append(prepared_item)
                            pbar.set_postfix_str(f"Source: {source}")
                            pbar.update(1)

                        except AttributeError as e:
                            processing_errors += 1
                            logging.warning(
                                f"⚠️  Skipping item that doesn't conform to DatabaseDocument protocol: {e}"
                            )
                            pbar.update(1)
                            continue
                        except Exception as e:
                            processing_errors += 1
                            logging.error(f"❌ Error processing item: {e}")
                            pbar.update(1)
                            continue

                except Exception as e:
                    processing_errors += len(source_items)
                    logging.error(f"❌ Error processing source '{source}': {e}")
                    pbar.update(len(source_items))
                    continue

        # Log processing statistics
        processing_time = time.time() - insert_start_time
        logging.info("=== Item Processing completed ===")
        logging.info("📊 Processing Statistics:")
        logging.info(f"   • Total items received: {len(data)}")
        logging.info(f"   • Duplicates found: {duplicates_found}")
        logging.info(f"   • Processing errors: {processing_errors}")
        logging.info(f"   • Items to insert: {len(data_to_insert)}")
        logging.info(f"   • Processing time: {processing_time:.2f}s")
        logging.info(f"   • Processing rate: {len(data)/processing_time:.1f} items/sec")

        if duplicates_found > 0:
            logging.info(f"⏭️  Skipped {duplicates_found} duplicate items")

        if processing_errors > 0:
            logging.warning(f"⚠️  {processing_errors} items had processing errors")

        if not data_to_insert:
            logging.info(
                "ℹ️  No new data to insert after duplicate check and validation."
            )
            return

        # Insert the data with progress tracking
        try:
            logging.info(
                f"📥 Inserting {len(data_to_insert)} new items into collection '{self.config.collection}'..."
            )

            insert_api_start = time.time()

            result = self.client.insert(
                collection_name=self.config.collection,
                partition_name=self.config.partition,
                data=data_to_insert,
            )

            insert_api_time = time.time() - insert_api_start
            insert_count = result.get("insert_count", len(data_to_insert))

            # Flush to ensure data is persisted
            flush_start = time.time()
            self.client.flush(self.config.collection)
            flush_time = time.time() - flush_start

            total_insert_time = time.time() - insert_start_time

            logging.info("✅ Data insertion completed successfully")
            logging.info("📊 Database Insertion Statistics:")
            logging.info(f"   • Items inserted: {insert_count}")
            logging.info(f"   • API insert time: {insert_api_time:.2f}s")
            logging.info(f"   • Flush time: {flush_time:.2f}s")
            logging.info(f"   • Total insert time: {total_insert_time:.2f}s")
            logging.info(
                f"   • Insert rate: {insert_count/insert_api_time:.1f} items/sec"
            )

        except MilvusException as e:
            logging.error(
                f"❌ Failed to insert data into collection '{self.config.collection}': {e}"
            )
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error during data insertion: {e}")
            raise
