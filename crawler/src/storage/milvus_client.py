import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple

from numpy import partition
from pymilvus import (
    MilvusClient,
    MilvusException,
)

from .database_client import (
    DatabaseClient,
    DatabaseClientConfig,
    DatabaseDocument,
)
from .milvus_utils import create_schema, create_index


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
    ):
        """
        Initialize the Milvus database client.

        Args:
            config: DatabaesClientConfig instance with connection parameters
            embedding_dimension: Vector embedding dimensionality
            metadata_schema: JSON schema defining user metadata fields
        """

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.metadata_schema = metadata_schema if metadata_schema is not None else {}

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
            self.embedding_dimension, self.metadata_schema
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
            logging.info(f"Created partition: {self.config.partition}")

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
            logging.error(
                f"Failed to check for duplicate (source: {source}, chunk_index: {chunk_index}): {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error checking for duplicate: {e}")
            raise

    def insert_data(self, data: List[DatabaseDocument]) -> None:
        """
        Insert data into the collection with duplicate detection.

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
        if not data:
            logging.info("Received empty data list. No data to insert.")
            return

        # Filter out duplicates and prepare data for insertion
        data_to_insert = []
        duplicates_found = 0

        for item in data:
            try:
                # Validate required fields exist (these should be guaranteed by protocol)
                source = item.source
                chunk_index = item.chunk_index
                text = item.text
                text_embedding = item.text_embedding

                # Check for duplicates
                if self.check_duplicate(source, chunk_index):
                    duplicates_found += 1
                    logging.debug(
                        f"Duplicate found for source: {source}, chunk_index: {chunk_index}. Skipping."
                    )
                    continue

                # Prepare the data for insertion
                prepared_item = {
                    "document_id": item.get("document_id", str(uuid.uuid4())),
                    "text": text,
                    "text_embedding": text_embedding,
                    "chunk_index": chunk_index,
                    "source": source,
                    "minio": item.get("minio", ""),
                }

                # Extract additional metadata fields
                metadata = {}
                required_fields = {
                    "text",
                    "text_embedding",
                    "chunk_index",
                    "source",
                    "document_id",
                    "minio",
                }

                # For dict-like objects, iterate through all keys
                if hasattr(item, "keys"):
                    for key in item.keys():
                        if key not in required_fields:
                            value = (
                                item[key]
                                if hasattr(item, "__getitem__")
                                else getattr(item, key)
                            )
                            metadata[key] = value
                            prepared_item[key] = value
                else:
                    # For objects with attributes, use dir() or __dict__
                    if hasattr(item, "__dict__"):
                        for key, value in item.__dict__.items():
                            if key not in required_fields and not key.startswith("_"):
                                metadata[key] = value
                                prepared_item[key] = value

                # Add metadata as JSON string
                prepared_item["metadata"] = json.dumps(metadata, separators=(",", ":"))

                data_to_insert.append(prepared_item)

            except AttributeError as e:
                logging.warning(
                    f"Skipping item that doesn't conform to DatabaseDocument protocol: {e}"
                )
                continue
            except Exception as e:
                logging.error(f"Error processing item: {e}")
                continue

        if duplicates_found > 0:
            logging.info(
                f"Found {duplicates_found} duplicates. Skipping insertion for these items."
            )

        if not data_to_insert:
            logging.info("No new data to insert after duplicate check and validation.")
            return

        # Insert the data
        try:
            logging.info(
                f"Inserting {len(data_to_insert)} new items into collection '{self.config.collection}'"
            )

            result = self.client.insert(
                collection_name=self.config.collection,
                partition_name=self.config.partition,
                data=data_to_insert,
            )

            # Flush to ensure data is persisted
            self.client.flush(self.config.collection)

            insert_count = result.get("insert_count", len(data_to_insert))
            logging.info(
                f"Successfully inserted {insert_count} items into collection '{self.config.collection}'"
            )

        except MilvusException as e:
            logging.error(
                f"Failed to insert data into collection '{self.config.collection}': {e}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data insertion: {e}")
            raise
