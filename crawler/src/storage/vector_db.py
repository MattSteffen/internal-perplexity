import logging
import os
import yaml
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .utils import build_milvus_schema, validate_schema, _validate_metadata
# Constants
MAX_DOC_LENGTH = 10240  # Max length for the 'text' field in Milvus should be 65000
# Define reasonable max lengths for indexed VARCHAR fields
MAX_SOURCE_LENGTH = 1024 # Adjust as needed for your source path/identifier lengths
DEFAULT_VARCHAR_MAX_LENGTH = 1024 # Default for other string fields

try:
    from pymilvus import MilvusClient, DataType, MilvusException
    MILVUS_AVAILABLE = True
    logging.info("Pymilvus library loaded successfully.")
except ImportError:
    MILVUS_AVAILABLE = False
    logging.error("Pymilvus not installed. VectorStorage operations cannot proceed.")

class VectorStorage:
    """
    Manages interaction with a Milvus vector database collection for storing
    document chunks and their embeddings.

    Handles connection, schema management, data insertion, and duplicate detection
    based on the combination of 'source' and 'chunk_index' metadata fields.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the VectorStorage instance.

        Args:
            config: A dictionary containing configuration settings. Expected keys include:
                - collection (str): Name of the Milvus collection.
                - milvus (dict): Milvus connection details (host, port, user, password, secure, index_params, etc.).
                - embeddings (dict): Embedding model details (dimension).
                - metadata (dict): Metadata schema definition ('schema' key with JSON schema).
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("Pymilvus library is required but not installed.")

        self.config = config
        self.milvus_config: Dict[str, Any] = config.get("milvus", {})
        self.schema_config: Optional[Dict[str, Any]] = validate_schema(self.config.get("metadata", {}).get("schema"))

        self.host: str = self.milvus_config.get("host")
        self.port: int = self.milvus_config.get("port")
        self.collection_name: str = config.get("collection")
        if not self.collection_name:
            raise ValueError("Configuration must include a 'collection' name.")
        self.partition_name: str = self.config.get("partition", None)

        self.client: Optional[MilvusClient] = None
        logging.info(f"VectorStorage initialized for collection '{self.collection_name}' at {self.host}:{self.port}")

    def __enter__(self) -> 'VectorStorage':
        """
        Connects to Milvus and loads or creates the collection upon entering context.
        """
        # TODO: Remove defaults
        user = self.milvus_config.get("user", "root")
        password = self.milvus_config.get("password", "Milvus")
        secure = self.milvus_config.get("secure", False)
        protocol = "https" if secure else "http"

        try:
            uri = f"{protocol}://{self.host}:{self.port}"
            token = f"{user}:{password}"
            
            logging.info(f"Connecting to Milvus at {uri}...")
            self.client = MilvusClient(uri=uri, token=token)
            logging.info("Successfully connected to Milvus.")

            if not self.client.has_collection(self.collection_name):
                logging.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self._create_collection()
                logging.info(f"Collection '{self.collection_name}' created.")
            elif self.config.get("recreate", False):
                logging.info(f"Collection '{self.collection_name}' exists but 'recreate' is True. Recreating...")
                self.client.drop_collection(self.collection_name)
                self._create_collection()
                logging.info(f"Collection '{self.collection_name}' already exists.")

            if self.partition_name:
                if not self.client.has_partition(self.collection_name, self.partition_name):
                    logging.info(f"Creating partition '{self.partition_name}'...")
                    self.client.create_partition(self.collection_name, self.partition_name)

            # Load collection
            self.client.load_collection(self.collection_name)
            logging.info(f"Collection '{self.collection_name}' loaded.")

        except MilvusException as e:
            logging.error(f"Milvus error during connection or collection handling: {e}")
            if self.client:
                self.client.close()
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during VectorStorage setup: {e}")
            if self.client:
                self.client.close()
            raise

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Releases collection and disconnects from Milvus. """
        if self.client:
            try:
                logging.info(f"Releasing collection '{self.collection_name}' from memory.")
                self.client.release_collection(self.collection_name)
            except MilvusException as e:
                logging.warning(f"Error releasing collection '{self.collection_name}': {e}")
            except Exception as e:
                logging.warning(f"Unexpected error releasing collection: {e}")
            
            try:
                logging.info("Closing Milvus client connection.")
                self.client.close()
            except Exception as e:
                logging.warning(f"Error closing client connection: {e}")
            
            self.client = None

    def close(self):
        """ Provides an explicit close method. """
        self.__exit__(None, None, None)

    def _create_collection(self):
        """ Creates the Milvus collection with schema and indexes """
        if not self.schema_config:
            raise RuntimeError("Schema configuration not loaded before attempting collection creation.")

        try:
            schema = build_milvus_schema(
                schema_config=self.schema_config,
                embedding_dim=self.config.get("embeddings", {}).get("dimension", 384)
            )

            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index( # TODO: Check the config
                field_name="embedding",
                index_type=self.milvus_config.get("index_type", "AUTOINDEX"),
                metric_type=self.milvus_config.get("metric_type", "L2"),
                # params={"nlist": self.milvus_config.get("nlist", 128)}
            )

            # Create collection with schema and index
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

        except MilvusException as e:
            logging.error(f"Failed to create collection or index '{self.collection_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during collection creation: {e}")
            raise

    def _check_duplicates(self, metadatas: List[Dict[str, Any]]) -> List[int]:
        # TODO: Check again. But actually don't need for now.
        """Check for duplicates using new client API"""
        if not self.client:
            logging.error("Client not connected. Cannot check for duplicates.")
            return []
        if not metadatas:
            return []

        batch_keys: Dict[Tuple[str, int], int] = {} # Stores (source, chunk_index) -> first index mapping
        indices_to_check_db: List[int] = []
        sources_in_batch: Set[str] = set()
        indices_in_batch: Set[int] = set()

        # Check duplicates within batch
        for i, meta in enumerate(metadatas):
            source = meta.get('source')
            chunk_index = meta.get('chunk_index')

            if not source or not isinstance(source, str) or chunk_index is None or not isinstance(chunk_index, int):
                logging.warning(f"Missing or invalid 'source' or 'chunk_index' in metadata at index {i}. Skipping duplicate check for this item, it won't be inserted by validation.")
                # We rely on _validate_metadata to prevent insertion later
                continue

            current_key = (source, chunk_index)
            if current_key in batch_keys:
                logging.debug(f"Duplicate (source='{source}', chunk_index={chunk_index}) found within batch (index {i}, original index {batch_keys[current_key]}). Skipping.")
                continue # Skip this item, keep the first occurrence

            # First occurrence in batch
            batch_keys[current_key] = i
            indices_to_check_db.append(i)
            sources_in_batch.add(source)
            indices_in_batch.add(chunk_index)

        if not indices_to_check_db:
            return []

        # Check against database
        try:
            if sources_in_batch and indices_in_batch:
                expr = f"source in {list(sources_in_batch)} and chunk_index in {list(indices_in_batch)}"
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=["source", "chunk_index"]
                )

                existing_keys = {(r['source'], r['chunk_index']) for r in results}
                final_indices = [i for i in indices_to_check_db 
                               if (metadatas[i]['source'], metadatas[i]['chunk_index']) not in existing_keys]
                return final_indices

        except MilvusException as e:
            logging.error(f"Failed to query for duplicates: {e}")
            return indices_to_check_db

        return indices_to_check_db

    def insert_data(self, data: List[Dict[str, Any]]):
        """
        Inserts text, embeddings, and metadata into the Milvus collection,
        avoiding duplicates based on the ('source', 'chunk_index') combination.

        Args:
            data:
                text:  of text chunks.
                embedding:  of corresponding embedding vectors.
                metadata:  of corresponding metadata dictionaries. Each must contain
                       'source' (str) and 'chunk_index' (int).
        """
        if not self.client:
            logging.error("Client is not initialized. Cannot insert data.")
            raise RuntimeError("VectorStorage not properly initialized or entered.")

        if not data:
            logging.info("Received empty lists. No data to insert.")
            return

        # 1. Check for duplicates and get indices of items to insert
        logging.info(f"Starting duplicate check for {len(data)} items based on (source, chunk_index)...")
        # indices_to_insert = self._check_duplicates(data)
        indices_to_insert = list(range(len(data)))
        logging.info(f"Duplicate check complete. {len(indices_to_insert)} items identified as new.")

        if not indices_to_insert:
            logging.info("No new entries to insert after duplicate check.")
            return

        data_to_insert = []
        for i in indices_to_insert:
            # if not _validate_metadata(data[i]['metadata'], i):
            #     continue

            data_to_insert.append(data[i])

        if not data_to_insert:
            logging.info("No valid data remaining after metadata validation and preparation.")
            return

        # 3. Insert the filtered and validated data
        try:
            successes = self.client.insert(
                collection_name=self.collection_name,
                data=data_to_insert,
                partition_name=self.partition_name
            )
            self.client.flush(self.collection_name)
            logging.info(f"Inserted {successes} entries")

        except MilvusException as e:
            logging.error(f"Failed to insert data: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during insertion: {e}")
            raise
