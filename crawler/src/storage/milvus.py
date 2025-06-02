import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple

"""
The goal of this class:
- Connect to Milvus
- Create a collection (and or partition) if it doesn't exist
- Create a base schema (always the same), and a schema provided by the user.
- It needs to be able to insert data into the collection
    - Avoiding duplicates (measured by source and chunk_index)
"""

from pymilvus import (
    MilvusClient,
    MilvusException,
)



from .milvus_utils import create_schema, create_index



class MilvusStorage:
    """
    Manages interaction with a Milvus vector database collection for storing
    document chunks and their embeddings.
    """

    def __init__(self, milvus_config: Dict[str, Any], dimension: int, metadata_schema: Dict[str, Any]):
        """
        Initializes the MilvusStorage instance.
        """

        self.milvus_config = milvus_config
        self.host = self.milvus_config.get("host")
        self.port = self.milvus_config.get("port")
        self.username = self.milvus_config.get("username")
        self.password = self.milvus_config.get("password")
        self.collection_name = self.milvus_config.get("collection")
        self.partition_name = self.milvus_config.get("partition", None)

        self.recreate = self.milvus_config.get("recreate", False)
        self.dimension = dimension
        self.metadata_schema = metadata_schema if metadata_schema is not None else {}

        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", token=f"{self.username}:{self.password}")
        # test the connection
        try:
            self.client.list_collections()
            print("Connected to Milvus!")
        except MilvusException as e:
            logging.error(f"Error connecting to Milvus: {e}")
            raise e
        
        self.create_collection(self.dimension, self.metadata_schema)
        
        
    def create_collection(self, embedding_size: int, schema: Dict[str, Any]) -> None:
        """
        Creates a new collection with the specified schema.
        """
        if self.client.has_collection(self.collection_name):
           if self.recreate:
               self.client.drop_collection(self.collection_name)
           
        if not self.client.has_collection(self.collection_name):
            self._create_collection(embedding_size, schema)
        
    

    def _create_collection(self, embedding_size: int, schema: Dict[str, Any]) -> None:
        """
        Creates a new collection with the specified schema.
        """
        collection_schema = create_schema(embedding_size, schema)
        index = create_index(self.client)
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=embedding_size,
            schema=collection_schema,
            index_params=index,
            vector_field_name="text_embedding",
            auto_id=True,
        )

        if self.partition_name:
            self.client.create_partition(self.collection_name, self.partition_name)


    def _check_duplicate(self, source: str, chunk_index: int) -> bool:
        """
        Check if a document with the given source and chunk_index already exists.
        
        Args:
            source: The source identifier
            chunk_index: The chunk index
            
        Returns:
            bool: True if duplicate exists, False otherwise
        """
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"source == '{source}' AND chunk_index == {chunk_index}",
                output_fields=["source"],
                limit=1
            )
            return len(results) > 0
            
        except MilvusException as e:
            logging.error(f"Failed to check for duplicate (source: {source}, chunk_index: {chunk_index}): {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error checking for duplicate: {e}")
            raise

    def insert_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Inserts data into the collection.
        
        Expected data format:
        [
            {
                "text": "content text",
                "embedding": [0.1, 0.2, ...],  # vector embedding
                "chunk_index": 0,
                "source": "filename",
                "minio": optional - the url recieved when inserting document into minio
                "other_field1": "value1",  # will go into metadata
                "other_field2": "value2",  # will go into metadata
                ...
            },
            ...
        ]
        """
        if not data:
            logging.info("Received empty data list. No data to insert.")
            return

        # Filter out duplicates and prepare data for insertion
        data_to_insert = []
        duplicates_found = 0
        
        for item in data:
            # Validate required fields
            required_fields = ["text", "text_embedding", "chunk_index", "source"]
            missing_fields = [field for field in required_fields if field not in item]
            
            if missing_fields:
                logging.warning(f"Skipping item missing required fields: {missing_fields}")
                continue
            
            source = item["source"]
            chunk_index = item["chunk_index"]
            
            # Check for duplicates
            if self._check_duplicate(source, chunk_index):
                duplicates_found += 1
                logging.debug(f"Duplicate found for source: {source}, chunk_index: {chunk_index}. Skipping.")
                continue
            
            # Prepare the data for insertion
            # Extract required fields
            # TODO: Document ID should be set prior to insert.
            # "document_id": str(uuid.uuid4()),  # Generate a unique document ID
            prepared_item = {
                "document_id": item.get("document_id", str(uuid.uuid4())),
                "text": item["text"],
                "text_embedding": item["text_embedding"],
                "chunk_index": chunk_index,
                "source": source,
                "minio": item.get("minio", ""),
            }
            
            # Create metadata from remaining fields
            metadata = {}
            for key, value in item.items():
                if key not in required_fields + ["minio"]:
                    metadata[key] = value
                    prepared_item[key] = value
            
            # Add metadata as JSON string (not pretty printed)
            prepared_item["metadata"] = json.dumps(metadata, separators=(',', ':'))
            
            data_to_insert.append(prepared_item)

        if duplicates_found > 0:
            logging.info(f"Found {duplicates_found} duplicates. Skipping insertion for these items.")

        if not data_to_insert:
            logging.info("No new data to insert after duplicate check and validation.")
            return

        # Insert the data
        try:
            logging.info(f"Inserting {len(data_to_insert)} new items into collection '{self.collection_name}'")
            
            result = self.client.insert(
                collection_name=self.collection_name,
                partition_name=self.partition_name,
                data=data_to_insert,
            )
            
            # Flush to ensure data is persisted
            self.client.flush(self.collection_name)

            logging.info(f"Successfully inserted {result.get("insert_count", -1)} items into collection '{self.collection_name}'")
            
        except MilvusException as e:
            logging.error(f"Failed to insert data into collection '{self.collection_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data insertion: {e}")
            raise