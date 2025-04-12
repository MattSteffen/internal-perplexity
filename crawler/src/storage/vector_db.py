import logging
import os
import yaml
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_DOC_LENGTH = 10240  # Max length for the 'text' field in Milvus
# Define reasonable max lengths for indexed VARCHAR fields
MAX_SOURCE_LENGTH = 1024 # Adjust as needed for your source path/identifier lengths
DEFAULT_VARCHAR_MAX_LENGTH = 1024 # Default for other string fields

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        MilvusException,
    )
    MILVUS_AVAILABLE = True
    logging.info("Pymilvus library loaded successfully.")
except ImportError:
    MILVUS_AVAILABLE = False
    logging.error("Pymilvus not installed. VectorStorage operations cannot proceed.")
    # Define dummy classes/functions if needed for basic structure compilation without Milvus
    # class Collection: pass
    # class MilvusException(Exception): pass
    # class connections:
    #     @staticmethod
    #     def connect(*args, **kwargs): pass
    #     @staticmethod
    #     def disconnect(*args, **kwargs): pass
    # class utility:
    #     @staticmethod
    #     def has_collection(name): return False
    #     @staticmethod
    #     def drop_collection(name): pass

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
        milvus_config: Dict[str, Any] = config.get("milvus", {})
        embeddings_config: Dict[str, Any] = config.get("embeddings", {})
        metadata_config: Dict[str, Any] = config.get("metadata", {})

        self.host: str = milvus_config.get("host", "localhost")
        self.port: int = milvus_config.get("port", 19530)
        self.collection_name: str = config.get("collection", "documents")
        if not self.collection_name:
            raise ValueError("Configuration must include a 'collection' name.")

        self.dim: int = embeddings_config.get("dimension", 384)
        self.schema_config: Optional[Dict[str, Any]] = metadata_config.get("schema")

        self.collection: Optional[Collection] = None
        self.metadata_fields: List[str] = [] # Populated after schema load

        logging.info(f"VectorStorage initialized for collection '{self.collection_name}' at {self.host}:{self.port}")

    def _load_or_create_schema(self) -> Dict[str, Any]:
        """Loads schema from config, performs basic validation."""
        if self.schema_config:
            logging.info("Using schema provided directly in configuration.")
            # Basic validation
            if not isinstance(self.schema_config, dict) or "properties" not in self.schema_config:
                 raise ValueError("Invalid schema configuration: must be a dictionary with a 'properties' key.")
            return self.schema_config
        else:
            logging.error("Schema configuration ('metadata.schema') is missing.")
            raise ValueError("Schema configuration must be provided in the config object.")

    def __enter__(self) -> 'VectorStorage':
        """
        Connects to Milvus and loads or creates the collection upon entering context.
        """
        milvus_config = self.config.get("milvus", {})
        user = milvus_config.get("user")
        password = milvus_config.get("password")
        secure = milvus_config.get("secure", False)
        alias = f"{self.collection_name}_alias_{os.getpid()}" # More unique alias

        try:
            logging.info(f"Connecting to Milvus at {self.host}:{self.port} with alias '{alias}'...")
            connections.connect(
                alias=alias,
                host=self.host,
                port=self.port,
                user=user,
                password=password,
                secure=secure,
                timeout=milvus_config.get("connect_timeout", 10), # Add timeout
            )
            logging.info("Successfully connected to Milvus.")

            self.schema_config = self._load_or_create_schema()
            # Extract expected metadata field names from the schema config
            # We will ensure 'source' and 'chunk_index' are handled by build_collection_schema
            base_properties = {"embedding", "text"}
            schema_props = self.schema_config.get("properties", {})
            self.metadata_fields = [
                name for name in schema_props.keys()
                if name not in base_properties
            ]
            # Ensure core fields used for logic are tracked if defined in schema
            if "source" not in self.metadata_fields and "source" in schema_props:
                 self.metadata_fields.append("source")
            if "chunk_index" not in self.metadata_fields and "chunk_index" in schema_props:
                 self.metadata_fields.append("chunk_index")
            # Add fields that build_collection_schema might add by default if missing
            if "source" not in self.metadata_fields:
                 self.metadata_fields.append("source")
            if "chunk_index" not in self.metadata_fields:
                 self.metadata_fields.append("chunk_index")
            partition = self.config.get("partition")
            if partition:
                partitions = [partition]
            else:
                partitions = []


            if utility.has_collection(self.collection_name, using=alias):
                logging.info(f"Collection '{self.collection_name}' already exists. Loading collection.")
                self.collection = Collection(self.collection_name, using=alias)
                # Optional: Add schema validation against existing collection here if needed
                # self._validate_existing_schema()

                logging.info(f"Loading collection '{self.collection_name}' into memory...")
                self.collection.load(partitions)
                logging.info(f"Collection '{self.collection_name}' loaded.")
            else:
                logging.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self._create_collection(alias)
                logging.info(f"Collection '{self.collection_name}' created.")
                # Load explicitly after creation and indexing
                logging.info(f"Loading newly created collection '{self.collection_name}' into memory...")
                self.collection.load()
                logging.info(f"Collection '{self.collection_name}' loaded.")


        except MilvusException as e:
            logging.error(f"Milvus error during connection or collection handling: {e}")
            try: connections.disconnect(alias)
            except Exception: pass
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during VectorStorage setup: {e}")
            try: connections.disconnect(alias)
            except Exception: pass
            raise

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Releases collection and disconnects from Milvus. """
        alias = f"{self.collection_name}_alias_{os.getpid()}"
        if self.collection:
            try:
                logging.info(f"Releasing collection '{self.collection_name}' from memory.")
                self.collection.release()
            except MilvusException as e:
                logging.warning(f"Error releasing collection '{self.collection_name}': {e}")
            except Exception as e:
                 logging.warning(f"Unexpected error releasing collection: {e}")
            self.collection = None

        try:
            # Check if connection exists before disconnecting
            if connections.has_connection(alias):
                logging.info(f"Disconnecting from Milvus alias '{alias}'.")
                connections.disconnect(alias)
            else:
                logging.debug(f"Milvus connection alias '{alias}' not found or already closed.")
        except MilvusException as e:
            logging.warning(f"Error disconnecting alias '{alias}': {e}")
        except Exception as e:
            logging.warning(f"Unexpected error during disconnection: {e}")

    def close(self):
        """ Provides an explicit close method. """
        self.__exit__(None, None, None)

    def _create_collection(self, alias: str):
        """ Creates the Milvus collection, including essential and configured fields, and indexes. """
        if not self.schema_config:
             raise RuntimeError("Schema configuration not loaded before attempting collection creation.")

        milvus_config = self.config.get("milvus", {})
        try:
            logging.info(f"Building collection schema for '{self.collection_name}'...")
            schema = build_collection_schema(self.schema_config, self.dim)

            logging.info(f"Creating collection '{self.collection_name}'...")
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=alias,
                consistency_level=milvus_config.get("consistency_level", "Bounded")
            )
            logging.info(f"Collection '{self.collection_name}' created.")

            # --- Create ONLY the Vector Index ---
            index_field = milvus_config.get("index_field", "embedding")
            embedding_index_name = f"{index_field}_idx" # Define the name we will use
            index_params = milvus_config.get("index_params", {
                "index_type": "AUTOINDEX",
                "metric_type": milvus_config.get("metric_type", "L2"),
                "params": {}
            })
            logging.info(f"Creating vector index on field '{index_field}' with name '{embedding_index_name}' and params: {index_params}")
            self.collection.create_index(
                index_field,
                index_params,
                index_name=embedding_index_name # Specify the index name here
            )
            logging.info(f"Vector index creation initiated for collection '{self.collection_name}'.")

            # --- REMOVED Scalar Index Creation ---
            # The following blocks attempting to create indexes on 'source' and 'chunk_index'
            # have been removed as per the requirement to only index 'embedding'.
            # try:
            #      logging.info("Creating index on scalar field 'source' for faster filtering.")
            #      self.collection.create_index("source", index_name="source_idx")
            #      logging.info("Scalar index created on 'source'.")
            # except MilvusException as ie:
            #       logging.warning(f"Could not create scalar index on 'source': {ie}. Duplicate checks might be slow.")
            # # ... and similar block for chunk_index removed ...

            # --- Wait Specifically for the Embedding Index ---
            logging.info(f"Waiting for index '{embedding_index_name}' creation to complete...")
            utility.wait_for_index_building_complete(
                self.collection_name,
                index_name=embedding_index_name, # <--- Specify the index name here
                using=alias
            )
            # Optional: Add a timeout to the wait function
            # utility.wait_for_index_building_complete(
            #     self.collection_name,
            #     index_name=embedding_index_name,
            #     timeout=milvus_config.get("index_wait_timeout", 300), # e.g., 5 minutes
            #     using=alias
            # )
            logging.info(f"Index '{embedding_index_name}' building complete.")


        except MilvusException as e:
            logging.error(f"Failed to create collection or index '{self.collection_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during collection creation: {e}")
            raise


    def _check_duplicates(self, metadatas: List[Dict[str, Any]]) -> List[int]:
        """
        Checks for duplicate entries based on the combination of 'source' and 'chunk_index'.
        Filters both within the batch and against existing records in the collection.

        Args:
            metadatas: A list of metadata dictionaries for the batch. Each dict
                       must contain 'source' and 'chunk_index' keys.

        Returns:
            A list of indices from the original batch that represent non-duplicate entries
            and should be inserted.
        """
        if not self.collection:
             logging.error("Collection not loaded. Cannot check for duplicates.")
             return []
        if not metadatas:
            return []

        batch_keys: Dict[Tuple[str, int], int] = {} # Stores (source, chunk_index) -> first index mapping
        indices_to_check_db: List[int] = []
        keys_to_check_db: Set[Tuple[str, int]] = set()
        sources_in_batch: Set[str] = set()
        indices_in_batch: Set[int] = set()

        # --- Step 1: Check for duplicates within the batch ---
        for i, meta in enumerate(metadatas):
            source = meta.get('source')
            chunk_index = meta.get('chunk_index')

            # Validate presence and type (basic check, more thorough in _validate_metadata)
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
            keys_to_check_db.add(current_key)
            sources_in_batch.add(source)
            # Ensure chunk_index is hashable (int is fine)
            indices_in_batch.add(chunk_index)


        if not indices_to_check_db:
            logging.info("No valid items found in the batch after initial filtering/batch duplicate check.")
            return []

        # --- Step 2: Check against existing records in Milvus ---
        existing_keys = set()
        try:
            # Query strategy: Fetch records where source is in batch sources AND chunk_index is in batch indices.
            # This is generally more efficient than multiple OR clauses for individual pairs.
            # We will filter the results in Python afterwards.
            if sources_in_batch and indices_in_batch:
                # Ensure lists are not empty before querying
                source_list = list(sources_in_batch)
                index_list = list(indices_in_batch)

                # Escape quotes in source strings if they might contain them
                escaped_source_list = [s.replace("'", "''").replace('"', '""') for s in source_list]

                # Format lists for the 'in' operator
                source_expr = "','".join(escaped_source_list)
                index_expr = ",".join(map(str, index_list))

                # Construct the query expression
                query_expr = f"source in ['{source_expr}'] and chunk_index in [{index_expr}]"
                logging.debug(f"Querying Milvus for existing source/chunk_index pairs with expression: {query_expr} (checking {len(keys_to_check_db)} unique keys)")

                # Adjust consistency level based on config
                consistency_level = self.config.get("milvus", {}).get("consistency_level", "Bounded")

                results = self.collection.query(
                    expr=query_expr,
                    output_fields=["source", "chunk_index"], # Only need these fields
                    consistency_level=consistency_level
                )

                # Build a set of existing (source, chunk_index) tuples from the results
                for res in results:
                    existing_keys.add((res['source'], res['chunk_index']))

                logging.debug(f"Found {len(existing_keys)} existing (source, chunk_index) pairs in Milvus matching the batch's sources and indices.")

        except MilvusException as e:
            logging.error(f"Failed to query Milvus for duplicate check: {e}. Proceeding without DB check (potential duplicates).")
            # Return indices that passed the batch check, risking duplicates
            return indices_to_check_db
        except Exception as e:
            logging.error(f"Unexpected error during duplicate query: {e}. Proceeding without DB check.")
            return indices_to_check_db


        # --- Step 3: Filter potential indices based on the query results ---
        final_indices_to_insert = []
        for i in indices_to_check_db:
            meta = metadatas[i]
            # We already validated source/chunk_index exist in the first loop
            source = meta['source']
            chunk_index = meta['chunk_index']
            current_key = (source, chunk_index)

            if current_key in existing_keys:
                logging.info(f"Skipping insert for already existing item: (source='{source}', chunk_index={chunk_index}) (batch index {i})")
            else:
                final_indices_to_insert.append(i)

        return final_indices_to_insert

    def _validate_metadata(self, metadata: Dict[str, Any], index: int) -> bool:
        """
        Validates a single metadata dictionary. Ensures mandatory fields
        ('source', 'chunk_index') are present and correctly typed.

        Args:
            metadata: The metadata dictionary to validate.
            index: The original batch index for logging purposes.

        Returns:
            True if metadata is considered valid for insertion, False otherwise.
        """
        # --- Mandatory Field Check ---
        source = metadata.get('source')
        chunk_index = metadata.get('chunk_index')

        if not source or not isinstance(source, str):
             logging.error(f"Mandatory metadata field 'source' missing or not a string in item at index {index}. Skipping insertion.")
             return False
        # Add length check for source if necessary based on schema
        if len(source) > MAX_SOURCE_LENGTH:
             logging.warning(f"Metadata field 'source' exceeds max length ({MAX_SOURCE_LENGTH}) at index {index}. It might be truncated by Milvus or cause errors.")
             # Decide if truncation is acceptable or if it should fail validation
             # return False # Uncomment to reject long sources

        if chunk_index is None or not isinstance(chunk_index, int):
             logging.error(f"Mandatory metadata field 'chunk_index' missing or not an integer in item at index {index}. Skipping insertion.")
             return False

        # --- Optional: Field Existence and Type Check (based on self.metadata_fields) ---
        for field_name in self.metadata_fields:
             if field_name not in metadata:
                  # Check if field is required in the original schema? (more complex)
                  logging.debug(f"Optional metadata field '{field_name}' missing in item at index {index}.")
             else:
                  # Optional: Add type checking based on self.schema_config
                  pass

        return True


    def insert_data(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Inserts text, embeddings, and metadata into the Milvus collection,
        avoiding duplicates based on the ('source', 'chunk_index') combination.

        Args:
            texts: List of text chunks.
            embeddings: List of corresponding embedding vectors.
            metadatas: List of corresponding metadata dictionaries. Each must contain
                       'source' (str) and 'chunk_index' (int).
        """
        if not self.collection:
            logging.error("Collection is not initialized. Cannot insert data.")
            raise RuntimeError("VectorStorage not properly initialized or entered.")

        if not (len(texts) == len(embeddings) == len(metadatas)):
            logging.error(f"Input list length mismatch: texts({len(texts)}), embeddings({len(embeddings)}), metadatas({len(metadatas)})")
            raise ValueError("Input lists (texts, embeddings, metadatas) must have the same length.")

        if not texts:
            logging.info("Received empty lists. No data to insert.")
            return

        # 1. Check for duplicates and get indices of items to insert
        logging.info(f"Starting duplicate check for {len(metadatas)} items based on (source, chunk_index)...")
        indices_to_insert = self._check_duplicates(metadatas)
        logging.info(f"Duplicate check complete. {len(indices_to_insert)} items identified as new.")

        if not indices_to_insert:
            logging.info("No new entries to insert after duplicate check.")
            return

        # 2. Prepare data for insertion, validating metadata along the way
        data_to_insert: List[Dict[str, Any]] = []
        prep_errors = 0
        for i in indices_to_insert:
            metadata = metadatas[i]

            # 2a. Validate metadata for the item (ensures source/chunk_index are valid)
            if not self._validate_metadata(metadata, i):
                 logging.warning(f"Skipping item at original index {i} due to invalid metadata after duplicate check.")
                 prep_errors += 1
                 continue # Skip this item

            # 2b. Prepare the data dictionary for Milvus
            field_values = {}

            # Add core fields
            field_values["embedding"] = embeddings[i]
            field_values["text"] = texts[i][:MAX_DOC_LENGTH] # Truncate text

            # Add required fields explicitly (validation ensures they exist)
            field_values["source"] = metadata["source"]
            field_values["chunk_index"] = metadata["chunk_index"]

            # Add other metadata fields defined in the schema config
            schema_properties = self.schema_config.get("properties", {})
            for field_name in self.metadata_fields:
                 # Avoid overwriting core/required fields already added
                 if field_name not in ["embedding", "text", "source", "chunk_index"]:
                     if field_name in metadata:
                          # Basic type check/conversion could be added here if needed
                          field_values[field_name] = metadata[field_name]
                     else:
                          # Handle missing optional fields (e.g., set to None or default)
                          field_values[field_name] = None # Or get default from schema?


            # Add any other fields from the metadata that are *also* in the schema's properties
            # This handles fields maybe not captured perfectly by self.metadata_fields earlier
            for key, value in metadata.items():
                 if key in schema_properties and key not in field_values:
                      field_values[key] = value

            data_to_insert.append(field_values)

        if prep_errors > 0:
            logging.warning(f"{prep_errors} items were skipped during data preparation due to validation errors.")

        if not data_to_insert:
            logging.info("No valid data remaining after metadata validation and preparation.")
            return

        # 3. Insert the filtered and validated data
        try:
            logging.info(f"Attempting to insert {len(data_to_insert)} new entries into collection '{self.collection_name}'...")
            # Optionally specify partition if configured
            partition_name = self.config.get('partition', None)
            
            # Create partition if it doesn't exist and is specified
            if partition_name:
                if not utility.has_partition(self.collection_name, partition_name):
                    logging.info(f"Creating partition '{partition_name}' in collection '{self.collection_name}'")
                    self.collection.create_partition(partition_name)
                
            insert_result = self.collection.insert(data_to_insert, partition_name=partition_name)
            # Flush is important to make data searchable/persistent immediately
            logging.info("Flushing collection to make inserts visible...")
            self.collection.flush()
            logging.info(f"Successfully inserted {len(insert_result.primary_keys)} new entries (PKs: {insert_result.primary_keys[:5]}...). Flushed collection.")

        except MilvusException as e:
            logging.error(f"Failed to insert data into Milvus collection '{self.collection_name}': {e}")
            # Consider logging problematic data if possible (might be large)
            # logging.debug(f"Data attempted: {data_to_insert[:1]}") # Log first item for debugging
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during data insertion: {e}")
            raise


# --- Helper Functions ---

# load_schema_config remains the same

def build_collection_schema(schema_config: Dict[str, Any], default_dim: int) -> CollectionSchema:
    """
    Builds a Milvus CollectionSchema based on the provided schema configuration dictionary.
    Automatically includes 'id', 'embedding', 'text'.
    Ensures 'source' and 'chunk_index' fields exist for duplicate checking.

    Args:
        schema_config: The schema configuration dictionary (e.g., loaded from YAML).
        default_dim: The dimension for the 'embedding' vector field.

    Returns: A Pymilvus CollectionSchema object.
    Raises: ValueError: If schema_config is invalid or unsupported types are found.
    """
    if not isinstance(schema_config, dict) or "properties" not in schema_config:
         raise ValueError("Invalid schema_config provided. Must be a dict with 'properties'.")

    fields_config = schema_config.get("properties", {})
    schema_description = schema_config.get("description", "Collection storing document chunks and embeddings")

    # --- Standard Required Fields ---
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description="Auto-generated unique record ID"),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=default_dim, description="Dense vector embedding"),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=MAX_DOC_LENGTH, description="Original text chunk"),
    ]
    processed_field_names = {'id', 'embedding', 'text'}

    # --- Ensure Core Fields for Duplicate Check Exist ---
    # Check if 'source' is defined in the config
    if 'source' not in fields_config:
        logging.warning("Schema config missing 'source' field. Adding default VARCHAR field for duplicate checking.")
        fields.append(FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=MAX_SOURCE_LENGTH, description="Source identifier (e.g., file path)"))
        processed_field_names.add('source')
    # Check if 'chunk_index' is defined in the config
    if 'chunk_index' not in fields_config:
        logging.warning("Schema config missing 'chunk_index' field. Adding default INT64 field for duplicate checking.")
        fields.append(FieldSchema(name='chunk_index', dtype=DataType.INT64, description="Index of the chunk within the source"))
        processed_field_names.add('chunk_index')

    # --- Add Fields from Schema Config ---
    for field_name, field_def in fields_config.items():
        if field_name in processed_field_names:
            logging.debug(f"Skipping field '{field_name}' from schema config as it's already handled.")
            continue

        if not isinstance(field_def, dict) or "type" not in field_def:
             logging.warning(f"Skipping invalid field definition for '{field_name}': {field_def}")
             continue

        field_type = field_def["type"]
        field_description = field_def.get("description", f"Metadata field: {field_name}")

        try:
            # Handle different types based on schema config
            if field_name == 'source': # Handle source explicitly if defined in config
                 max_length = field_def.get("maxLength", MAX_SOURCE_LENGTH)
                 fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=field_description))
            elif field_name == 'chunk_index': # Handle chunk_index explicitly if defined
                 fields.append(FieldSchema(name=field_name, dtype=DataType.INT64, description=field_description))
            elif field_type == "string":
                max_length = field_def.get("maxLength", DEFAULT_VARCHAR_MAX_LENGTH)
                fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=field_description))
            elif field_type == "integer":
                 fields.append(FieldSchema(name=field_name, dtype=DataType.INT64, description=field_description))
            elif field_type == "float":
                 fields.append(FieldSchema(name=field_name, dtype=DataType.DOUBLE, description=field_description))
            elif field_type == "boolean":
                 fields.append(FieldSchema(name=field_name, dtype=DataType.BOOL, description=field_description))
            elif field_type == "array": # Simple array handling (store as string)
                 max_length = field_def.get("maxLength", 2048)
                 logging.warning(f"Mapping array field '{field_name}' to VARCHAR({max_length}). Ensure data is serialized.")
                 fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length, description=f"{field_description} (serialized)"))
            elif field_type == "float_vector":
                 logging.warning(f"Schema config requests additional float_vector field '{field_name}'. Skipping.")
                 continue
            else:
                raise ValueError(f"Unsupported field type '{field_type}' for field '{field_name}'.")

            processed_field_names.add(field_name)
            logging.debug(f"Added field '{field_name}' of type {field_type} from schema config.")

        except Exception as e:
             logging.error(f"Error processing schema field '{field_name}': {e}")
             # Decide if this should halt schema creation
             # raise

    logging.info(f"Built collection schema with {len(fields)} fields: {[f.name for f in fields]}")
    return CollectionSchema(fields=fields, description=schema_description, enable_dynamic_field=False) # Consider dynamic fields if needed


# import logging
# from typing import List, Dict, Any, Optional
# import numpy as np
# import re 
# import yaml 
# from uuid import uuid4

# MAX_DOC_LENGTH = 10240

# try:
#     from pymilvus import (
#         connections,
#         utility,
#         FieldSchema,
#         CollectionSchema,
#         DataType,
#         Collection,
#     )
#     MILVUS_AVAILABLE = True
# except ImportError:
#     MILVUS_AVAILABLE = False
#     logging.warning("Pymilvus not installed. Using mock implementation for development.")

# class VectorStorage:
#     def __init__(self, config: dict = {}):
#         # Now use nested config sections for various parameters.
#         self.config = config
#         milvus_config = config.get("milvus", {})
#         embeddings_config = config.get("embeddings", {})

#         self.host = milvus_config.get("host", "localhost")
#         self.port = milvus_config.get("port", 19530)
#         # Use top-level "collection" key for the collection name.
#         self.collection_name = config.get("collection", "documents")
#         self.dim = embeddings_config.get("dimension", 384)
#         # If you want read_only to be controlled separately, you can either leave it top-level
#         # or add another key (for example under "vector_db"). Here we assume top-level.
#         self.read_only = config.get("read_only", False)
#         self.collection = None

#         # Milvus
#         self.client = None

#     def __enter__(self):
#         milvus_config = self.config.get("milvus", {})
#         # Configure with security credentials from the "milvus" section.
#         user = milvus_config.get("user")
#         password = milvus_config.get("password")
#         secure = milvus_config.get("secure", False)
#         if user and password:
#             connections.connect(host=self.host, port=self.port, user=user, password=password, secure=secure)
#         else:
#             connections.connect(host=self.host, port=self.port)
        
#         if not utility.has_collection(self.collection_name):
#             self._create_collection()
#         else:
#             # self.collection = Collection(self.collection_name) # TODO: This is a bug, don't delete every time.
#             utility.drop_collection(self.collection_name)
#             self._create_collection()
#             self.collection.load()
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         if self.collection:
#             self.collection.release()
#         # Disconnect using the collection name.
#         connections.disconnect(self.collection_name)

#     def close(self):
#         self.__exit__(None, None, None)

#     def _create_collection(self):
#         # Get the schema config from the "metadata" section.
#         schema_config = self.config.get("metadata", {}).get("schema")
#         if not schema_config:
#             schema_config = load_schema_config(self.schema_config_path)
#         # Also pass milvus-specific config to the collection creation helper.
#         milvus_config = self.config.get("milvus", {})
#         self.collection = create_collection(schema_config, self.collection_name, self.dim, milvus_config)

#     def insert_data(self, texts: list, embeddings: list, metadatas: list):
#         # TODO: Confirm that the schema is generated correctly.
#         # TODO: Confirm that the metadata matches the schema
#         """
#         Inserts data into Milvus collection, ensuring no duplicate records
#         (based on a unique combination of "source" and "chunk_index") are inserted.
#         """
#         if self.read_only:
#             print("Storage is in read-only mode. Insert operation skipped.")
#             return

#         if not (len(texts) == len(embeddings) == len(metadatas)):
#             raise ValueError("All input lists must have the same length")
        
#         # TODO: Make sure not to insert duplicates
        
#         # new_entries = []
#         # for i in range(len(texts)):
#         #     meta = metadatas[i]
#         #     source = meta.get('source', '')
#         #     chunk_index = meta.get('chunk_index')
#         #     if chunk_index is None:
#         #         raise ValueError("Each metadata dict must include a 'chunk_index' field")
#         #     new_entries.append((source, chunk_index))
        
#         # seen = set()
#         # indices_to_check = []
#         # for i, key in enumerate(new_entries):
#         #     print(f"Checking key: {key}")
#         #     if key not in seen:
#         #         seen.add(key)
#         #         indices_to_check.append(i)
#         #     else:
#         #         print(f"Skipping duplicate within batch for key {key}")
        
#         # print("lens: ", len(seen), len(indices_to_check), len(texts))

#         # if len(indices_to_check) == 0:
#         #     print("No new entries to insert (all are duplicates in batch).")
#         #     return
        
#         # filter_clauses = []
#         # for key in seen:
#         #     s, ci = key
#         #     filter_clauses.append(f'(source == "{s}" and chunk_index == {ci})')
#         # filter_expr = " or ".join(filter_clauses)
        
#         # existing_records = self.collection.query(expr=filter_expr, output_fields=["source", "chunk_index"])
#         # existing_keys = {(rec["source"], rec["chunk_index"]) for rec in existing_records}
        
#         # final_indices = []
#         # for i in indices_to_check:
#         #     key = new_entries[i]
#         #     if key in existing_keys:
#         #         print(f"Skipping duplicate existing entry for key {key}")
#         #     else:
#         #         final_indices.append(i)
        
#         # if not final_indices:
#         #     print("No new entries to insert after duplicate check.")
#         #     return

#         # texts_to_insert = [texts[i] for i in final_indices]
#         # embeddings_to_insert = [embeddings[i] for i in final_indices]
#         final_indexes = list(range(len(texts)))

#         # Get metadata fields from the "metadata" schema.
#         metadata_schema = self.config.get("metadata", {}).get("schema", {})
#         metadata_fields = list(metadata_schema.get("properties", {}).keys())
        
#         data: list[dict] = []
#         for i in final_indexes:
#             field_values = {field: metadatas[i].get(field, "") for field in metadata_fields} # only use the fields defined in the schema
#             field_values["embedding"] = embeddings[i]
#             field_values["text"] = texts[i][:MAX_DOC_LENGTH]
#             data.append(field_values)
        

#         # Optionally, if a partition is defined at the top level, retrieve it.
#         self.collection.insert(data, partition_name=self.config.get('partition', None))
#         self.collection.flush()
#         print(f"Inserted {len(texts)} new chunks.")






# def load_schema_config(config_file: str) -> dict:
#     """Loads the collection schema configuration from a YAML file."""
#     with open(config_file, "r") as f:
#         config = yaml.safe_load(f)
#     return config



# def build_collection_schema(schema_config: dict, default_dim: int = 384) -> CollectionSchema:
#     # TODO: I should include the description of the fields in the schema. These should be found in the schema file.
#     """
#     Builds a CollectionSchema based on the provided schema configuration.
#     The passed schema_config is now the actual JSON schema (not nested under "schema").
#     """
#     # Now access properties directly from the schema_config.
#     fields_config = schema_config.get("properties", {})
#     # Always add the auto-generated id field.
#     fields = [
#         FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=default_dim),
#         FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=MAX_DOC_LENGTH),
#     ]
    
#     for field_name, field_def in fields_config.items():
#         field_type = field_def["type"]
#         if field_type == "string":
#             if field_name != "text": # Not allowed to duplicate text
#                 max_length = field_def.get("maxLength", 1024)
#                 fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length))
#         elif field_type == "integer":
#             fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
#         elif field_type == "float_vector":
#             if field_name != "embedding": # Not allowed to duplicate embedding
#                 dim = field_def.get("dim", default_dim)
#                 fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim))
#         elif field_type == "array":
#             max_length = field_def.get("maxLength", 1024)
#             fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length))
#         else:
#             raise ValueError(f"Unsupported field type: {field_type}")
    
#     description = schema_config.get("description", "Document chunks with embeddings")

#     return CollectionSchema(fields, description=description)

# def create_collection(schema_config: dict, collection_name: str, default_dim: int, milvus_config: dict) -> Collection:
#     """
#     Creates and returns a new Milvus collection based on the schema configuration.
#     The milvus_config is used to obtain index parameters.
#     """
#     schema = build_collection_schema(schema_config, default_dim)
#     collection = Collection(collection_name, schema)
    
#     # Get index details from the milvus config.
#     index_field = milvus_config.get("index_field", "embedding")
#     index_params = milvus_config.get("index_params", {
#         "index_type": "IVF_FLAT",
#         "metric_type": "L2",
#         "params": {"nlist": 128}
#     })
#     collection.create_index(index_field, index_params)
    
#     return collection


