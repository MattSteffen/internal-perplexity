# Connect to Milvus

This folder contains the code that connects to Milvus. Allows the creation of new collections provided a configured yaml file.

Sample modifications to current code for future:

```python
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import re
import yaml
import os
from datetime import datetime

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("Pymilvus not installed. Using mock implementation for development.")

class VectorStorage:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the VectorStorage with configuration.

        Args:
            config_path (str): Path to the configuration file containing both
                              connection and collection settings.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Connection settings
        connection_config = self.config.get("connection", {})
        self.host = connection_config.get("host", "localhost")
        self.port = connection_config.get("port", "19530")
        self.user = connection_config.get("user", "")
        self.password = connection_config.get("password", "")
        self.use_secure = connection_config.get("use_secure", False)
        self.connection_timeout = connection_config.get("timeout", 10)

        # Collection settings
        collection_config = self.config.get("collection", {})
        self.collection_name = collection_config.get("name", "documents")
        self.dim = collection_config.get("embedding_dim", 384)

        # Security settings
        security_config = self.config.get("security", {})
        self.enable_encryption = security_config.get("enable_encryption", False)
        self.encryption_key = security_config.get("encryption_key", "")
        self.access_control = security_config.get("access_control", [])

        self.collection = None
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def __enter__(self):
        """Connect to Milvus and load or create the collection when entering context."""
        self._connect()

        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release collection and disconnect when exiting context."""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")

    def _connect(self):
        """Establish connection to Milvus with configured parameters."""
        connect_params = {
            "host": self.host,
            "port": self.port,
        }

        # Add security parameters if enabled
        if self.use_secure:
            connect_params["secure"] = True

        # Add authentication if provided
        if self.user and self.password:
            connect_params["user"] = self.user
            connect_params["password"] = self.password

        # Add timeout
        if self.connection_timeout:
            connect_params["timeout"] = self.connection_timeout

        self.logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
        connections.connect("default", **connect_params)

    def _create_collection(self):
        """Create a new collection based on schema configuration."""
        schema_config = self.config.get("schema", {})
        if not schema_config:
            schema_path = self.config.get("collection", {}).get("schema_path", "milvus_schema.yaml")
            if os.path.exists(schema_path):
                schema_config = load_schema_config(schema_path)
            else:
                self.logger.error(f"Schema file {schema_path} not found.")
                raise FileNotFoundError(f"Schema file {schema_path} not found.")

        self.collection = create_collection(
            schema_config,
            self.collection_name,
            self.dim,
            security_config=self.config.get("security", {})
        )

    def insert_data(self, texts: list, embeddings: list, metadatas: list):
        """
        Inserts data into Milvus collection, ensuring no duplicate records
        (based on a unique combination of "source" and "chunk_index") are inserted.

        Args:
            texts (list): List of text chunks.
            embeddings (list): List of embeddings for each chunk.
            metadatas (list): List of metadata dictionaries. Each metadata **must**
                              include the keys "source" and "chunk_index".
        """
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length")

        # Build a list of unique keys from new records: (source, chunk_index)
        new_entries = []
        for i in range(len(texts)):
            meta = metadatas[i]
            source = meta.get('source', '')
            chunk_index = meta.get('chunk_index')
            if chunk_index is None:
                raise ValueError("Each metadata dict must include a 'chunk_index' field")
            new_entries.append((source, chunk_index))

        # Remove duplicates within the new batch
        seen = set()
        indices_to_check = []
        for i, key in enumerate(new_entries):
            self.logger.debug(f"Checking key: {key}")
            if key not in seen:
                seen.add(key)
                indices_to_check.append(i)
            else:
                self.logger.debug(f"Skipping duplicate within batch for key {key}")

        self.logger.debug(f"lens: {len(seen)}, {len(indices_to_check)}, {len(texts)}")

        # If all entries are duplicates, return early
        if len(indices_to_check) == 0:
            self.logger.info("No new entries to insert (all are duplicates in batch).")
            return

        # Build a filter expression to query for existing records.
        filter_clauses = []
        for key in seen:
            s, ci = key
            filter_clauses.append(f'(source == "{s}" and chunk_index == {ci})')
        filter_expr = " or ".join(filter_clauses)

        # Query the collection for existing entries that match any of these keys.
        existing_records = self.collection.query(expr=filter_expr, output_fields=["source", "chunk_index"])
        existing_keys = {(rec["source"], rec["chunk_index"]) for rec in existing_records}

        # Determine final indices to insert (exclude any already existing records)
        final_indices = []
        for i in indices_to_check:
            key = new_entries[i]
            if key in existing_keys:
                self.logger.debug(f"Skipping duplicate existing entry for key {key}")
            else:
                final_indices.append(i)

        if not final_indices:
            self.logger.info("No new entries to insert after duplicate check.")
            return

        # Get field names and types from schema config
        schema_config = self.config.get("schema", {})
        fields_config = schema_config.get("fields", [])
        field_names = [field["name"] for field in fields_config if field["name"] != "id" and field["name"] != "embedding"]

        # Prepare data for insertion using dynamic field mapping
        data = []
        # Always add text and embedding as required fields
        data.append([texts[i] for i in final_indices])  # text
        data.append([embeddings[i] for i in final_indices])  # embedding

        # Add all other fields from metadata based on schema
        for field_name in field_names:
            if field_name != "text":  # text is already added
                data.append([metadatas[i].get(field_name, '') for i in final_indices])

        # Add security metadata if enabled
        if self.enable_encryption:
            # Add timestamp for security auditing
            timestamp = datetime.now().isoformat()
            if "insert_timestamp" in field_names:
                data.append([timestamp for _ in final_indices])

        self.collection.insert(data)
        self.collection.flush()
        self.logger.info(f"Inserted {len(final_indices)} new chunks.")

    def search(self, query_embedding: list, limit: int = 5, filters: list[str] = []) -> list:
        """
        Search for similar chunks using a query embedding.
        Optionally filter by a list of authors or other valid filters.

        Args:
            query_embedding (list): The query vector. If empty or None, the search will
                                    only filter based on the provided filters.
            limit (int): Maximum number of results to return.
            filters (list[str], optional): A list of filter clauses as strings.
                                           Each filter should follow the pattern:
                                           "field == value" or "field in [value1, value2, ...]".

        Returns:
            list: List of dictionaries containing search results with metadata.
        """
        # Get schema fields for validation
        schema_config = self.config.get("schema", {})
        fields_config = schema_config.get("fields", [])
        allowed_fields = {field["name"] for field in fields_config}

        def is_valid_filter(filter_str: str) -> bool:
            """
            A simple validator for filter expressions.
            The expected format is optionally enclosed in parentheses, then:
              field (==|in|<|>) value
            """
            pattern = r'^\s*\(?\s*([a-zA-Z_]+)\s*(==|in|<|>)\s*(.+)\s*\)?\s*$'
            match = re.match(pattern, filter_str)
            if not match:
                return False
            field = match.group(1)
            return field in allowed_fields

        # Validate the filters provided and keep only those that are valid.
        valid_filter_list = []
        for f in filters:
            if is_valid_filter(f):
                valid_filter_list.append(f)
            else:
                self.logger.warning(f"Invalid filter skipped: {f}")

        # Add security filters if access control is enabled
        if self.access_control:
            for access_rule in self.access_control:
                field = access_rule.get("field")
                allowed_values = access_rule.get("values", [])
                if field and allowed_values:
                    filter_str = f"{field} in [{', '.join([f'"{v}"' for v in allowed_values])}]"
                    valid_filter_list.append(filter_str)

        # Combine valid filters using 'and'
        filter_expr = " and ".join(valid_filter_list) if valid_filter_list else ""

        # Get output fields from schema
        output_fields = [field["name"] for field in fields_config
                         if field["name"] != "id" and field["name"] != "embedding"]

        if query_embedding:
            # Get search parameters from config or use defaults
            search_params = self.config.get("search", {}).get("params", {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            })

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=output_fields
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {field: hit.entity.get(field) for field in output_fields}
                    result["distance"] = hit.distance
                    formatted_results.append(result)

            return formatted_results
        else:
            # No embedding provided: query using the filter expression (or all docs if no filter).
            results = self.collection.query(
                expr=filter_expr,
                output_fields=output_fields
            )
            return results[:limit]

    def close(self):
        """Disconnect from Milvus."""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")




def load_schema_config(config_file: str) -> dict:
    """Loads the collection schema configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config



def build_collection_schema(config: dict, default_dim: int, security_config: dict = None) -> CollectionSchema:
    """
    Builds a CollectionSchema based on the provided configuration.

    Args:
        config (dict): Schema configuration
        default_dim (int): Default embedding dimension
        security_config (dict, optional): Security configuration for the collection

    Returns:
        CollectionSchema: The configured schema
    """
    fields_config = config.get("fields", [])
    # Always add the auto-generated id field.
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)
    ]

    for field in fields_config:
        name = field["name"]
        field_type = field["type"]

        if field_type == "string":
            max_length = field.get("max_length", 1024)
            fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length))
        elif field_type == "integer":
            fields.append(FieldSchema(name=name, dtype=DataType.INT64))
        elif field_type == "float_vector":
            dim = field.get("dim", default_dim)
            fields.append(FieldSchema(name=name, dtype=DataType.FLOAT_VECTOR, dim=dim))
        elif field_type == "array":
            max_length = field.get("max_length", 1024)
            fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=max_length))
        elif field_type == "float":
            fields.append(FieldSchema(name=name, dtype=DataType.FLOAT))
        elif field_type == "boolean":
            fields.append(FieldSchema(name=name, dtype=DataType.BOOL))
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

    # Add security fields if enabled
    if security_config and security_config.get("enable_encryption", False):
        # Add timestamp field for auditing
        if not any(field.name == "insert_timestamp" for field in fields):
            fields.append(FieldSchema(
                name="insert_timestamp",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Timestamp when record was inserted"
            ))

    description = config.get("description", "Document chunks with embeddings")
    return CollectionSchema(fields, description=description)

def create_collection(config: dict, collection_name: str, default_dim: int, security_config: dict = None) -> Collection:
    """
    Creates and returns a new Milvus collection based on the configuration.

    Args:
        config (dict): Schema configuration
        collection_name (str): Name of the collection to create
        default_dim (int): Default embedding dimension
        security_config (dict, optional): Security configuration

    Returns:
        Collection: The created Milvus collection
    """
    schema = build_collection_schema(config, default_dim, security_config)
    collection = Collection(name=collection_name, schema=schema)

    # Create index for vector search.
    index_field = config.get("index_field", "embedding")
    index_params = config.get("index_params", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    collection.create_index(index_field, index_params)

    # Create scalar field indices for filtering performance if specified
    scalar_indices = config.get("scalar_indices", [])
    for index_config in scalar_indices:
        field_name = index_config.get("field")
        index_type = index_config.get("type", "SCALAR")
        if field_name:
            collection.create_index(field_name, {"index_type": index_type})

    # Set collection properties for security if enabled
    if security_config and security_config.get("enable_encryption", False):
        # Note: This is a placeholder. The actual implementation would depend on
        # Milvus' security features which may vary by version
        logging.info(f"Collection {collection_name} created with encryption enabled")

    return collection
```

Configuration files:

```yaml
# Main configuration file for Vector Database

# Connection settings
connection:
  host: "localhost"
  port: "19530"
  user: ""
  password: ""
  use_secure: false
  timeout: 10

# Collection settings
collection:
  name: "documents"
  embedding_dim: 384
  schema_path: "milvus_schema.yaml"

# Security settings
security:
  enable_encryption: false
  encryption_key: ""
  access_control:
    - field: "author_role"
      values: ["writer", "editor"]
    - field: "source"
      values: ["public_docs", "internal_docs"]

# Search settings
search:
  params:
    metric_type: "L2"
    params:
      nprobe: 10
```

```yaml
# Milvus schema configuration file
description: "Document chunks with embeddings"

# Index configuration for vector search
index_field: "embedding"
index_params:
  index_type: "IVF_FLAT"
  metric_type: "L2"
  params:
    nlist: 128

# Scalar indices for faster filtering
scalar_indices:
  - field: "source"
    type: "SCALAR"
  - field: "author"
    type: "SCALAR"
  - field: "chunk_index"
    type: "SCALAR"

# Collection metadata
metadata:
  description: "Collection for storing document embeddings with metadata"
  version: "1.0"
  created_by: "VectorStorage"

# Field definitions
fields:
  - name: "text"
    type: "string"
    max_length: 1024
    description: "Text content of the document chunk."

  - name: "embedding"
    type: "float_vector"
    dim: 384
    description: "Embedding vector of the document chunk."

  - name: "source"
    type: "string"
    max_length: 1024
    description: "Source identifier of the document chunk."

  - name: "title"
    type: "string"
    max_length: 255
    description: "Title of the document."

  - name: "author"
    type: "string"
    max_length: 255
    description: "Author of the document."

  - name: "author_role"
    type: "string"
    max_length: 255
    description: "Role of the author in the document (e.g., writer, editor)."

  - name: "url"
    type: "string"
    max_length: 1024
    description: "URL associated with the document."

  - name: "chunk_index"
    type: "integer"
    description: "Index of the document chunk."

  - name: "insert_timestamp"
    type: "string"
    max_length: 64
    description: "Timestamp when the document was inserted."

  - name: "access_level"
    type: "string"
    max_length: 32
    description: "Access level for security control (public, internal, confidential)."
```
