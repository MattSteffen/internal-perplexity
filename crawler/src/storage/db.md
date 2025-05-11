# Milvus Vector Storage Management

## Overview

This project provides a Python interface for managing and interacting with a [Milvus](https://milvus.io/) vector database. It simplifies connecting to Milvus, defining a collection schema (including standard fields and user-defined metadata), inserting document chunks with their embeddings, and performing basic checks like duplicate detection and source existence.

The system is designed to:

- Connect to a Milvus instance.
- Create a Milvus collection with a flexible schema defined in a configuration file.
- Automatically add six standard fields: `text`, `embedding`, `source`, `chunk_index`, `full_text_embedding`, and `metatext`.
- Define indexes for dense (`embedding`) and sparse (`full_text_embedding`) vectors.
- Insert data into the collection, with an initial (though currently partly disabled) mechanism for duplicate prevention.
- Check if data from a specific source already exists.

## Files

The project consists of two main Python files:

### 1. `vector_storage.py`

- **Purpose**: This file contains the `VectorStorage` class, which serves as the primary client for interacting with a specific Milvus collection. It encapsulates the logic for connection management, schema creation, data insertion, and querying metadata.
- **`VectorStorage` Class**:
  - **Description**: Manages all operations related to a Milvus collection designed for storing document chunks, their dense vector embeddings, sparse embeddings for metadata, and associated metadata. It ensures the collection is properly set up with the required schema and indexes before use.
  - **Required Inputs (for the constructor `__init__`)**:
        The class is initialized with a `config` dictionary, which should contain the following keys:
    - `collection` (str): **Required**. The name of the Milvus collection to be managed.
    - `milvus` (dict): **Required**. Connection details for the Milvus server.
      - `host` (str): Hostname or IP address of the Milvus server.
      - `port` (int): Port number of the Milvus server.
      - `user` (str, optional): Username for Milvus authentication. Defaults to "root" if not provided (Note: Default credentials should be handled carefully).
      - `password` (str, optional): Password for Milvus authentication. Defaults to "Milvus" if not provided (Note: Default credentials should be handled carefully).
      - `secure` (bool, optional): Set to `True` to use HTTPS, `False` for HTTP. Defaults to `False`.
    - `embeddings` (dict): **Required**. Configuration related to embeddings.
      - `dimension` (int): The dimension of the dense vector embeddings (e.g., for the `embedding` field). Defaults to 384 if not specified.
    - `metadata` (dict): **Required**. Configuration for metadata fields.
      - `schema` (dict): **Required**. A JSON schema-like dictionary defining the custom metadata fields and their types. This schema is used by `utils.build_milvus_schema` to create the collection schema.
      - `full_text_search` (List[str], optional): A list of metadata field names that should be concatenated into the `metatext` field for full-text search capabilities.
    - `partition` (str, optional): The name of a specific partition to use within the collection. If provided and it doesn't exist, it will be created.
    - `recreate` (bool, optional): If `True`, the collection will be dropped and recreated if it already exists. Defaults to `False`.

  - **Key Methods**:
    - `__init__(self, config: Dict[str, Any])`: Initializes the `VectorStorage` instance with the provided configuration. It validates the Pymilvus installation and sets up internal configuration attributes.
    - `__enter__(self) -> 'VectorStorage'`: Establishes a connection to the Milvus server when entering a `with` statement. It checks if the specified collection exists. If not, it creates the collection using `_create_collection`. If `recreate` is true, it drops and recreates an existing collection. It also creates a partition if `partition_name` is specified and it doesn't exist. Finally, it loads the collection into memory.
    - `__exit__(self, exc_type, exc_value, traceback)`: Cleans up resources when exiting a `with` statement. It releases the collection from memory and closes the Milvus client connection.
    - `close(self)`: Provides an explicit method to release the collection and close the client connection, effectively calling `__exit__`.
    - `_create_collection(self)`: (Internal method) Creates the Milvus collection. It uses `utils.build_milvus_schema` to generate the schema based on the configuration. It then prepares and adds two predefined indexes:
            1. `embedding_index` for the `embedding` field (dense vectors, `AUTOINDEX`, `COSINE` metric).
            2. `full_text_embedding_index` for the `full_text_embedding` field (sparse vectors, `SPARSE_INVERTED_INDEX`, `BM25` metric).
    - `_check_duplicates(self, metadatas: List[Dict[str, Any]]) -> List[int]`: (Internal method, currently not fully utilized in `insert_data`) Designed to identify indices of new items to insert by checking for duplicates based on the `('source', 'chunk_index')` combination. It first checks for duplicates within the input batch and then queries the database for existing entries. *Note: The call to this method in `insert_data` is currently commented out.*
    - `check_source(self, source: str) -> bool`: Queries the Milvus collection to check if any records with the given `source` identifier already exist. Returns `True` if found, `False` otherwise.
    - `insert_data(self, data: List[Dict[str, Any]])`: Inserts a list of data entries into the Milvus collection. Each entry in the `data` list is a dictionary where keys correspond to field names in the collection schema.
      - It originally intended to use `_check_duplicates` to filter out existing entries, but this step is currently bypassed.
      - It constructs the `metatext` field for each entry by JSON-dumping a dictionary of selected metadata fields (specified in `config.metadata.full_text_search`).
      - It then inserts the processed data into the collection and flushes it.

### 2. `utils.py`

- **Purpose**: This file provides utility functions that support the `VectorStorage` class, primarily focusing on schema generation and validation.
- **Key Functions**:
  - `validate_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]`: Performs a basic validation on the schema configuration provided in the main config. It checks if `schema_config` is a dictionary and contains a `properties` key. *Note: It has a TODO to implement full JSON-Schema validation.*
  - `_validate_metadata(x, y)`: This function is intended to validate individual metadata entries against the defined schema (e.g., checking field types, lengths). *Currently, it's a placeholder and always returns `True`.*
  - `build_milvus_schema(schema_config: Dict[str, Any], embedding_dim: int) -> CollectionSchema`: Constructs a `pymilvus.CollectionSchema` object.
    - It starts by defining six standard fields:
            1. `id`: `INT64`, primary key, auto-generated.
            2. `embedding`: `FLOAT_VECTOR` with the specified `embedding_dim`.
            3. `text`: `VARCHAR` (max length `MAX_DOC_LENGTH`) for the document chunk.
            4. `metatext`: `VARCHAR` (max length 20000) for storing a concatenation of metadata fields for full-text search, with an analyzer enabled.
            5. `full_text_embedding`: `SPARSE_FLOAT_VECTOR` for BM25 scores from `metatext`.
            6. `source`: `VARCHAR` (max length `MAX_SOURCE_LENGTH`).
            7. `chunk_index`: `INT64`.
    - It then iterates through the `properties` defined in the `schema_config` (user-defined metadata) and adds corresponding `FieldSchema` objects. It maps JSON schema types (`string`, `integer`, `number`/`float`, `boolean`, `array`) to appropriate Milvus `DataType`.
      - Arrays are mapped to `DataType.ARRAY` if the element type is supported (string, int, float, bool). If the array type or structure is unsupported, it falls back to serializing the array as a `VARCHAR`.
      - `object` types are currently not supported and will raise a `NotImplementedError`.
    - Finally, it adds a BM25 `Function` to the schema, which maps the `metatext` input field to the `full_text_embedding` output field.

## How to Use

### Prerequisites

1. **Python Environment**: Ensure you have Python installed.
2. **Pymilvus**: Install the Pymilvus library: `pip install pymilvus`
3. **Milvus Instance**: Have a Milvus instance running and accessible.

### Configuration

Prepare a configuration dictionary as described in the `VectorStorage` class's "Required Inputs" section.

Example `config.json`:

```json
{
  "collection": "research_documents",
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "user": "root",
    "password": "Milvus"
  },
  "embeddings": {
    "dimension": 768
  },
  "metadata": {
    "schema": {
      "description": "Schema for research documents",
      "properties": {
        "author": { "type": "string", "maxLength": 256, "description": "Author of the document" },
        "year": { "type": "integer", "description": "Publication year" },
        "keywords": { "type": "array", "items": { "type": "string", "maxLength": 100 }, "maxItems": 20, "description": "List of keywords" },
        "category": { "type": "string", "maxLength": 128, "description": "Document category" }
      }
    },
    "full_text_search": ["author", "category", "keywords_serialized_for_metatext"]
    // Note: If 'keywords' is an array, you'd typically pre-process it into a string
    // or handle its inclusion in metatext carefully. For simplicity, this example assumes
    // you might have a separate field or process it before metatext creation.
    // The current `insert_data` metatext creation simply JSON dumps selected fields.
  },
  "partition": "signal_processing_papers",
  "recreate": false
}
```

*(Adjust `full_text_search` based on how you want to construct `metatext`. If using array fields, you'll need to decide how they contribute to a single string `metatext` field).*

### Initialization and Usage

```python
import json
from vector_storage import VectorStorage # Assuming your file is vector_storage.py

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Sample data to insert
# Each dictionary must contain keys for the 6 standard fields
# (text, embedding, source, chunk_index, full_text_embedding, metatext)
# plus any custom fields defined in your metadata.schema.
# 'full_text_embedding' and 'metatext' are generated internally by insert_data for now.
# 'embedding' and 'full_text_embedding' should be actual vectors.
# For simplicity, placeholders are used here for embeddings.

sample_data_to_insert = [
    {
        "text": "This is the first chunk of document A.",
        "embedding": [0.1] * config["embeddings"]["dimension"], # Replace with actual embedding
        "source": "doc_A.pdf",
        "chunk_index": 0,
        # "full_text_embedding": <generated_by_bm25_if_not_provided>, # Example: sparse vector
        # "metatext": <generated_from_metadata_fields>,
        "author": "Jane Doe",
        "year": 2023,
        "keywords": ["ai", "signal processing"], # This will be part of metatext
        "category": "Research Paper"
    },
    {
        "text": "This is a chunk from document B about machine learning.",
        "embedding": [0.2] * config["embeddings"]["dimension"], # Replace with actual embedding
        "source": "doc_B.txt",
        "chunk_index": 0,
        "author": "John Smith",
        "year": 2022,
        "keywords": ["ml", "algorithms"], # This will be part of metatext
        "category": "Internal Report"
    }
]

# Ensure all required fields are present in sample_data_to_insert,
# including those defined in config.metadata.schema.
# The `metatext` and `full_text_embedding` fields are handled by `insert_data`
# and the schema's BM25 function, respectively, based on the current code.
# You would typically generate dense 'embedding' vectors externally.

try:
    with VectorStorage(config=config) as vs:
        # Check if a source already exists
        if not vs.check_source("doc_A.pdf"):
            print("Source 'doc_A.pdf' not found, proceeding with insert.")
        else:
            print("Source 'doc_A.pdf' already exists.")

        # Insert data
        print(f"Inserting {len(sample_data_to_insert)} items...")
        vs.insert_data(sample_data_to_insert)
        print("Data insertion process complete.")

except ImportError as e:
    print(f"Error: {e}. Pymilvus might not be installed.")
except ValueError as e:
    print(f"Configuration or Schema Error: {e}")
except RuntimeError as e:
    print(f"Runtime Error (e.g., Milvus connection): {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

## Improvements

Here are suggestions to make the files more manageable, usable, and easier to understand:

### `vector_storage.py`

1. **Configuration Management**:
    - Use Pydantic models for validating the `config` dictionary. This provides clear error messages and type safety for configuration.
    - Avoid hardcoding default credentials (`user`, `password`) in `__enter__`. Rely solely on the configuration. Remove the `TODO` once done.
    - Make index parameters (e.g., `metric_type`, `index_type` for both `embedding` and `full_text_embedding` indexes) configurable via the `config` file rather than being hardcoded in `_create_collection`.
2. **Duplicate Checking (`_check_duplicates`)**:
    - Clarify the necessity and finalize the implementation of `_check_duplicates`. The comment "TODO: Check again. But actually don't need for now." should be addressed.
    - If duplicate checking is required, ensure the implementation is efficient for large batches and re-enable its use in `insert_data`.
3. **Data Insertion (`insert_data`)**:
    - Re-enable and complete the metadata validation step using `_validate_metadata` (once implemented in `utils.py`).
    - The creation of `metatext`:
        - Make the process more robust. Instead of `data[i][k]`, use `data[i].get(k, default_value)` to avoid KeyErrors if a field listed in `full_text_search` is missing from an item.
        - Consider how array fields should contribute to `metatext` (e.g., join array elements into a string).
4. **Error Handling**:
    - Implement more specific custom exceptions for different error conditions (e.g., `SchemaError`, `MilvusConnectionError`).
5. **Modularity & Clarity**:
    - The `_create_collection` method is quite long. Index parameter setup could be refactored into a helper method.
    - The logging message `Collection '{self.collection_name}' already exists.` after *recreating* a collection in `__enter__` seems incorrect and should state that the collection was recreated.
6. **Embedding Dimension**: The `TODO` for determining `embedding_dim` from the embedding model (in `_create_collection`) is good; this would make the system more flexible.
7. **Constants**: `MAX_DOC_LENGTH`, `MAX_SOURCE_LENGTH` are defined here and in `utils.py`. Consolidate them into a single `constants.py` file or pass them via configuration.

### `utils.py`

1. **Schema Validation (`validate_schema`)**:
    - Implement full JSON-Schema validation using a library like `jsonschema` to enforce the structure and types of `schema_config`. Address the `TODO`.
2. **Metadata Validation (`_validate_metadata`)**:
    - **Crucial**: Implement this function thoroughly. It should validate each piece of incoming data against the `schema_config` (e.g., checking types, `maxLength` for strings, `maxItems` for arrays, allowed values if `enum` is used in JSON schema).
3. **Schema Building (`build_milvus_schema`)**:
    - **Array Element Type Bug**: In the `case "array":` block, when creating `FieldSchema` for an array, `element_type=DataType.VARCHAR,` is hardcoded. This should be `element_type=milvus_element_type,` to use the dynamically determined type.
    - **Object Type Support**: Implement the `TODO` for handling `object` types (likely by serializing to JSON and using Milvus's JSON field type) if this is a requirement.
    - **Default Values**: Review default values assigned to fields to ensure they are sensible (e.g., `default_value="Unknown"` for strings, `0` for integers).
    - **Clarity**: Add more comments explaining fallback logic (e.g., why an array might be serialized to VARCHAR).
    - **Error Messages**: Provide more specific error messages if a field definition within `schema_config.properties` is invalid.
4. **Constants**: Consolidate constants as mentioned for `vector_storage.py`.

### General Improvements

1. **Testing**: Add comprehensive unit tests for:
    - Schema building (`build_milvus_schema`) with various valid and invalid inputs.
    - Metadata validation (`_validate_metadata`) once implemented.
    - `VectorStorage` class methods (connection, creation, insertion, checks).
2. **Docstrings & Type Hinting**: Continue the good practice of using docstrings and type hints. Ensure `_validate_metadata` gets a proper docstring once implemented.
3. **Dependencies**: Create a `requirements.txt` file listing all dependencies (`pymilvus`, `jsonschema` if added, etc.).
4. **Examples**:
    - Provide more complete examples, including how to generate embeddings and how to perform hybrid searches or queries using the created indexes. The commented-out section in `vector_storage.py` regarding hybrid search can be a good basis for an example script.
5. **Security**: Reiterate the importance of not hardcoding credentials, especially for production environments. Configuration should be managed securely.
6. **Logging**: Standardize logging levels and ensure log messages are consistently informative.
