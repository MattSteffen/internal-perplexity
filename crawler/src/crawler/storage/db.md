# Vector Database Interface

This module provides a unified Python interface for storing document chunks and their embeddings in vector databases. It supports multiple database providers through a common interface, with Milvus as the primary implementation.

## Features

- **Provider-agnostic interface**: Easily switch between different vector database providers
- **Automatic schema creation** with support for custom metadata fields
- **Duplicate detection** based on source and chunk index
- **Protocol-based document handling** for flexible data structures
- **Full-text search capabilities** using BM25 (Milvus)
- **Vector similarity search** using embeddings
- **Partition support** for data organization

## Architecture Overview

The module consists of several key components:

1. **DatabaseClient Interface**: Abstract base class defining the contract for all database implementations
2. **DatabaseDocument Protocol**: Protocol defining the minimum interface for document data
3. **Configuration Classes**: Typed configuration objects for different database providers
4. **Provider Implementations**: Database-specific implementations (currently Milvus)

## Installation Requirements

```bash
pip install pymilvus  # For Milvus support
```

## Basic Usage

### 1. Configuration Setup

```python
from database_interface import MilvusConfig, MilvusStorage

# Create typed configuration
config = MilvusConfig(
    collection="document_collection",
    host="localhost",
    port=19530,
    username="root",
    password="Milvus",
    partition="documents_2024",  # Optional
    recreate=False
)
```

### 2. Initialize Database Client

```python
# Define custom metadata schema
# optionally an actual jsonschema
metadata_schema = {
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 512,
            "description": "Document title"
        },
        "author": {
            "type": "string",
            "maxLength": 256,
            "description": "Document author"
        },
        "date": {
            "type": "string",
            "maxLength": 32,
            "description": "Publication date"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 20,
            "description": "Document keywords"
        },
        "page_count": {
            "type": "integer",
            "description": "Number of pages"
        },
        "is_published": {
            "type": "boolean",
            "description": "Publication status"
        }
    }
}

# Initialize client
client = MilvusStorage(
    config=config,
    embedding_dimension=768, # depends on the embedding model you choose
    metadata_schema=metadata_schema
)
```

### 3. Working with Document Data

The interface uses the `DatabaseDocument` protocol for flexible document handling:

```python
from dataclasses import dataclass
from typing import List, Any

@dataclass
class DocumentChunk:
    """Example implementation of DatabaseDocument protocol"""
    default_text: str
    default_text_embedding: List[float]
    default_chunk_index: int
    default_source: str
    default_document_id: str = ""
    minio: str = ""
    title: str = ""
    author: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

# Create document data
documents = [
    DocumentChunk(
        default_text="This is the first chunk of the document...",
        default_text_embedding=[0.1, 0.2, 0.3] * 256,  # 768-dimensional vector
        default_chunk_index=0,
        default_source="document_001.pdf",
        minio="https://minio.example.com/bucket/document_001.pdf",
        title="Machine Learning Fundamentals",
        author="Dr. Jane Smith"
    ),
    DocumentChunk(
        default_text="This is the second chunk of the document...",
        default_text_embedding=[0.4, 0.5, 0.6] * 256,
        default_chunk_index=1,
        default_source="document_001.pdf",
        title="Machine Learning Fundamentals",
        author="Dr. Jane Smith"
    )
]

# Insert data (automatically handles duplicates)
client.insert_data(documents)
```

### 4. Using Dictionary-based Documents

The protocol also supports regular dictionaries:

```python
# Dictionary-based documents
dict_documents = [
    {
        "default_text": "This is document content...",
        "default_text_embedding": [0.1, 0.2, 0.3] * 256,
        "default_chunk_index": 0,
        "default_source": "document_002.pdf",
        "title": "Deep Learning Guide",
        "author": "Prof. John Doe",
        "keywords": ["deep learning", "neural networks"],
        "page_count": 300,
        "is_published": True
    }
]

# Convert to protocol-compatible format if needed
class DictDocument:
    def __init__(self, data: dict):
        self._data = data

    @property
    def default_text(self) -> str:
        return self._data["default_text"]

    @property
    def default_text_embedding(self) -> List[float]:
        return self._data["default_text_embedding"]

    @property
    def default_chunk_index(self) -> int:
        return self._data["default_chunk_index"]

    @property
    def default_source(self) -> str:
        return self._data["default_source"]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

protocol_documents = [DictDocument(doc) for doc in dict_documents]
client.insert_data(protocol_documents)
```

## API Reference

### DatabaseClient Interface

#### Abstract Methods

All database implementations must provide these methods:

```python
def __init__(
    self,
    config: DatabaseClientConfig,
    embedding_dimension: int,
    metadata_schema: Dict[str, Any]
) -> None:
    """Initialize the database client."""

def create_collection(self, recreate: bool = False) -> None:
    """Create a collection with the specified schema."""

def insert_data(self, data: List[DatabaseDocument]) -> None:
    """Insert data with duplicate detection."""

def check_duplicate(self, default_source: str, default_chunk_index: int) -> bool:
    """Check if a document chunk already exists."""
```

### DatabaseDocument Protocol

Documents must implement this protocol:

```python
class DatabaseDocument(Protocol):
    # Required attributes (using prefixed field names)
    default_text: str
    default_text_embedding: List[float]
    default_chunk_index: int
    default_source: str

    # Required methods for dict-like access
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
```

### Configuration Classes

#### MilvusConfig

```python
@dataclass
class MilvusConfig(DatabaseClientConfig):
    host: str = "localhost"
    port: int = 19530
    username: str = "root"
    password: str = "Milvus"

    @property
    def uri(self) -> str:
        """Get the connection URI."""
        return f"http://{self.host}:{self.port}"

    @property
    def token(self) -> str:
        """Get the authentication token."""
        return f"{self.username}:{self.password}"
```

#### Base Configuration

```python
@dataclass
class DatabaseClientConfig:
    provider: DatabaseProvider
    collection: str
    partition: Optional[str] = None
    recreate: bool = False
    collection_description: Optional[str] = None
```

### MilvusStorage Implementation

#### Constructor

```python
MilvusStorage(
    config: DatabaseClientConfig,
    embedding_dimension: int,
    metadata_schema: Dict[str, Any]
)
```

**Parameters:**

- `config`: MilvusConfig instance with connection details
- `embedding_dimension`: Dimension of the embedding vectors (e.g., 768)
- `metadata_schema`: JSON schema defining custom metadata fields

#### Methods

##### create_collection()

```python
create_collection(recreate: bool = False) -> None
```

Creates a collection with the specified schema. Handles existing collections based on the `recreate` parameter.

##### insert_data()

```python
insert_data(data: List[DatabaseDocument]) -> None
```

Inserts document chunks with automatic duplicate detection.

**Features:**

- Automatically skips duplicates based on `source` + `chunk_index`
- Works with any object implementing the DatabaseDocument protocol
- Generates unique UUIDs for document chunks
- Serializes additional metadata as JSON

##### check_duplicate()

```python
check_duplicate(source: str, chunk_index: int) -> bool
```

Checks if a document chunk already exists in the collection.

## Schema Configuration

### Base Schema Fields

Every collection automatically includes these base fields:

| Field                               | Type                | Description                       |
| ----------------------------------- | ------------------- | --------------------------------- |
| `id`                                | INT64               | Auto-generated primary key        |
| `default_document_id`               | VARCHAR(64)         | UUID for the document chunk       |
| `default_minio`                     | VARCHAR(256)        | URL to original document in MinIO |
| `default_chunk_index`               | INT64               | Index of chunk within document    |
| `default_source`                    | VARCHAR(512)        | Source identifier                 |
| `default_text`                      | VARCHAR(65535)      | The text content                  |
| `default_text_embedding`            | FLOAT_VECTOR        | Dense vector embedding            |
| `default_text_sparse_embedding`     | SPARSE_FLOAT_VECTOR | BM25 sparse embedding             |
| `default_metadata`                  | VARCHAR(65535)      | JSON string of custom metadata    |
| `default_metadata_sparse_embedding` | SPARSE_FLOAT_VECTOR | BM25 embedding of metadata        |

### Custom Schema Fields

Define additional fields using JSON schema format (same as before):

#### Supported Field Types

- **String fields**: `{"type": "string", "maxLength": 512}`
- **Numeric fields**: `{"type": "number"}` or `{"type": "integer"}`
- **Boolean fields**: `{"type": "boolean"}`
- **Array fields**: `{"type": "array", "items": {...}, "maxItems": 50}`
- **Object fields**: `{"type": "object"}`

## Provider Extension

To add support for new database providers:

### 1. Create Configuration Class

```python
@dataclass
class NewProviderConfig(DatabaseClientConfig):
    connection_string: str
    api_key: str

    def __post_init__(self):
        super().__post_init__()
        self.provider = DatabaseProvider.NEW_PROVIDER
```

### 2. Implement DatabaseClient

```python
class NewProviderStorage(DatabaseClient):
    def __init__(
        self,
        config: DatabaseClientConfig,
        embedding_dimension: int,
        metadata_schema: Dict[str, Any]
    ):
        # Implementation specific to new provider
        pass

    def create_collection(self, recreate: bool = False) -> None:
        # Provider-specific collection creation
        pass

    def insert_data(self, data: List[DatabaseDocument]) -> None:
        # Provider-specific data insertion
        pass

    def check_duplicate(self, default_source: str, default_chunk_index: int) -> bool:
        # Provider-specific duplicate checking
        pass
```

### 3. Update Provider Enum

```python
class DatabaseProvider(Enum):
    MILVUS = "milvus"
    NEW_PROVIDER = "new_provider"
```

## Error Handling

The interface provides comprehensive error handling:

### Common Patterns

- **Configuration validation**: Ensures proper config types and required fields
- **Protocol compliance**: Validates that documents implement required interface
- **Provider-specific errors**: Each implementation handles its own error types
- **Graceful degradation**: Continues processing valid items when some items fail

### Example Error Handling

```python
try:
    client.insert_data(documents)
except ValueError as e:
    print(f"Configuration or protocol error: {e}")
except Exception as e:
    print(f"Database-specific error: {e}")
```

## Migration from Legacy Interface

If migrating from the old dictionary-based interface:

### Before (Legacy)

```python
storage = MilvusStorage(milvus_config_dict, recreate=False)
storage.create_collection(768, schema_dict)
storage.insert_data(list_of_dicts)
```

### After (New Interface)

```python
config = MilvusConfig(collection="my_collection", host="localhost")
client = MilvusStorage(config, 768, schema_dict)
# Collection created automatically in constructor
client.insert_data(protocol_compatible_documents)
```

## Performance Considerations

### Indexing (Milvus)

- **Vector index**: AUTOINDEX with COSINE similarity for dense embeddings
- **Sparse indexes**: SPARSE_INVERTED_INDEX for BM25 full-text search
- **Automatic flushing**: Ensures data persistence after insertion

### Best Practices

- **Batch insertions**: Insert multiple documents at once for better performance
- **Duplicate checking**: Built-in duplicate detection prevents redundant data
- **Memory efficiency**: Protocol-based approach minimizes data copying
- **Type safety**: Strongly typed configurations reduce runtime errors
