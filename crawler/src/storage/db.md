# Milvus Vector Database Storage

This module provides a Python interface for storing document chunks and their embeddings in a Milvus vector database.

## Features

- Automatic schema creation with support for custom metadata fields
- Duplicate detection based on source and chunk index
- Full-text search capabilities using BM25
- Vector similarity search using embeddings
- Partition support for data organization

## Architecture Overview

The module consists of two main components:

1. **MilvusStorage**: Main class for database operations (connection, insertion, duplicate checking)
2. **Schema Utilities**: Functions for creating and managing collection schemas

## Installation Requirements

```bash
pip install pymilvus
```

## Basic Usage

### 1. Initialize Connection

```python
from milvus_storage import MilvusStorage

# Configuration
milvus_config = {
    "host": "localhost",
    "port": "19530",
    "user": "username",
    "password": "password",
    "collection": "document_collection",
    "partition": "documents_2024"  # Optional
}

# Initialize storage
storage = MilvusStorage(milvus_config, recreate=False)
```

### 2. Create Collection with Custom Schema

```python
# Define custom metadata schema
custom_schema = {
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

# Create collection
storage.create_collection(embedding_size=768, schema=custom_schema)
```

### 3. Insert Document Data

```python
# Prepare data for insertion
document_data = [
    {
        "text": "This is the first chunk of the document...",
        "embedding": [0.1, 0.2, 0.3, ...],  # 768-dimensional vector
        "chunk_index": 0,
        "source": "document_001.pdf",
        "minio": "https://minio.example.com/bucket/document_001.pdf",
        # Custom metadata fields
        "title": "Machine Learning Fundamentals",
        "author": "Dr. Jane Smith",
        "date": "2024-01-15",
        "keywords": ["machine learning", "AI", "algorithms"],
        "page_count": 250,
        "is_published": True
    },
    {
        "text": "This is the second chunk of the document...",
        "embedding": [0.4, 0.5, 0.6, ...],
        "chunk_index": 1,
        "source": "document_001.pdf",
        "title": "Machine Learning Fundamentals",
        "author": "Dr. Jane Smith",
        # ... other fields
    }
]

# Insert data (automatically handles duplicates)
storage.insert_data(document_data)
```

## API Reference

### MilvusStorage Class

#### Constructor

```python
MilvusStorage(milvus_config: Dict[str, Any], recreate: bool = False)
```

**Parameters:**

- `milvus_config`: Configuration dictionary containing connection details
  - `host`: Milvus server hostname
  - `port`: Milvus server port
  - `user`: Username for authentication
  - `password`: Password for authentication
  - `collection`: Collection name
  - `partition`: Partition name (optional)
- `recreate`: Whether to recreate the collection if it exists

#### Methods

##### create_collection()

```python
create_collection(embedding_size: int, schema: Dict[str, Any]) -> None
```

Creates a new collection with the specified embedding size and custom schema.

**Parameters:**

- `embedding_size`: Dimension of the embedding vectors (e.g., 768 for BERT)
- `schema`: JSON schema defining custom metadata fields

##### insert_data()

```python
insert_data(data: List[Dict[str, Any]]) -> None
```

Inserts document chunks into the collection with automatic duplicate detection.

**Parameters:**

- `data`: List of document chunks, each containing:
  - `text` (required): The text content
  - `embedding` (required): Vector embedding of the text
  - `chunk_index` (required): Index of the chunk within the document
  - `source` (required): Source identifier (e.g., filename)
  - `minio` (optional): URL to the original document in MinIO
  - Custom metadata fields as defined in the schema

**Features:**

- Automatically skips duplicates based on `source` + `chunk_index`
- Validates required fields before insertion
- Generates unique UUIDs for each document chunk
- Serializes custom metadata as JSON

## Schema Configuration

### Base Schema Fields

Every collection automatically includes these base fields:

| Field                       | Type                | Description                       |
| --------------------------- | ------------------- | --------------------------------- |
| `id`                        | INT64               | Auto-generated primary key        |
| `document_id`               | VARCHAR(64)         | UUID for the document chunk       |
| `minio`                     | VARCHAR(256)        | URL to original document in MinIO |
| `chunk_index`               | INT64               | Index of chunk within document    |
| `text`                      | VARCHAR(65535)      | The text content                  |
| `text_embedding`            | FLOAT_VECTOR        | Dense vector embedding            |
| `text_sparse_embedding`     | SPARSE_FLOAT_VECTOR | BM25 sparse embedding             |
| `metadata`                  | VARCHAR(65535)      | JSON string of custom metadata    |
| `metadata_sparse_embedding` | SPARSE_FLOAT_VECTOR | BM25 embedding of metadata        |

### Custom Schema Fields

You can define additional fields using JSON schema format:

#### Supported Field Types

##### String Fields

```python
{
    "field_name": {
        "type": "string",
        "maxLength": 512,  # Optional, default 1024, max 1024
        "description": "Field description"
    }
}
```

##### Numeric Fields

```python
{
    "price": {
        "type": "number",  # or "float" or "integer"
        "description": "Document price"
    }
}
```

##### Boolean Fields

```python
{
    "is_published": {
        "type": "boolean",
        "description": "Publication status"
    }
}
```

##### Array Fields

```python
{
    "keywords": {
        "type": "array",
        "items": {
            "type": "string",
            "maxLength": 100
        },
        "maxItems": 50,  # Optional, default 100, max 100
        "description": "Document keywords"
    }
}
```

Supported array element types: `string`, `integer`, `number`, `boolean`

##### Object Fields

```python
{
    "metadata_object": {
        "type": "object",
        "description": "Complex metadata object"
    }
}
```

### Field Limitations

- **VARCHAR fields**: Maximum length of 1024 characters
- **Array fields**: Maximum capacity of 100 elements
- **Array string elements**: Maximum length of 512 characters per element
- **Reserved field names**: Cannot use base schema field names

## Full-Text Search Features

The module automatically creates BM25 sparse embeddings for:

1. **Text content**: Enables full-text search on document content
2. **Metadata**: Enables search across custom metadata fields

### BM25 Configuration

Default parameters:

- `bm25_k1`: 1.2 (term frequency saturation parameter)
- `bm25_b`: 0.75 (field length normalization parameter)
- Algorithm: `DAAT_MAXSCORE` (Dynamic pruning algorithm)

## Error Handling

The module includes comprehensive error handling:

### Common Exceptions

- **MilvusException**: Raised for Milvus-specific errors (connection, query, insertion)
- **ValueError**: Raised for invalid schema configurations or field types
- **Logging**: Detailed logging for debugging and monitoring

### Duplicate Handling

- Duplicates are detected using `source` + `chunk_index` combination
- Duplicate items are automatically skipped during insertion
- Detailed logging reports the number of duplicates found

## Performance Considerations

### Indexing

The module creates optimized indexes:

- **Vector index**: AUTOINDEX with COSINE similarity for dense embeddings
- **Sparse indexes**: SPARSE_INVERTED_INDEX for BM25 full-text search
