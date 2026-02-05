# Vector Database Module Overview

This module provides a type-safe, provider-agnostic interface for storing document chunks and their embeddings in vector databases. All data models use Pydantic BaseModels for automatic validation and type safety.

## Files in This Module

### `__init__.py`
Exports the public API for the storage module. Provides clean imports for:
- `DatabaseClient`, `DatabaseClientConfig`, `DatabaseDocument` - Core abstractions
- `SearchResult`, `UpsertResult` - Operation result types
- `CollectionDescription` - Collection metadata with crawler config
- `MilvusDB` - Milvus implementation of DatabaseClient
- `MilvusBenchmark` - Benchmarking tools for Milvus
- `get_db`, `get_db_benchmark` - Factory functions

### `database_client.py`
Core abstractions and data models for the storage layer. Contains:

**Pydantic Models:**

- `DatabaseDocument` - Type-safe model for document chunks with validation
  - Required fields: `document_id` (str, UUID), `text` (str), `text_embedding` (list[float]), `chunk_index` (int, >= 0), `source` (str), `security_group` (list[str]), `metadata` (dict)
  - Optional fields: `id` (int, auto-generated), `text_sparse_embedding`, `metadata_sparse_embedding`, `benchmark_questions`
  - Validates `chunk_index >= 0`
  - Provides dict-like access methods (`__getitem__`, `get()`) for backward compatibility
  - Methods: `to_dict()`, `from_dict()`, `to_string()`

- `SearchResult` - Result from a search operation
  - Contains: `document` (DatabaseDocument), `distance` (float), `score` (float)
  - Factory method: `from_milvus_hit()` for converting Milvus results
  
- `UpsertResult` - Result from an upsert operation
  - Contains: `inserted_count`, `updated_count`, `failed_ids`
  - Properties: `total_count`, `has_failures`

- `CollectionDescription` - Metadata for collection configuration
  - Contains: `collection_config` (CrawlerConfig), `llm_prompt`, `columns`
  - Properties: `collection_security_groups`, `metadata_schema`, `description`
  - Methods: `to_json()`, `from_json()`, `to_crawler_config()`

- `DatabaseClientConfig` - Configuration model for database connections
  - Required: `provider`, `collection`
  - Optional: `partition`, `recreate`, `collection_description`, `access_level`
  - Connection params: `host`, `port` (validated 1-65535), `username`, `password`
  - Properties: `uri`, `token`
  - Factory method: `DatabaseClientConfig.milvus()` for convenient Milvus config

- `BenchmarkResult` - Results for a single benchmark query
- `BenchmarkRunResults` - Aggregated benchmark statistics with IR metrics

**Abstract Base Classes:**

- `DatabaseClient` - Interface that all database implementations must follow
  - Connection: `connect()`, `disconnect()`, `is_connected()`
  - Search: `search(texts, filters, limit)` -> `list[SearchResult]`
  - Get: `get_chunk(id)`, `get_document(document_id)`
  - Write: `upsert(documents)` -> `UpsertResult`
  - Delete: `delete_chunk(id)`, `delete_document(document_id)`
  - Collection: `create_collection()`, `get_collection()`
  - Utility: `exists(source, chunk_index)`

- `DatabaseBenchmark` - Interface for benchmarking database performance
  - Methods: `search()`, `run_benchmark()`, `plot_results()`, `save_results()`

### `database_utils.py`
Factory functions for creating database and benchmark instances:

- `get_db(config, dimension, crawler_config, embedder)` -> `DatabaseClient`
  - Returns appropriate DatabaseClient implementation (not yet connected)
  - Caller must call `connect()` before using

- `get_db_benchmark(db_config, embed_config, db)` -> `DatabaseBenchmark`
  - Returns appropriate DatabaseBenchmark implementation
  - Can optionally reuse an existing connected DatabaseClient

### `milvus_client.py`
Concrete implementation of DatabaseClient for Milvus vector database:

**Class: MilvusDB**

Implements the complete DatabaseClient interface for Milvus with:
- Connection management with state tracking
- Hybrid search (dense + sparse vectors) and filter-only query
- Full CRUD operations
- Security group filtering

`crawler_config` is optional: required only for `create_collection()`. When connecting to an existing collection for search/query only, pass `crawler_config=None`. When `search(texts=[], ...)` is called with no query text, MilvusDB performs a filter-only Milvus query (no embedding); results use `distance=1.0` / `score=0.0`.

**Connection Lifecycle:**
```python
# Create client (not yet connected)
db = MilvusDB(config, embedding_dimension, crawler_config, embedder)

# Connect with optional collection creation
db.connect(create_if_missing=True)

# Perform operations
results = db.search(["query text"], limit=10)
chunk = db.get_chunk(123)
doc = db.get_document("uuid-here")
result = db.upsert(documents)
db.delete_document("uuid-here")

# Disconnect when done
db.disconnect()
```

**Key Methods:**
- `connect(create_if_missing=False)` - Establishes connection, optionally creates collection
- `disconnect()` - Closes connection (safe to call multiple times)
- `is_connected()` - Returns connection state
- `search(texts, filters, limit)` - Hybrid search with RRF ranking; if `texts` is empty, runs filter-only query (no embedding)
- `get_chunk(id)` - Get single chunk by database ID
- `get_document(document_id)` - Get all chunks for a document
- `upsert(documents)` - Insert or update documents (uses source+chunk_index as key)
- `delete_chunk(id)` - Delete single chunk by ID
- `delete_document(document_id)` - Delete all chunks for a document
- `create_collection(recreate)` - Create collection with schema
- `get_collection()` - Get CollectionDescription with config
- `exists(source, chunk_index)` - Check if chunk exists (replaces check_duplicate)
- `set_embedder(embedder)` - Set embedder for search operations

**Deprecated Methods (for backwards compatibility):**
- `check_duplicate()` - Use `exists()` instead
- `insert_data()` - Use `upsert()` instead

### `milvus_utils.py`
Schema and index definitions for Milvus collections:

**Functions:**
- `create_schema(embedding_size, crawler_config)` -> `CollectionSchema`
  - Defines all fields including sparse embeddings for BM25
  - Sets up BM25 functions for automatic sparse embedding generation
  - Stores CrawlerConfig in collection description for pipeline restoration

- `create_index(client)` -> Index params
  - AUTOINDEX with COSINE metric for dense embeddings
  - SPARSE_INVERTED_INDEX with BM25 for full-text search
  - BITMAP index for security group filtering

- `extract_collection_description(description)` -> `CollectionDescription | None`
  - Parse JSON description from Milvus collection

- `create_description(fields, crawler_config)` -> JSON string
  - Creates collection description with config and LLM prompt

**Constants:**
- `MAX_DOC_LENGTH = 65535` - Maximum VARCHAR length
- `DEFAULT_SECURITY_GROUP = ["public"]` - Default RBAC group
- `DEFAULT_OUTPUT_FIELDS` - Standard fields for search/query operations

### `milvus_benchmarks.py`
Benchmarking tools for evaluating search quality:

**Class: MilvusBenchmark**

Now uses `MilvusDB.search()` instead of duplicating search logic.

**Key Features:**
- Uses MilvusDB for search operations
- IR metrics: MRR, Recall@K, Precision@K, NDCG@K, Hit Rate@K
- Progress tracking with tqdm
- Comprehensive logging

**Methods:**
- `__init__(db_config, embed_config, db=None)` - Can reuse existing MilvusDB
- `search(queries, filters)` - Delegates to MilvusDB.search()
- `run_benchmark(generate_queries, k_values, skip_docs_without_questions)` - Full benchmark

**BenchmarkMetrics utility class:**
- `calculate_mrr(placements)` - Mean Reciprocal Rank
- `calculate_recall_at_k(placements, k)` - Recall@K
- `calculate_precision_at_k(placements, k)` - Precision@K
- `calculate_ndcg_at_k(placements, k)` - NDCG@K
- `calculate_hit_rate_at_k(placements, k)` - Hit Rate@K
- `calculate_summary_stats(placements)` - Mean, median, std dev

### `db.md`
Comprehensive documentation for the storage module (may need updating).

## Design Decisions

### Connection State Management
The new interface separates construction from connection:
1. `__init__` creates the client with configuration
2. `connect()` establishes the connection
3. All operations require `is_connected() == True`
4. `disconnect()` cleanly closes resources

This enables better control over connection lifecycle and error handling.

### Unified Search Interface
Search is now part of the main `DatabaseClient` interface, not a separate benchmark class. This allows:
- Consistent search behavior across use cases
- Proper embedder integration
- Security group filtering in all searches

### Upsert Instead of Insert
`upsert()` replaces `insert_data()` with proper semantics:
- Uses `(source, chunk_index)` as unique key
- Returns `UpsertResult` with counts and failures
- Handles both insert and update in one call

### SearchResult Type
Search returns typed `SearchResult` objects instead of raw dicts:
- Contains full `DatabaseDocument`
- Includes `distance` and normalized `score`
- Factory method for converting Milvus results

### Collection Description for Config Restoration
`CollectionDescription` stores the complete `CrawlerConfig`, enabling:
- Pipeline reconstruction from collection metadata
- Schema validation on connect
- LLM prompt generation from metadata schema

## Dependencies

- `pymilvus>=2.6.0` - Milvus vector database client
- `pydantic>=2.0` - Data validation and settings management
- `tqdm>=4.66.0` - Progress bars
- `matplotlib>=3.10.5` - Plotting benchmark results (optional)

## Usage Examples

### Basic Usage
```python
from crawler.vector_db import get_db, DatabaseClientConfig

# Create config
config = DatabaseClientConfig.milvus(
    collection="my_docs",
    host="localhost",
    port=19530,
)

# Create and connect
db = get_db(config, embedding_dimension=384, crawler_config=crawler_config, embedder=embedder)
db.connect(create_if_missing=True)

# Search
results = db.search(["What is machine learning?"], limit=10)
for result in results:
    print(f"{result.document.source}: {result.score:.3f}")

# Get document
chunks = db.get_document("uuid-here")

# Upsert
result = db.upsert(database_documents)
print(f"Inserted: {result.inserted_count}, Updated: {result.updated_count}")

# Cleanup
db.disconnect()
```

### With Context Manager (recommended pattern)
```python
db = get_db(config, dimension, crawler_config, embedder)
try:
    db.connect(create_if_missing=True)
    # ... operations ...
finally:
    db.disconnect()
```

See also:
- `examples/arxiv.py` - Real-world usage with ArXiv papers
- `examples/xmidas.py` - XMIDAS document processing
