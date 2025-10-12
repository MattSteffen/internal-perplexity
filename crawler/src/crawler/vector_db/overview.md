# Storage Module Overview

This module provides a type-safe, provider-agnostic interface for storing document chunks and their embeddings in vector databases. All data models use Pydantic BaseModels for automatic validation and type safety.

## Files in This Module

### `__init__.py`
Exports the public API for the storage module. Provides clean imports for:
- `DatabaseClient`, `DatabaseClientConfig`, `DatabaseDocument` - Core abstractions
- `MilvusDB` - Milvus implementation of DatabaseClient
- `MilvusBenchmark` - Benchmarking tools for Milvus
- `get_db`, `get_db_benchmark` - Factory functions for creating database instances

### `database_client.py`
Core abstractions and data models for the storage layer. Contains:

**Pydantic Models:**
- `DatabaseDocument` - Type-safe model for document chunks with validation
  - Required fields: document_id, text, text_embedding, chunk_index, source, security_group
  - Optional fields: metadata, id, minio URL, sparse embeddings, benchmark questions
  - Validates chunk_index >= 0
  - Provides dict-like access for backward compatibility
  
- `DatabaseClientConfig` - Configuration model for database connections
  - Required: provider, collection
  - Optional: partition, recreate flag, description
  - Connection params: host, port (validated 1-65535), username, password
  - Properties: uri, token
  - Factory method: `DatabaseClientConfig.milvus()` for convenient Milvus config
  
- `BenchmarkResult` - Results for a single benchmark query
  - Tracks: query text, expected source, placement order, distance, search time, found flag
  - Validates placement_order >= 1, time_to_search >= 0
  
- `BenchmarkRunResults` - Aggregated benchmark statistics
  - Contains: results by document, placement/distance/time distributions, top-k percentages
  - Provides JSON serialization with integer key conversion

**Abstract Base Classes:**
- `DatabaseClient` - Interface that all database implementations must follow
  - Methods: `__init__`, `create_collection`, `insert_data`, `check_duplicate`
  
- `DatabaseBenchmark` - Interface for benchmarking database performance
  - Methods: `__init__`, `search`, `run_benchmark`, `plot_results`, `save_results`

### `database_utils.py`
Factory functions for creating database and benchmark instances based on provider:
- `get_db(config, dimension, metadata)` - Returns appropriate DatabaseClient implementation
- `get_db_benchmark(db_config, embed_config)` - Returns appropriate DatabaseBenchmark implementation
- Currently supports "milvus" provider, raises ValueError for unsupported providers

### `milvus_client.py`
Concrete implementation of DatabaseClient for Milvus vector database:

**Class: MilvusDB**
- Implements the DatabaseClient interface for Milvus
- Uses MilvusClient from pymilvus library
- Key features:
  - Automatic collection creation with schema from `milvus_utils`
  - Partition support for data organization
  - Optimized duplicate detection using bulk queries
  - Batch processing with progress tracking (tqdm)
  - Automatic data flushing for persistence
  - Comprehensive logging of insertion statistics
  
**Methods:**
- `create_collection(recreate)` - Creates collection and partition if needed
- `insert_data(data)` - Inserts DatabaseDocument instances with duplicate detection
- `check_duplicate(source, chunk_index)` - Checks if a specific chunk exists
- `_create_collection()` - Internal method to create collection with schema
- `_existing_chunk_indexes(source)` - Bulk fetches existing chunks for a source

### `milvus_utils.py`
Schema and index definitions for Milvus collections:

**Functions:**
- `create_schema(embedding_size, user_metadata_json_schema)` - Creates Milvus collection schema
  - Defines base fields: id, document_id, minio URL, chunk_index, text, embeddings
  - Adds sparse embeddings for BM25 full-text search
  - Includes metadata fields and security group for RBAC
  - Sets up BM25 functions for automatic sparse embedding generation
  - Enables dynamic fields for flexibility
  
- `create_index(client)` - Creates search indexes
  - AUTOINDEX with COSINE metric for dense text embeddings
  - SPARSE_INVERTED_INDEX with BM25 for full-text search (text and metadata)
  - BITMAP index for security group filtering
  - Configures BM25 parameters (k1=1.2, b=0.75)

**Constants:**
- `MAX_DOC_LENGTH = 65535` - Maximum VARCHAR length
- `DEFAULT_SECURITY_GROUP = ["public"]` - Default RBAC group
- `JSON_TYPE_TO_MILVUS_ELEMENT_TYPE` - Type mapping for schema generation

### `milvus_benchmarks.py`
Benchmarking tools for evaluating Milvus search performance:

**Class: MilvusBenchmark**
- Implements DatabaseBenchmark interface
- Uses Ollama for embedding generation
- Performs hybrid search (dense + sparse BM25)

**Key Features:**
- Comprehensive logging with emoji indicators
- Progress tracking with tqdm
- Hybrid search combining:
  - Dense vector similarity (COSINE)
  - BM25 full-text search on text
  - BM25 full-text search on metadata
  - RRF (Reciprocal Rank Fusion) for result merging
  
**Methods:**
- `__init__(db_config, embed_config)` - Connects to Milvus and Ollama
- `get_embedding(text)` - Generates embeddings via Ollama
- `search(queries, filters)` - Performs hybrid search
- `run_benchmark(generate_queries)` - Runs comprehensive benchmark
  - Loads documents from collection
  - Generates or uses stored benchmark questions
  - Executes searches and tracks metrics
  - Calculates top-k performance
  - Returns BenchmarkRunResults with full statistics
  
**Output Fields:**
- Defined in `OUTPUT_FIELDS` constant: source, chunk_index, text, metadata, custom fields

### `db.md`
Comprehensive documentation for the storage module:
- Architecture overview and features
- Installation requirements (pymilvus, pydantic)
- Basic usage examples with Pydantic models
- API reference for all classes and methods
- Schema configuration guide
- Provider extension instructions
- Error handling patterns
- Migration guide from legacy interface
- Performance considerations and best practices

## Design Decisions

### Pydantic for Type Safety
All data models use Pydantic BaseModel instead of dataclasses to provide:
- Automatic validation at creation and assignment time
- Clear error messages for invalid data
- Runtime type checking
- Easy serialization/deserialization (model_dump, model_validate)
- IDE autocomplete support
- Field-level validation (e.g., min/max constraints)

### System Field Prefixing
All system fields use `default_` prefix to avoid conflicts with user-defined metadata. This allows users to have fields like "text" or "source" in their metadata without collision.

### Security Group for RBAC
Every document includes a `security_group` field (array of strings) for row-level access control. Default is `["public"]` but should be explicitly set based on user permissions.

### Hybrid Search Architecture
Milvus implementation combines three search strategies:
1. Dense vector similarity (COSINE metric)
2. BM25 full-text search on document text
3. BM25 full-text search on metadata
Results are merged using RRF ranking for better relevance.

### Optimized Duplicate Detection
Instead of checking duplicates one-by-one, the implementation:
1. Groups documents by source
2. Fetches all existing chunk indexes for a source in one query
3. Checks against cached set for each chunk
This reduces database round-trips significantly for large batches.

## Dependencies

- `pymilvus>=2.6.0` - Milvus vector database client
- `pydantic>=2.0` - Data validation and settings management
- `ollama>=0.5.3` - Embedding generation (for benchmarks)
- `tqdm>=4.66.0` - Progress bars
- `matplotlib>=3.10.5` - Plotting benchmark results

## Usage Examples

See `db.md` for detailed usage examples, or refer to:
- `examples/arxiv.py` - Real-world usage with ArXiv papers
- `examples/xmidas.py` - XMIDAS document processing

