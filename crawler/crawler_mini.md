Here's the minimum file structure with documentation for each file needed to get your MVP crawler running with Milvus:

```
crawler/
├── Dockerfile
├── requirements.txt
├── config/
│   └── default.yaml
├── src/
│   ├── cli.py
│   ├── discovery.py
│   ├── processing/
│   │   ├── extractors.py
│   │   ├── chunking.py
│   │   └── embeddings.py
│   └── storage/
│       └── vector_db.py
```

### File Documentation

1. **Dockerfile**

```dockerfile
"""
Crawler container setup with Python dependencies and Milvus client support

Builds a minimal Python 3.10 image with:
- Required system libraries for text processing (libmagic, tesseract-ocr, etc.)
- Python dependencies from requirements.txt
- Entrypoint configured to run the CLI

Environment Variables:
- MILVUS_URI: Milvus server connection string
- INPUT_DIR: Default input directory for crawling
- OUTPUT_DIR: Default output directory for results
"""
```

2. **requirements.txt**

```text
"""
Core Python dependencies for the crawler MVP

Includes:
- Milvus client SDK
- File processing libraries (python-magic, unstructured, pdfplumber)
- Embedding generation (sentence-transformers)
- CLI framework (click)
- Environment management (python-dotenv)
"""
```

3. **config/default.yaml**

```yaml
"""
Default configuration for crawler parameters

Sections:
- file_types: Supported extensions and MIME types
- exclusion: Patterns and directories to ignore
- chunking: Size/overlap parameters for text splitting
- milvus: Collection settings and connection details
- paths: Default input/output directories
"""
```

4. **src/cli.py**

```python
"""
Command-line interface for directory crawler

Uses Click to provide:
- Main 'crawl' command with directory arguments
- Options for custom config/input/output paths
- Dry-run mode for testing
- Verbose logging control

Handles:
- Configuration loading
- Pipeline orchestration
- Error handling/reporting
- Progress reporting
"""
```

5. **src/discovery.py**

```python
"""
File discovery and filtering system

Contains:
- Recursive directory scanner with exclusion patterns
- MIME type validator using python-magic
- File metadata extractor (size, modified time, owner)
- Duplicate file checker via content hashing
- Batched output generator for memory efficiency

Key Functions:
- find_files(): Main entry point for discovery
- is_supported_type(): Validation against config
- get_metadata(): File system metadata collection
"""
```

6. **src/processing/extractors.py**

```python
"""
Content extraction pipeline for supported file types

Implements:
- Base ContentExtractor class with common interface
- Type-specific handlers (JSONHandler, PDFHandler, etc.)
- Error recovery for malformed files
- Structured output format with sections/metadata
- Fallback text extraction for unsupported types

Special Handling:
- JSON files: Parsing with schema validation
- Encrypted files: Error logging and skip
- Large files: Stream processing support
"""
```

7. **src/processing/chunking.py**

```python
"""
Semantic text chunking implementation

Features:
- Context-aware splitting (keep headers with content)
- Configurable chunk sizes and overlap
- Sentence boundary detection
- Token counting for LLM compatibility
- Batch processing for large documents

Main Class:
- TextSplitter: Orchestrates splitting strategies
"""
```

8. **src/processing/embeddings.py**

```python
"""
Embedding generation subsystem

Components:
- LocalEmbedder: Uses sentence-transformers models
- APIEmbedder: For OpenAI-compatible services
- Batch processing with parallelization
- Embedding cache implementation
- Dimension validation for Milvus compatibility

Key Methods:
- generate(): Main embedding creation interface
- normalize(): Vector normalization for cosine similarity
"""
```

9. **src/storage/vector_db.py**

```python
"""
Milvus vector database integration layer

Responsible for:
- Connection management and retry logic
- Collection schema verification/creation
- Batch insertion of vectors + metadata
- Index management and optimization
- Search configuration (consistency level, etc.)

Implements:
- VectorStorage: Main class with context manager
- Error handling for Milvus-specific exceptions
- Conversion between Milvus records and local formats
"""
```

### MVP Execution Flow

1. Start Milvus via Docker
2. Run crawler with test JSON file:

```bash
python src/cli.py -i ./test_data/ -o ./output/ --config config/default.yaml
```

3. System will:
   - Discover and validate JSON file
   - Extract content using JSONHandler
   - Split into chunks
   - Generate embeddings
   - Store in Milvus with metadata
   - Write processing results to output directory

This structure provides the minimum components needed to test with your JSON file while maintaining the architecture for future expansion. Each component is documented with its responsibilities and interface points.

To delete the milvus data and all collections:

```python
from pymilvus import connections, utility

# Connect to your Milvus instance
connections.connect("default", host="localhost", port="19530")

# Get all collection names
collection_names = utility.list_collections()

# Iterate through and drop each collection
for collection_name in collection_names:
    utility.drop_collection(collection_name)
    print(f"Dropped collection: {collection_name}")

print("All collections have been deleted.")
```
