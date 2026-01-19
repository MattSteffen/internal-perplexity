# Document Module

## Overview

The document module provides a unified `Document` class that serves as the central data structure for the document processing pipeline. This class is designed to be mutable and flows through multiple processing stages, with each stage populating specific fields. The Document class uses Pydantic BaseModel for automatic validation, serialization, and type safety.

## Files

### `__init__.py`

Exports the `Document` class for easy importing by other modules.

### `document.py`

Contains the core `Document` class (Pydantic BaseModel) with the following responsibilities:

- Defines the document structure with fields for all processing stages
- Provides automatic validation and serialization via Pydantic
- Provides factory methods for creating new documents (`create()`, `from_file()`)
- Implements validation and status checking methods
- Tracks document state through the pipeline
- Provides methods for saving/loading documents to/from JSON files
- Converts documents to database entities for storage

## Document Processing Pipeline

The `Document` class flows through the following stages:

1. **Creation**: Document is created with `document_id` (UUID) and `source` (file path/URL)
2. **Converter**: Populates `content` (bytes) and `markdown` (str)
3. **Extractor**: Populates `metadata` (dict) and `benchmark_questions` (list of strings)
4. **Chunker + Embedder**: Populates `chunks` (list of `Chunk` objects) - each chunk contains text and its dense embedding. Sparse embeddings are optional and computed by the database on insert.
5. **Vector DB**: Uses `to_database_documents()` to create `DatabaseDocument` objects for storage

## Document Class Fields

### Required (at creation)

- `document_id` (str): Unique identifier (UUID string, auto-generated if not provided)
- `source` (str): Source identifier (file path, URL, etc.)

### Converter Stage (populated by converter)

- `content` (bytes | None): Raw binary content of the document
- `markdown` (str | None): Markdown representation of the document
- `source_name` (str | None): Source name from converter (e.g., filename)
- `stats` (ConversionStats): Conversion statistics
- `warnings` (list[str]): Conversion warnings

### Extractor Stage (populated by extractor)

- `metadata` (dict[str, Any] | None): Extracted structured metadata
- `benchmark_questions` (list[str] | None): Generated questions for testing

### Chunker + Embedder Stage (populated together)

- `chunks` (list[Chunk] | None): List of `Chunk` objects, each containing:
  - `text` (str): The text content of the chunk
  - `chunk_index` (int): Zero-based index of the chunk
  - `text_embedding` (list[float]): Dense vector embedding for the chunk (required)
  - `sparse_text_embedding` (list[float] | None): Sparse embedding for text (optional, computed by DB)
  - `sparse_metadata_embedding` (list[float] | None): Sparse embedding for metadata (optional, computed by DB)

### Optional

- `security_group` (list[str]): RBAC access control groups (defaults to ["public"])
- `minio_url` (str | None): URL to document in object storage

## Usage Example

```python
from crawler.document import Document
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.chunker import Chunker, ChunkingConfig
from crawler.vector_db import get_db, DatabaseClientConfig
from crawler.llm import LLMConfig, get_llm, get_embedder, EmbedderConfig

# 1. Create document
doc = Document.create(source="example.pdf")

# 2. Convert to markdown
converter_config = PyMuPDF4LLMConfig(type="pymupdf4llm", vlm_config=LLMConfig.ollama(model_name="llava"))
converter = create_converter(converter_config)
converter.convert_document(doc)  # Modifies doc in place

# 3. Extract metadata
llm = get_llm(LLMConfig.ollama(model_name="llama3.2"))
extractor_config = MetadataExtractorConfig(json_schema={...}, context="")
extractor = MetadataExtractor(llm=llm, config=extractor_config)
result = extractor.run(doc)  # Returns MetadataExtractionResult
doc.metadata = result.metadata
doc.benchmark_questions = result.benchmark_questions

# 4. Chunk the markdown and generate embeddings
chunker = Chunker(ChunkingConfig.create(chunk_size=1000))
embedder = get_embedder(EmbedderConfig.ollama(model="all-minilm:v2"))

# Get text chunks and generate embeddings, then create Chunk objects
text_chunks = chunker.chunk_text(doc)  # Returns list[str]
embeddings = embedder.embed_batch(text_chunks)  # Returns list[list[float]]
from crawler.document import Chunk
doc.chunks = [
    Chunk(text=text, chunk_index=i, text_embedding=embedding)
    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings))
]

# 6. Store in vector database
from crawler.config import CrawlerConfig
db_config = DatabaseClientConfig.milvus(collection="my_collection")
crawler_config = CrawlerConfig.create(...)  # Your crawler config
db = get_db(db_config, embedder.get_dimension(), crawler_config)
db_docs = doc.to_database_documents()  # Convert to DatabaseDocument objects
db.insert_data(db_docs)

# Check status at any point
print(doc)  # Shows processing status (e.g., "Document(id=abc123..., source=example.pdf, status=[converted, extracted, chunked(5), embedded(5)])")
doc.validate()  # Raises ValueError if invalid state

# Pydantic features
doc_dict = doc.model_dump()  # Convert to dict
doc_json = doc.model_dump_json()  # Convert to JSON string
doc_copy = Document.model_validate(doc_dict)  # Recreate from dict

# Save and load
doc.save("document.json")  # Save to JSON file (content is base64-encoded)
doc2 = Document.from_file("document.json")  # Load from JSON file
```

## Design Decisions

### Pydantic BaseModel

The `Document` class uses Pydantic `BaseModel` for several benefits:

- **Automatic validation**: Field types are validated on assignment (via `validate_assignment=True`)
- **Serialization**: Built-in `.model_dump()` and `.model_dump_json()` for easy conversion
- **Type safety**: Better IDE support and runtime type checking
- **Mutability**: The class is intentionally mutable to allow each processing stage to modify it in place, providing a clear data flow and avoiding multiple intermediate representations

### Field Prefixes

No prefixes are used on field names (unlike `DatabaseDocument` which uses `default_` prefixes). The `Document` class represents the logical document model, while `DatabaseDocument` represents the storage model.

### Validation

The `validate()` method enforces logical dependencies:

- Cannot have chunks without markdown
- Cannot have metadata without markdown

### Status Tracking

Helper methods allow checking document state without directly inspecting fields:

- `is_converted() -> bool`: Returns True if document has markdown
- `is_extracted() -> bool`: Returns True if metadata has been extracted
- `is_chunked() -> bool`: Returns True if document has chunks
- `is_ready_for_storage() -> bool`: Returns True if document has markdown, metadata, and chunks (chunks must have embeddings)

### Serialization

The Document class provides methods for saving and loading:

- `save(filepath: str) -> None`: Save document to JSON file (content is base64-encoded, images/tables/stats/embeddings are not saved)
- `load(filepath: str) -> None`: Load document from JSON file (updates current instance)
- `from_file(filepath: str) -> Document`: Class method to create a new Document from a JSON file

### Database Conversion

- `to_database_documents() -> list[DatabaseDocument]`: Converts document to a list of `DatabaseDocument` objects for database insertion. Each `DatabaseDocument` represents one chunk. Sparse embeddings are optional (computed by the database via BM25 functions). Raises ValueError if document is not ready for storage (missing chunks or embeddings).
