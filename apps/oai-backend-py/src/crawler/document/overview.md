# Document Module

## Overview

The document module provides a unified `Document` class that serves as the central data structure for the document processing pipeline. This class is designed to be mutable and flows through multiple processing stages, with each stage populating specific fields.

## Files

### `__init__.py`

Exports the `Document` class for easy importing by other modules.

### `document.py`

Contains the core `Document` class (Pydantic BaseModel) with the following responsibilities:

- Defines the document structure with fields for all processing stages
- Provides automatic validation and serialization via Pydantic
- Provides factory methods for creating new documents
- Implements validation and status checking methods
- Tracks document state through the pipeline

### `integration_examples.py`

Provides example integration methods for each processing module:

- `convert_document()`: Example method for Converter class
- `extract_document()`: Example method for MetadataExtractor class
- `chunk_document()`: Example method for Chunker class
- `store_document()`: Example method for DatabaseClient class
- `process_document_pipeline()`: Complete end-to-end pipeline example

These examples demonstrate the pattern for adding Document support to each module.

### `test_document.py`

Unit tests demonstrating Document functionality:

- `test_document_creation()`: Tests basic document creation and factory method
- `test_document_pipeline_states()`: Tests state transitions through the pipeline
- `test_document_validation()`: Tests validation logic and error handling
- `test_document_repr()`: Tests string representation at various states
- `test_pydantic_features()`: Tests Pydantic serialization and validation features

Run tests with: `uv run python -m crawler.document.test_document`

## Document Processing Pipeline

The `Document` class flows through the following stages:

1. **Creation**: Document is created with `document_id` and `source`
2. **Converter**: Populates `content` (bytes) and `markdown` (str)
3. **Extractor**: Populates `metadata` (dict) and `benchmark_questions` (list)
4. **Chunker**: Populates `chunks` (list of strings)
5. **Vector DB**: Reads all fields to create `DatabaseDocument` instances for storage

## Document Class Fields

### Required (at creation)

- `document_id`: Unique identifier (UUID string)
- `source`: Source identifier (file path, URL, etc.)

### Converter Stage

- `content`: Raw binary content of the document
- `markdown`: Markdown representation of the document

### Extractor Stage

- `metadata`: Extracted structured metadata (dict)
- `benchmark_questions`: Generated questions for testing (list of strings)

### Chunker Stage

- `chunks`: Text chunks for embedding and storage (list of strings)

### Optional

- `security_group`: RBAC access control groups (defaults to ["public"])
- `minio_url`: URL to document in object storage

## Usage Example

```python
from crawler.document import Document
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractor
from crawler.chunker import Chunker
from crawler.vector_db import MilvusClient

# 1. Create document
doc = Document.create(source="example.pdf")

# 2. Convert to markdown
config = PyMuPDF4LLMConfig(type="pymupdf4llm")
converter = create_converter(config)
converter.convert_document(doc)  # Modifies doc in place

# 3. Extract metadata
extractor = MetadataExtractor(llm, config)
extractor.extract_document(doc)  # Modifies doc in place

# 4. Chunk the markdown
chunker = Chunker(config)
chunker.chunk_document(doc)  # Modifies doc in place

# 5. Store in vector database
db = MilvusClient(config)
db.store_document(doc)  # Converts to DatabaseDocument and stores

# Check status at any point
print(doc)  # Shows processing status
doc.validate()  # Raises if invalid state

# Pydantic features
doc_dict = doc.model_dump()  # Convert to dict
doc_json = doc.model_dump_json()  # Convert to JSON string
doc_copy = Document.model_validate(doc_dict)  # Recreate from dict
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

Helper methods (`is_converted()`, `is_extracted()`, `is_chunked()`, `is_ready_for_storage()`) allow checking document state without directly inspecting fields.
