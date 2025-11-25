# Document Processing Pipeline

This directory contains the core logic for processing documents, from initial conversion to content extraction and embedding generation. The system is designed to be modular, allowing for different implementations of converters, extractors, and language models.

All configuration models use Pydantic BaseModels for type safety and automatic validation.

## Overview

The document processing system transforms raw files into structured, searchable data through a multi-stage pipeline:

1. **Document Conversion**: Transform various file formats (PDF, DOCX, etc.) into standardized Markdown
2. **Metadata Extraction**: Use LLMs to extract structured information from documents
3. **Text Chunking**: Split documents into manageable pieces for processing
4. **Embedding Generation**: Convert text chunks into vector representations
5. **Quality Assurance**: Validate extracted data and handle errors gracefully

The modular design allows you to swap out components for different use cases while maintaining a consistent interface.

## Core Processing Stages

1.  **Conversion**: Raw source files (like PDFs) are converted into a standardized Markdown format.
2.  **Extraction**: Structured metadata is extracted from the Markdown content using a Large Language Model (LLM). The text is also broken down into smaller, manageable chunks.
3.  **LLM Interaction**: A dedicated module handles all communication with LLMs, supporting various providers and features like structured JSON output.
4.  **Embedding**: Text chunks are transformed into numerical vector embeddings for similarity search.

---

## Key Modules

### `converter/`

This package is responsible for converting various document formats into clean Markdown. See `converter/overview.md` for detailed documentation.

- **`Converter` (Abstract Base Class)**: Defines the standard interface for all converter implementations.
- **Implementations**:
  - `MarkItDownConverter`: Uses the `markitdown` library for conversion with vision model support.
  - `PyMuPDF4LLMConverter`: A comprehensive converter that uses `PyMuPDF` and `pymupdf4llm` to extract text, tables, and images. Includes VLM integration for image description.
- **Factory Function**: `create_converter(config: ConverterConfig)` provides type-safe converter instantiation.
- **Configuration**: `MarkItDownConfig`, `PyMuPDF4LLMConfig` (discriminated union via `ConverterConfig`)

### `extractor/`

This module extracts structured information from the converted Markdown text. See `extractor/extractor.py` for implementation.

- **`MetadataExtractor`**: Single-schema metadata extractor that uses an LLM and a user-provided JSON schema to extract structured metadata from text.
- **`MetadataExtractorConfig`**: Pydantic configuration for extractor settings including schema, context, structured output mode, and benchmark question generation.
- **Features**:
  - Supports both `response_format` (json_schema) and `tools` structured output modes
  - Strict mode for schema enforcement (drops extra keys, fills missing required fields)
  - Optional benchmark question generation
  - Document truncation for large documents
  - Optional jsonschema validation

### `llm/`

This package provides standardized interfaces for interacting with Large Language Models and embedding models.

**`llm.py`**:
- **`LLM` (Abstract Base Class)**: Defines the core `invoke` method for sending prompts to a model.
- **`OllamaLLM`**: Interacts with models served via the Ollama platform. Supports:
  - Simple text generation
  - Conversational history
  - Structured JSON output via `response_format` or `tools`
  - Request timeouts and detailed logging
- **`VllmLLM`**: Interacts with models served via vLLM using the OpenAI-compatible `/v1/chat/completions` API. Supports the same structured-output modes.
- **`LLMConfig`**: A Pydantic BaseModel for configuring LLM clients with factory methods (`ollama()`, `openai()`, `vllm()`)

**`embeddings.py`**:
- **`Embedder` (Abstract Base Class)**: Defines the interface for embedding models.
- **`OllamaEmbedder`**: Uses `ollama` library to generate embeddings from models hosted on an Ollama server.
- **`EmbedderConfig`**: A Pydantic BaseModel for configuring the embedding model with factory methods (`ollama()`, `openai()`)
- **Factory Function**: `get_embedder(config: EmbedderConfig)` returns the appropriate embedder implementation

### `chunker/`

This module provides text chunking functionality. See `chunker/overview.md` for detailed documentation.

- **`Chunker`**: Text chunker with configurable chunk sizes and strategies.
- **`ChunkingConfig`**: Pydantic configuration for chunking parameters (chunk_size, overlap, strategy, preserve_paragraphs, min_chunk_size)
- **Features**: Naive chunking strategy with paragraph/sentence/word boundary preservation

### `document/`

This module provides the unified `Document` class that flows through the processing pipeline. See `document/overview.md` for detailed documentation.

- **`Document`**: Pydantic BaseModel that serves as the central data structure
- **Features**: Mutable design, automatic validation, serialization, status tracking, database entity conversion

### `vector_db/`

This module provides a type-safe interface for storing document chunks in vector databases. See `vector_db/overview.md` for detailed documentation.

- **`DatabaseClient`**: Abstract interface for database implementations
- **`MilvusDB`**: Milvus implementation with hybrid search (dense + sparse BM25)
- **`DatabaseDocument`**: Pydantic model for document chunks with `default_` prefixed system fields
- **`DatabaseClientConfig`**: Pydantic configuration for database connections
- **Factory Functions**: `get_db()`, `get_db_benchmark()`

### `main.py`

This module contains the main `Crawler` class that orchestrates the entire document processing pipeline.

- **`Crawler`**: Main orchestrator class that coordinates all processing stages
- **`CrawlerConfig`**: Pydantic configuration model that aggregates all component configurations
- **Features**:
  - File/directory crawling with duplicate detection
  - Caching of processed documents
  - Progress tracking with tqdm
  - Statistics collection
  - Benchmark support

### `collection_config.py`

This module provides the `CollectionConfig` class for storing and retrieving collection settings.

- **`CollectionConfig`**: Pydantic model that encapsulates all configuration needed for a collection
- **Features**:
  - Stores metadata schema, library context, crawler config, and LLM system prompt
  - Serializes to/from JSON for storage in collection descriptions
  - Enables easy recreation of collection setups

### `collection_manager.py`

This module provides the `CollectionManager` class for simplified collection management.

- **`CollectionManager`**: High-level API for managing collections and adding documents
- **Features**:
  - Create/recreate collections with stored configurations
  - Load collection configurations from database descriptions
  - Add documents to existing collections
  - Support for custom converters and extractors
  - All configuration stored as JSON in collection descriptions

---

## Pydantic Configuration Models

All configuration classes in the processing module use Pydantic BaseModels for enhanced type safety and validation:

### Benefits

- **Automatic Validation**: All fields are validated at creation time
- **Type Safety**: Runtime type checking prevents configuration errors
- **Field Constraints**: Min/max values, string lengths, numeric ranges
- **Clear Error Messages**: Validation errors include detailed information
- **IDE Support**: Full autocomplete and type hints
- **Serialization**: Easy conversion to/from JSON and dictionaries

### Configuration Models

- `LLMConfig`: Validates model names, URLs, timeouts, context lengths, structured output mode
- `EmbedderConfig`: Validates model, base URL, and embedding dimension
- `ConverterConfig`: Discriminated union of converter configs (MarkItDownConfig, PyMuPDF4LLMConfig)
- `MetadataExtractorConfig`: Validates JSON schema, context, structured output mode, benchmark settings
- `ChunkingConfig`: Validates chunk size, overlap, strategy, and boundary preservation settings
- `DatabaseClientConfig`: Validates database provider, collection, connection parameters
- `CrawlerConfig`: Aggregates all component configurations with validation

### Example Validation

```python
from crawler.llm import LLMConfig
from pydantic import ValidationError

# Valid configuration using factory method
config = LLMConfig.ollama(model_name="llama3.2")

# Invalid configuration will raise ValidationError
try:
    bad_config = LLMConfig(
        model_name="",  # Empty string not allowed
        ctx_length=-1,  # Must be positive
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # Output shows which fields are invalid and why
```

---

## Error Handling and Best Practices

### Error Handling

The processing pipeline includes comprehensive error handling:

- **Connection Errors**: Automatic retry logic for LLM and database connections
- **Schema Validation**: JSON schema validation for extracted metadata
- **File Processing Errors**: Graceful handling of corrupted or unsupported files
- **Timeout Management**: Configurable timeouts for long-running operations
- **Resource Management**: Proper cleanup of temporary files and connections

### Best Practices

#### 1. Configuration Management

```python
# Use environment variables for sensitive data
import os

config = {
    "llm": {
        "model_name": os.getenv("LLM_MODEL", "llama3.2"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_timeout": 300.0,
    },
    "database": {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "password": os.getenv("MILVUS_PASSWORD"),
    }
}
```

#### 2. Chunk Size Optimization

- **Small chunks (500-1000 tokens)**: Better for precise search and retrieval
- **Large chunks (2000+ tokens)**: Better for maintaining context and coherence
- **Consider your use case**: Research papers benefit from larger chunks, while documentation works well with smaller chunks

#### 3. Schema Design

```python
# Good schema design
metadata_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 500},
        "authors": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "maxLength": 2000},
        "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 10}
    },
    "required": ["title"],
    "additionalProperties": False  # Prevent unexpected fields
}
```

#### 4. Memory Management

- Process documents in batches for large collections
- Use streaming for very large files
- Monitor memory usage with logging
- Clean up temporary files regularly

#### 5. Performance Optimization

- Use appropriate embedding models for your domain
- Batch embedding requests when possible
- Enable caching for repeated operations
- Use partitions for large document collections

---

## Simplified Collection API

The `CollectionManager` provides a simplified interface for creating collections and adding documents. All configuration is stored in the collection description, making it easy to reuse collections.

### Basic Usage

```python
from crawler import CollectionManager, CollectionConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.chunker import ChunkingConfig
from crawler.main import CrawlerConfig

# Create collection configuration
embeddings = EmbedderConfig.ollama(model="all-minilm:v2")
llm = LLMConfig.ollama(model_name="llama3.2:3b")
vision_llm = LLMConfig.ollama(model_name="llava:latest")
database = DatabaseClientConfig.milvus(collection="research_papers")
converter = PyMuPDF4LLMConfig(type="pymupdf4llm", vlm_config=vision_llm)
extractor = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title"]
    },
    context="Research papers on machine learning"
)
chunking = ChunkingConfig.create(chunk_size=1000)

crawler_config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,
    extractor=extractor,
    chunking=chunking,
    metadata_schema=extractor.json_schema,
)

collection_config = CollectionConfig(
    metadata_schema=extractor.json_schema,
    library_context="Research papers on machine learning",
    crawler_config=crawler_config,
    llm_system_prompt="This collection contains AI research papers. Use metadata filtering to retrieve documents by title, author, and keywords."
)

# Initialize manager
manager = CollectionManager(
    db_host="localhost",
    db_port=19530,
    db_username="root",
    db_password="Milvus"
)

# Create collection
manager.create_collection(
    config=collection_config,
    collection_name="research_papers",
    partition="ml_papers",
    recreate=False
)

# Add documents later (configuration is automatically loaded from collection description)
manager.add_documents(
    path="path/to/pdfs",
    collection_name="research_papers",
    partition="ml_papers"
)

# Add documents with custom converter
from crawler.converter import Converter
from crawler.document import Document
from crawler.converter.types import ConversionStats
import json

class JSONConverter(Converter):
    @property
    def name(self) -> str:
        return "JSONConverter"
    
    def convert(self, document: Document) -> None:
        # Custom conversion logic for JSON files
        # Read content if not already set
        if document.content is None:
            from pathlib import Path
            source_path = Path(document.source)
            document.content = source_path.read_bytes()
        
        # Parse JSON from content
        data = json.loads(document.content.decode('utf-8'))
        
        # Convert JSON to markdown
        markdown = self._json_to_markdown(data)
        
        # Populate document fields
        document.markdown = markdown
        document.stats = ConversionStats(total_pages=1, processed_pages=1)
        if document.source_name is None:
            from pathlib import Path
            source_path = Path(document.source)
            document.source_name = source_path.name if source_path.name else document.source.split("/")[-1]
    
    def _json_to_markdown(self, data: dict) -> str:
        # Your custom conversion logic
        return "# Document\n\n" + json.dumps(data, indent=2)

custom_converter = JSONConverter(None)
manager.add_documents(
    path="path/to/json",
    collection_name="research_papers",
    partition="ml_papers",
    custom_converter=custom_converter
)

# Load and recreate collection
config = manager.load_collection("research_papers", partition="ml_papers")
manager.recreate_collection("research_papers", partition="ml_papers")

# List all collections
collections = manager.list_collections()
```

### Key Benefits

1. **Self-contained Configuration**: All settings stored in collection description
2. **Easy Reuse**: Load collection config and add new documents with same settings
3. **Custom Processing**: Pass custom converters/extractors when needed
4. **Simple API**: High-level methods hide complexity of underlying system

## Integration Examples

### Custom Processing Pipeline

```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.llm import get_llm, LLMConfig, get_embedder, EmbedderConfig
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.chunker import Chunker, ChunkingConfig
from crawler.document import Document

# Create components
converter = create_converter(PyMuPDF4LLMConfig(type="pymupdf4llm", vlm_config=LLMConfig.ollama(model_name="llava")))
llm = get_llm(LLMConfig.ollama(model_name="llama3.2"))
extractor_config = MetadataExtractorConfig(json_schema={...}, context="")
extractor = MetadataExtractor(llm=llm, config=extractor_config)
embedder = get_embedder(EmbedderConfig.ollama(model="all-minilm:v2"))
chunker = Chunker(ChunkingConfig.create(chunk_size=1000))

# Process a document
doc = Document.create(source="document.pdf")
converter.convert_document(doc)  # Populates markdown, images, tables, stats
result = extractor.run(doc)  # Returns MetadataExtractionResult
doc.metadata = result.metadata
doc.chunks = chunker.chunk_text(doc)  # Returns list[str]
doc.text_embeddings = embedder.embed_batch(doc.chunks)  # Returns list[list[float]]
```

### Extending the Pipeline

#### Custom Converter

```python
from crawler.converter import Converter
from crawler.document import Document
from crawler.converter.types import ConversionStats

class MyCustomConverter(Converter):
    @property
    def name(self) -> str:
        return "MyCustomConverter"
    
    def convert(self, document: Document) -> None:
        # Custom conversion logic
        # Read content if not already set
        if document.content is None:
            from pathlib import Path
            source_path = Path(document.source)
            document.content = source_path.read_bytes()
        
        # Process content
        content = document.content.decode('utf-8')
        markdown = self._custom_to_markdown(content)
        
        # Populate document fields
        document.markdown = markdown
        document.stats = ConversionStats(total_pages=1, processed_pages=1)
        if document.source_name is None:
            from pathlib import Path
            source_path = Path(document.source)
            document.source_name = source_path.name if source_path.name else document.source.split("/")[-1]

    def _custom_to_markdown(self, content: str) -> str:
        # Your custom processing logic
        return processed_markdown
```

#### Custom Extractor

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import LLM

class DomainSpecificExtractor(MetadataExtractor):
    def __init__(self, llm: LLM, config: MetadataExtractorConfig, domain_knowledge: dict):
        super().__init__(llm, config)
        self.domain_knowledge = domain_knowledge

    def extract(self, markdown: str) -> dict[str, Any]:
        # Use domain knowledge to improve extraction
        prompt = self._build_domain_prompt(markdown)
        response = self.llm.invoke(prompt)

        # Process and validate response
        return self._parse_domain_response(response)
```

---

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**

   - For Ollama: ensure the Ollama daemon is running and the model is pulled
   - For vLLM: ensure the server exposes the OpenAI-compatible `/v1` endpoints
   - Check network connectivity and firewall settings

2. **Memory Issues**

   - Reduce batch sizes
   - Process files individually for large documents
   - Use streaming for very large files

3. **Schema Validation Errors**

   - Review your metadata schema for consistency
   - Check that required fields are always present
   - Validate schema syntax with a JSON schema validator

4. **Embedding Errors**
   - Verify embedding model compatibility
   - Check model dimensions match database schema
   - Ensure text chunks are not too long for the model

### Debugging Tips

- Enable detailed logging: `config.log_level = "DEBUG"`
- Use smaller test documents first
- Validate each processing stage independently
- Check system resources (CPU, memory, disk space)
- Review error logs for specific error messages
