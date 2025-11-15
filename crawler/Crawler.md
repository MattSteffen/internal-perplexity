# Crawler Engine Documentation

## Overview

The Crawler is a flexible and extensible system designed to process local files, extract their content and metadata, and store them in a vector database for retrieval and analysis. It follows a modular architecture, allowing developers to easily swap out components for different technologies (e.g., document converters, LLMs, vector databases).

The system is orchestrated by the `Crawler` class, which uses a `CrawlerConfig` object to manage all settings.

---

## Core Workflow

The `crawl` method executes the following steps for each file:

1.  **Check for Duplicates**: Skips processing if the file has already been indexed in the database to prevent redundant work.
2.  **Load from Cache**: If a temporary JSON file exists for the document, it loads the pre-processed markdown and metadata to save time on subsequent runs.
3.  **Convert**: If not cached, it uses the configured `Converter` (e.g., `PyMuPDFConverter`) to transform the source file (like a PDF) into a standardized Markdown format.
4.  **Extract Metadata**: It uses the configured `Extractor` (e.g., `BasicExtractor`) and a Large Language Model (LLM) to pull structured metadata from the Markdown content. This extraction is guided by a user-provided JSON schema.
5.  **Cache Results**: The extracted Markdown and metadata are saved to a temporary file in the directory specified by `temp_dir` to speed up future processing.
6.  **Chunk Text**: The Markdown text is broken down into smaller, manageable chunks based on the configured `chunk_size`.
7.  **Embed Chunks**: Each text chunk is converted into a numerical vector embedding using the configured `Embedder`.
8.  **Store in Database**: The chunks, their embeddings, and the extracted metadata are formatted into `DatabaseDocument` objects and inserted into the configured vector database (e.g., Milvus).

---

## Configuration (`CrawlerConfig`)

All configuration is managed through the `CrawlerConfig` Pydantic BaseModel, which can be instantiated from a dictionary or using the type-safe `create()` method. This object centralizes the settings for all sub-components.

### Example Configuration (Type-Safe, Recommended)

```python
from crawler import Crawler, CrawlerConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.chunker import ChunkingConfig

# Create component configurations
embeddings = EmbedderConfig.ollama(model="all-minilm:v2", base_url="http://localhost:11434")
llm = LLMConfig.ollama(model_name="gemma3", base_url="http://localhost:11434")
vision_llm = LLMConfig.ollama(model_name="llava", base_url="http://localhost:11434")
database = DatabaseClientConfig.milvus(
    collection="test_collection",
    host="localhost",
    port=19530,
    username="root",
    password="123456",
    recreate=False
)
converter = PyMuPDF4LLMConfig(type="pymupdf4llm", vlm_config=vision_llm)
extractor = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}
        },
        "required": ["title", "author"]
    }
)
chunking = ChunkingConfig.create(chunk_size=1000)

# Create complete configuration
config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,
    extractor=extractor,
    chunking=chunking,
    metadata_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}
        },
        "required": ["title", "author"]
    },
    temp_dir="tmp/"
)
```

### Example Configuration (Dictionary-Based, Backward Compatible)

```python
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
    },
    "vision_llm": {
        "model_name": "llava",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "username": "root",
        "password": "123456",
        "collection": "test_collection",
        "recreate": False,
    },
    "llm": {
        "model_name": "gemma3",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}
        },
        "required": ["title", "author"]
    },
    "converter": {
        "type": "pymupdf4llm",
        "vlm_config": {
            "model_name": "llava",
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        }
    },
    "chunking": {
        "chunk_size": 1000,
        "overlap": 200
    },
    "temp_dir": "tmp/"
}

# How to instantiate
from crawler import CrawlerConfig
crawler_config = CrawlerConfig.from_dict(config_dict)
```

---

## API Reference

### `class Crawler`

The main orchestrator of the processing pipeline.

**`__init__(self, config: CrawlerConfig, converter: Converter = None, extractor: MetadataExtractor = None, vector_db: DatabaseClient = None, embedder: Embedder = None, llm: LLM = None, chunker: Chunker = None)`**

Initializes the crawler. While you can inject custom component instances, the common pattern is to provide a complete `config` object and let the `Crawler` initialize its own defaults.

-   **`config` (`CrawlerConfig`)**: The main configuration object (required).
-   **`converter` (`Converter`, optional)**: A custom instance of a `Converter`. If `None`, a converter is created based on `config.converter`.
-   **`extractor` (`MetadataExtractor`, optional)**: A custom instance of a `MetadataExtractor`. If `None`, an extractor is created based on `config.extractor`.
-   **`vector_db` (`DatabaseClient`, optional)**: A custom instance of a `DatabaseClient`. If `None`, a default is created via `get_db()`.
-   **`embedder` (`Embedder`, optional)**: A custom instance of an `Embedder`. If `None`, a default is created via `get_embedder()`.
-   **`llm` (`LLM`, optional)**: A custom instance of an `LLM`. If `None`, a default is created via `get_llm()`.
-   **`chunker` (`Chunker`, optional)**: A custom instance of a `Chunker`. If `None`, a chunker is created based on `config.chunking`.

**`crawl(self, path: str | list[str]) -> None`**

Starts the crawling process on the specified file(s) or directory. Processes each file through the complete pipeline: conversion, metadata extraction, chunking, embedding, and storage.

-   **`path` (`str | list[str]`)**: A single file path, a directory path, or a list of file paths to process. If a directory, recursively processes all files within it.
-   **Returns**: `None` (modifies database in place)
-   **Features**:
  - Duplicate detection (skips files already in database)
  - Caching (loads from temp_dir if available)
  - Progress tracking with tqdm
  - Error handling (continues processing on individual file failures)
  - Statistics collection

**`benchmark(self, generate_queries: bool = False) -> None`**

Runs benchmarking on the crawled documents if benchmarking was enabled in the config.

-   **`generate_queries` (`bool`)**: Whether to generate new benchmark queries or use stored ones.
-   **Returns**: `None` (saves results to `benchmark_results/` directory)

---

## Extensibility and Interfaces

The crawler is built on a set of abstract base classes (interfaces) that allow for custom implementations. This makes the system highly extensible.

### `converter.Converter`

-   **Purpose**: Defines the contract for converting source files into Markdown.
-   **Methods**: 
    -   `convert(self, doc: DocumentInput) -> ConvertedDocument`
    -   `convert_document(self, document: Document) -> None` (preferred for Document pipeline)
-   **Implementations**: `MarkItDownConverter`, `PyMuPDF4LLMConverter`.

### `extractor.MetadataExtractor`

-   **Purpose**: Extracts structured metadata from markdown text using LLMs.
-   **Methods**:
    -   `extract(self, markdown: str) -> dict[str, Any]`
    -   `run(self, document: Document) -> MetadataExtractionResult`
    -   `generate_benchmark_questions(self, markdown: str, n: int) -> list[str]`
-   **Configuration**: `MetadataExtractorConfig` with JSON schema, context, structured output mode

### `llm.embeddings.Embedder`

-   **Purpose**: Defines the interface for text embedding models.
-   **Methods**:
    -   `embed(self, query: str) -> list[float]`
    -   `embed_batch(self, queries: list[str]) -> list[list[float]]`
    -   `get_dimension(self) -> int`
-   **Implementations**: `OllamaEmbedder`.

### `llm.llm.LLM`

-   **Purpose**: Defines a standard way to interact with Large Language Models.
-   **Method**: `invoke(self, prompt_or_messages, response_format=None, tools=None, ...) -> Any`
-   **Implementations**: `OllamaLLM`, `VllmLLM`.

### `vector_db.DatabaseClient`

-   **Purpose**: Defines the interface for a vector database client.
-   **Methods**:
    -   `create_collection(self, recreate: bool = False) -> None`
    -   `insert_data(self, data: list[DatabaseDocument]) -> None`
    -   `check_duplicate(self, source: str, chunk_index: int) -> bool`
-   **Implementations**: `MilvusDB`.

### `vector_db.DatabaseDocument`

-   **Purpose**: A Pydantic BaseModel that defines the standard structure for documents being inserted into the database.
-   **Required Fields**: `default_document_id`, `default_text`, `default_text_embedding`, `default_chunk_index`, `default_source`, `security_group`, `metadata`
-   **Optional Fields**: `id`, `default_metadata`, `default_minio`, `default_text_sparse_embedding`, `default_metadata_sparse_embedding`, `default_benchmark_questions`
-   **Note**: All system fields use `default_` prefix to avoid conflicts with user metadata.

---

## Database Storage

### Milvus Integration

The crawler uses Milvus as its vector database for storing document embeddings and metadata. The system automatically creates collections with appropriate schemas based on your metadata configuration.

#### Collection Schema

The database schema is automatically generated from your `metadata_schema` configuration. For example:

```python
metadata_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "publication_date": {"type": "string"}
    },
    "required": ["title"]
}
```

This creates a Milvus collection with:
- `id`: Primary key (auto-generated)
- `text`: The document chunk text
- `text_embedding`: Vector embedding (384 dimensions for all-minilm:v2)
- `chunk_index`: Index of the chunk within the document
- `source`: File path of the original document
- `title`: Extracted title
- `author`: Extracted author
- `keywords`: Array of keywords
- `publication_date`: Publication date string

#### Partitioning

You can organize documents into partitions for better performance:

```python
config_dict = {
    "database": {
        "provider": "milvus",
        "collection": "documents",
        "partition": "research_papers",  # Documents will be stored in this partition
        "recreate": False,
    }
}
```

#### Duplicate Handling

The system automatically detects and skips duplicate chunks based on `source` and `chunk_index` to prevent redundant processing.

---

## Benchmarking and Evaluation

### Built-in Benchmarking

The crawler includes comprehensive benchmarking capabilities to evaluate search performance:

```python
# Enable benchmarking
config = CrawlerConfig.from_dict({
    "database": {"provider": "milvus", "collection": "benchmark_test"},
    "embeddings": {"provider": "ollama", "model": "all-minilm:v2"},
    "utils": {
        "benchmark": True,
        "generate_benchmark_questions": True,
        "num_benchmark_questions": 5
    }
})

crawler = Crawler(config)
crawler.crawl("documents/")
results = crawler.benchmark()

# View results
print(f"Top-1 Accuracy: {results.percent_in_top_k[1]:.2f}%")
print(f"Average Search Time: {sum(results.search_time_distribution) / len(results.search_time_distribution):.3f}s")
```

### Benchmark Metrics

- **Placement Distribution**: Shows where relevant results appear in search rankings
- **Distance Distribution**: Analyzes similarity scores between queries and results
- **Search Time Distribution**: Measures query response times
- **Top-K Accuracy**: Percentage of queries with relevant results in top K positions

### Custom Benchmarking

You can also create custom benchmark queries:

```python
from crawler.storage.database_utils import get_db_benchmark

# Create benchmark client
benchmark_client = get_db_benchmark(db_config, embed_config)

# Define custom queries
queries = [
    "What is the main topic of the document?",
    "Who is the author of this paper?",
    "What are the key findings?"
]

# Run custom benchmark
results = benchmark_client.run_benchmark(queries)
```