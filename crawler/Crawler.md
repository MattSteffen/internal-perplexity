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

All configuration is managed through the `CrawlerConfig` dataclass, which is instantiated from a dictionary. This object centralizes the settings for all sub-components.

### Example Configuration

```python
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    },
    "vision_llm": {
        "model": "gemma3",
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
            "title": { "type": "string" },
            "author": { "type": "string" }
        },
        "required": ["title", "author"]
    },
    "converter": {
        "type": "pymupdf",
        # ... other converter-specific options
    },
    "chunk_size": 1000,
    "temp_dir": "tmp/"
}

# How to instantiate
# from src.crawler import CrawlerConfig
# crawler_config = CrawlerConfig.from_dict(config_dict)
```

---

## API Reference

### `class Crawler(config, converter, extractor, vector_db, embedder, llm)`

The main orchestrator of the processing pipeline.

**`__init__(self, config, converter=None, ...)`**

Initializes the crawler. While you can inject custom component instances, the common pattern is to provide a complete `config` object and let the `Crawler` initialize its own defaults.

-   **`config` (`CrawlerConfig`)**: The main configuration object.
-   **`converter` (`Converter`, optional)**: A custom instance of a `Converter`. If `None`, a default is created based on the config.
-   **`extractor` (`Extractor`, optional)**: A custom instance of an `Extractor`. Defaults to `BasicExtractor`.
-   **`vector_db` (`DatabaseClient`, optional)**: A custom instance of a `DatabaseClient`. If `None`, a default is created via `get_db`.
-   **`embedder` (`Embedder`, optional)**: A custom instance of an `Embedder`. If `None`, a default is created via `get_embedder`.
-   **`llm` (`LLM`, optional)**: A custom instance of an `LLM`. If `None`, a default is created via `get_llm`.

**`crawl(self, path)`**

Starts the crawling process on the specified file(s) or directory.

-   **`path` (`Union[str, List[str]]`)**: A single absolute file path, a directory path, or a list of absolute file paths to process.

---

## Extensibility and Interfaces

The crawler is built on a set of abstract base classes (interfaces) that allow for custom implementations. This makes the system highly extensible.

### `processing.Converter`

-   **Purpose**: Defines the contract for converting source files into Markdown.
-   **Method**: `convert(self, filepath: str) -> str`
-   **Implementations**: `MarkItDownConverter`, `DoclingConverter`, `PyMuPDFConverter`.

### `processing.Extractor`

-   **Purpose**: Defines how to extract metadata and chunk text.
-   **Methods**:
    -   `extract_metadata(self, text: str) -> Dict[str, Any]`
    -   `chunk_text(self, text: str, chunk_size: int) -> List[str]`
-   **Implementations**: `BasicExtractor`, `MultiSchemaExtractor`.

### `processing.Embedder`

-   **Purpose**: Defines the interface for text embedding models.
-   **Methods**:
    -   `embed(self, query: str) -> List[float]`
    -   `get_dimension(self) -> int`
-   **Implementations**: `OllamaEmbedder`.

### `processing.LLM`

-   **Purpose**: Defines a standard way to interact with Large Language Models.
-   **Method**: `invoke(self, prompt_or_messages, ...)`
-   **Implementations**: `OllamaLLM`.

### `storage.DatabaseClient`

-   **Purpose**: Defines the interface for a vector database client.
-   **Methods**:
    -   `insert_data(self, data: List[DatabaseDocument])`
    -   `check_duplicate(self, source: str, chunk_index: int) -> bool`
-   **Implementations**: `MilvusDB`.

### `storage.DatabaseDocument`

-   **Purpose**: A dataclass that defines the standard structure for documents being inserted into the database, ensuring consistency.
-   **Fields**: `text`, `text_embedding`, `chunk_index`, `source`, `metadata`.