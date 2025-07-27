# Document Processing Pipeline

This directory contains the core logic for processing documents, from initial conversion to content extraction and embedding generation. The system is designed to be modular, allowing for different implementations of converters, extractors, and language models.

## Overview

The document processing pipeline consists of four main stages:

1.  **Conversion**: Raw source files (like PDFs) are converted into a standardized Markdown format.
2.  **Extraction**: Structured metadata is extracted from the Markdown content using a Large Language Model (LLM). The text is also broken down into smaller, manageable chunks.
3.  **LLM Interaction**: A dedicated module handles all communication with LLMs, supporting various providers and features like structured JSON output.
4.  **Embedding**: Text chunks are transformed into numerical vector embeddings for similarity search.

---

## Key Modules

### `converter.py`

This module is responsible for converting various document formats into clean Markdown.

-   **`Converter` (Abstract Base Class)**: Defines the standard interface for all converter implementations.
-   **Implementations**:
    -   `MarkItDownConverter`: Uses the `markitdown` library for conversion.
    -   `DoclingConverter`: Leverages the `docling` library, specializing in PDF processing with VLM (Vision Language Model) integration.
    -   `PyMuPDFConverter`: A comprehensive converter that uses `PyMuPDF` to extract not only text but also tables and images. It includes an `ImageDescriptionInterface` to generate text descriptions for images using a VLM, embedding them directly into the Markdown output.
-   **Factory Function**: `create_converter` provides a simple way to instantiate a specific converter based on a configuration string.

### `extractor.py`

This module extracts structured information from the converted Markdown text.

-   **`Extractor` (Abstract Base Class)**: Defines the interface for metadata extraction and text chunking.
-   **`BasicExtractor`**: A standard implementation that uses an LLM and a user-provided JSON schema to extract structured metadata from text. It also provides a simple method for splitting text into fixed-size chunks.
-   **`MultiSchemaExtractor`**: An extractor that can apply multiple JSON schemas sequentially to extract a broader range of metadata from a single document.

### `llm.py`

This module provides a standardized interface for interacting with Large Language Models.

-   **`LLM` (Abstract Base Class)**: Defines the core `invoke` method for sending prompts to a model.
-   **`OllamaLLM`**: An implementation for interacting with models served via the Ollama platform. It robustly handles various tasks:
    -   Simple text generation from a prompt.
    -   Conversational history.
    -   Structured JSON output based on a provided schema.
    -   Request timeouts to prevent indefinite hangs.
-   **`LLMConfig`**: A dataclass for configuring LLM clients, specifying the model, URL, system prompt, and other parameters.

### `embeddings.py`

This module is responsible for generating vector embeddings from text chunks.

-   **`Embedder` (Abstract Base Class)**: Defines the interface for embedding models, including methods to generate an embedding and retrieve the model's vector dimension.
-   **`OllamaEmbedder`**: An implementation that uses `langchain_ollama` to generate embeddings from models hosted on an Ollama server.
-   **`EmbedderConfig`**: A dataclass for configuring the embedding model, provider, and connection details.