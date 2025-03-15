# Document Processing System

This module provides a framework for extracting text and metadata from various document formats, processing the content with LLMs, and generating embeddings for search and retrieval.

## Overview

The system consists of three main components:

1. **Document Handlers** - Extract content from various file formats (PDF, TXT, HTML, etc.)
2. **Extractors** - Process document content with LLMs to extract structured metadata
3. **Embeddings** - Generate vector embeddings for document content

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Basic Document Processing

```python
from processing.extractors import LLMHandler

# Initialize the handler with your schema and model
handler = LLMHandler(
    schema_path="crawler/src/storage/document.json",
    model_name="llama3.2:1b",
    base_url="http://localhost:11434"
)

# Process a list of files
files_to_process = ["document1.pdf", "document2.txt", "document3.html"]
for text, metadata in handler.extract(files_to_process):
    print(f"Extracted metadata: {metadata}")
    # Process or store the extracted text and metadata
```

### Generating Embeddings

```python
from processing.embeddings import LocalEmbedder

# Initialize the embedder
embedder = LocalEmbedder(
    source="ollama",  # or "openai"
    model_name="all-minilm:v2"
)

# Generate embeddings for a query
query_embedding = embedder.embed_query("What is machine learning?")

# Generate embeddings for documents
documents = ["Document 1 content", "Document 2 content"]
doc_embeddings = embedder.embed_documents(documents)
```

## Supported File Types

The system currently supports the following file types:

- PDF (`.pdf`)
- Plain Text (`.txt`)
- Markdown (`.md`)
- HTML (`.html`)
- CSV (`.csv`)
- JSON (`.json`)
- Microsoft Word (`.docx`) - Requires additional dependencies
- Microsoft PowerPoint (`.pptx`) - Requires additional dependencies

## Configuration

The system can be configured using a YAML or JSON schema file that defines the metadata structure to extract from documents.

## Advanced Usage

### Custom Document Handlers

You can create custom document handlers by extending the `DocumentReader` base class:

```python
from processing.document_handlers import DocumentReader, DocumentContent

class CustomFormatReader(DocumentReader):
    def read(self, file_path: str) -> DocumentContent:
        content = DocumentContent()
        # Implement your custom reading logic here
        return content
```

### Using Different LLM Providers

The system supports multiple LLM providers:

```python
# Using Ollama (local)
handler = LLMHandler(
    model_name="llama3.2:1b",
    base_url="http://localhost:11434"
)

# Using Groq
handler = LLMHandler(
    model_name="llama-3.3-70b-versatile",
    model_provider="groq"
)
```

## Project Structure

- `extractors.py` - Main document processing and metadata extraction
- `document_handlers.py` - File format specific document readers
- `embeddings.py` - Vector embedding generation for documents

# Proposed Project Structure

Here's a suggested reorganization of the code to make it more maintainable:

```
crawler/
├── src/
│   ├── processing/
│   │   ├── __init__.py                 # Package initialization
│   │   ├── config.py                   # Centralized configuration handling
│   │   ├── extractors.py               # Main document processing (simplified)
│   │   ├── embeddings.py               # Vector embedding generation
│   │   ├── document_content.py         # DocumentContent class definition
│   │   ├── readers/                    # Document readers subpackage
│   │   │   ├── __init__.py             # Exports all readers
│   │   │   ├── base.py                 # Base DocumentReader class
│   │   │   ├── text.py                 # Text file reader
│   │   │   ├── pdf.py                  # PDF file reader
│   │   │   ├── markdown.py             # Markdown file reader
│   │   │   ├── html.py                 # HTML file reader
│   │   │   ├── csv.py                  # CSV file reader
│   │   │   ├── json.py                 # JSON file reader
│   │   │   ├── docx.py                 # Word document reader
│   │   │   └── pptx.py                 # PowerPoint reader
│   │   ├── llm/                        # LLM handling subpackage
│   │   │   ├── __init__.py             # Package initialization
│   │   │   ├── base.py                 # Base LLM interface
│   │   │   ├── ollama.py               # Ollama-specific implementation
│   │   │   ├── groq.py                 # Groq-specific implementation
│   │   │   └── vision.py               # Vision LLM implementation
│   │   └── utils/                      # Utility functions
│   │       ├── __init__.py             # Package initialization
│   │       ├── image_processing.py     # Image processing utilities
│   │       └── schema.py               # Schema handling utilities
│   └── storage/                        # Storage-related code
│       └── document.json               # Document schema
├── examples/                           # Example usage scripts
│   ├── process_documents.py            # Example document processing
│   └── generate_embeddings.py          # Example embedding generation
└── requirements.txt                    # Project dependencies
```

## Key Improvements

1. **Modular Organization**: Each document reader is in its own file, making it easier to maintain.

2. **Clear Separation of Concerns**:

   - `readers/` - Only responsible for reading documents
   - `llm/` - Handles LLM interactions
   - `utils/` - Contains shared utility functions

3. **Simplified Main Files**:

   - `extractors.py` becomes more focused on orchestration
   - `document_content.py` contains just the DocumentContent class

4. **Centralized Configuration**:

   - `config.py` handles all configuration loading

5. **Examples Directory**:
   - Provides clear usage examples for new users

This structure makes the codebase more:

- Maintainable: Smaller, focused files
- Extensible: Easy to add new readers or LLM providers
- Readable: Clear separation of responsibilities
- Testable: Components can be tested in isolation
