# Document Processing Pipeline

This directory contains the core logic for processing documents, from initial conversion to content extraction and embedding generation. The system is designed to be modular, allowing for different implementations of converters, extractors, and language models.

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

## Integration Examples

### Custom Processing Pipeline

```python
from crawler.processing import create_converter, get_llm, BasicExtractor
from crawler.processing.embeddings import OllamaEmbedder

# Create components
converter = create_converter("pymupdf", converter_config)
llm = get_llm(llm_config)
extractor = BasicExtractor(metadata_schema, llm)
embedder = OllamaEmbedder(embedding_config)

# Process a document
markdown_content = converter.convert("document.pdf")
metadata = extractor.extract_metadata(markdown_content)
chunks = extractor.chunk_text(markdown_content, chunk_size=1000)

# Generate embeddings
embeddings = [embedder.embed(chunk) for chunk in chunks]
```

### Extending the Pipeline

#### Custom Converter

```python
from crawler.processing.converter import Converter, ConverterConfig

class MyCustomConverter(Converter):
    def convert(self, filepath: str) -> str:
        # Custom conversion logic
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Process content and return markdown
        return self._custom_to_markdown(content)

    def _custom_to_markdown(self, content: str) -> str:
        # Your custom processing logic
        return processed_markdown
```

#### Custom Extractor

```python
from crawler.processing.extractor import Extractor

class DomainSpecificExtractor(Extractor):
    def __init__(self, llm, domain_knowledge: dict):
        super().__init__()
        self.llm = llm
        self.domain_knowledge = domain_knowledge

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        # Use domain knowledge to improve extraction
        prompt = self._build_domain_prompt(text)
        response = self.llm.invoke(prompt)

        # Process and validate response
        return self._parse_domain_response(response)
```

---

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Check Ollama service is running
   - Verify model is downloaded and available
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