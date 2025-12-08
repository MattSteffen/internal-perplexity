# Crawler: Document Processing and Vector Database System

A modular Python system for crawling, processing, and indexing documents into vector databases for retrieval-augmented generation (RAG) applications. The crawler converts documents to markdown, extracts structured metadata with LLMs, generates embeddings, and stores everything in Milvus for efficient semantic search.

## Features

- **Universal Document Processing**: Convert PDFs and other formats to searchable markdown with vision-LLM powered image descriptions
- **Intelligent Metadata Extraction**: Use LLMs with JSON Schema validation to extract structured metadata
- **Flexible Chunking**: Split documents with configurable strategies, sizes, and overlap
- **Vector Database Integration**: Store embeddings in Milvus with hybrid search (dense + sparse BM25)
- **Type-Safe Configuration**: Pydantic-based configuration with validation and factory methods
- **Benchmarking**: Built-in tools to evaluate search quality and performance
- **Caching**: Automatic caching of processed documents for faster re-runs
- **Extensible Architecture**: Easy to add custom converters, extractors, and processors

## Quick Start

### Installation

```bash
# Install from source
cd crawler
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

**Requirements:**
- Python 3.10+
- Ollama (for local LLMs and embeddings)
- Milvus (for vector storage)

### Start Required Services

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull all-minilm:v2

# Start Milvus (using Docker Compose)
cd database
docker-compose up -d
```

### Basic Usage

```python
from crawler import Crawler, CrawlerConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.chunker import ChunkingConfig

# Define metadata schema
metadata_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title"]
}

# Create configuration
config = CrawlerConfig.create(
    # Embeddings
    embeddings=EmbedderConfig.ollama(
        model="all-minilm:v2",
        base_url="http://localhost:11434"
    ),
    
    # LLM for metadata extraction
    llm=LLMConfig.ollama(
        model_name="llama3.2:3b",
        base_url="http://localhost:11434"
    ),
    
    # Vision LLM for image descriptions
    vision_llm=LLMConfig.ollama(
        model_name="llava:latest",
        base_url="http://localhost:11434"
    ),
    
    # Database
    database=DatabaseClientConfig.milvus(
        collection="my_documents",
        host="localhost",
        port=19530,
        username="root",
        password="Milvus"
    ),
    
    # Converter
    converter=PyMuPDF4LLMConfig(
        type="pymupdf4llm",
        vlm_config=LLMConfig.ollama(model_name="llava:latest")
    ),
    
    # Extractor
    extractor=MetadataExtractorConfig(
        json_schema=metadata_schema,
        context="Technical documentation and research papers"
    ),
    
    # Chunking
    chunking=ChunkingConfig.create(
        chunk_size=1000,
        overlap=200
    ),
    
    metadata_schema=metadata_schema,
    temp_dir="tmp/"
)

# Create crawler and process documents
crawler = Crawler(config)
crawler.crawl("path/to/documents/")
```

### Minimal Example

```python
from crawler import Crawler, CrawlerConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig

# Minimal configuration with defaults
config = CrawlerConfig.create(
    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    database=DatabaseClientConfig.milvus(collection="documents"),
    metadata_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"]
    }
)

crawler = Crawler(config)
crawler.crawl("document.pdf")
```

## Architecture

The crawler processes documents through a multi-stage pipeline:

```
┌──────────────┐
│   Document   │
│   (PDF/etc)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐     PyMuPDF4LLM + Vision LLM
│  Converter   │────────────────────────────────▶ Markdown + Images
└──────┬───────┘
       │
       ▼
┌──────────────┐     LLM + JSON Schema
│  Extractor   │────────────────────────────────▶ Structured Metadata
└──────┬───────┘
       │
       ▼
┌──────────────┐     Configurable Size/Overlap
│   Chunker    │────────────────────────────────▶ Text Chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐     Embedding Model
│   Embedder   │────────────────────────────────▶ Vector Embeddings
└──────┬───────┘
       │
       ▼
┌──────────────┐     Hybrid Search (Dense + BM25)
│   Milvus DB  │────────────────────────────────▶ Stored + Searchable
└──────────────┘
```

## Core Modules

### Converter

Transforms source documents (PDFs, etc.) into markdown format with image descriptions:

```python
from crawler.converter import PyMuPDF4LLMConfig, create_converter
from crawler.document import Document

config = PyMuPDF4LLMConfig(
    type="pymupdf4llm",
    vlm_config=LLMConfig.ollama(model_name="llava")
)
converter = create_converter(config)

doc = Document.create(source="paper.pdf")
converter.convert(doc)  # Populates doc.markdown, doc.stats, etc.
```

See [converter/overview.md](src/crawler/converter/overview.md) for details.

### Extractor

Extracts structured metadata using LLMs and JSON Schema:

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}
        },
        "required": ["title"]
    }
)

llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=config)

result = extractor.run(doc)  # Returns MetadataExtractionResult
```

See [extractor/overview.md](src/crawler/extractor/overview.md) for details.

### Chunker

Splits text into manageable chunks for embedding:

```python
from crawler.chunker import Chunker, ChunkingConfig

config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=200,
    preserve_paragraphs=True
)
chunker = Chunker(config)

doc.chunks = chunker.chunk_text(doc)  # Returns list of text chunks
```

See [chunker/overview.md](src/crawler/chunker/overview.md) for details.

### LLM & Embeddings

Provides unified interfaces for LLMs and embedding models:

```python
from crawler.llm import LLMConfig, EmbedderConfig, get_llm, get_embedder

# LLM for text generation
llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
response = llm.invoke("What is Python?")

# Embedder for vectors
embedder = get_embedder(EmbedderConfig.ollama(model="all-minilm:v2"))
embedding = embedder.embed("Hello, world!")
embeddings = embedder.embed_batch(["text1", "text2", "text3"])
```

See [llm/overview.md](src/crawler/llm/overview.md) for details.

### Vector Database

Stores document chunks and embeddings in Milvus with hybrid search:

```python
from crawler.vector_db import DatabaseClientConfig, get_db

config = DatabaseClientConfig.milvus(
    collection="documents",
    host="localhost",
    port=19530
)

db = get_db(config, dimension=384, metadata_schema={}, description="")
db.insert_data(database_documents)  # Insert documents
```

See [vector_db/overview.md](src/crawler/vector_db/overview.md) for details.

## Configuration

All configuration uses Pydantic models for type safety and validation.

### Using Factory Methods (Recommended)

```python
from crawler import CrawlerConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig

config = CrawlerConfig.create(
    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    database=DatabaseClientConfig.milvus(collection="docs"),
    # ... other config
)
```

### Using Dictionaries (Legacy Support)

```python
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434"
    },
    "llm": {
        "provider": "ollama",
        "model_name": "llama3.2:3b",
        "base_url": "http://localhost:11434"
    },
    # ... other config
}

config = CrawlerConfig.from_dict(config_dict)
```

See [config/overview.md](src/crawler/config/overview.md) for complete configuration guide.

## Examples

The `examples/` directory contains working configurations for different document types:

- **[arxiv.py](examples/arxiv.py)**: Process academic papers with comprehensive metadata
- **[irads.py](examples/irads.py)**: Process technical R&D documents
- **[xmidas.py](examples/xmidas.py)**: Process Q&A data and documentation with preprocessing

```bash
# Run an example
python examples/arxiv.py
```

See [examples/OVERVIEW.md](examples/OVERVIEW.md) for detailed example documentation.

## Advanced Features

### Benchmarking

Enable benchmarking to evaluate search quality:

```python
config = CrawlerConfig.create(
    # ... other config
    benchmark=True,
    generate_benchmark_questions=True,
    num_benchmark_questions=5
)

crawler = Crawler(config)
crawler.crawl("documents/")
crawler.benchmark()  # Generates metrics and visualizations
```

Results include:
- Placement distribution (where relevant results appear)
- Distance distribution (similarity scores)
- Search time distribution
- Top-K accuracy metrics

### Caching

The crawler automatically caches processed documents:

```python
config = CrawlerConfig.create(
    # ... other config
    temp_dir="tmp/",
    use_cache=True  # Default
)

# First run: processes and caches
crawler.crawl("documents/")

# Second run: loads from cache (much faster)
crawler.crawl("documents/")
```

### Custom Processing

Override individual components at runtime:

```python
crawler = Crawler(config)

# Use different LLM for specific run
crawler_with_larger_model = crawler.with_llm(
    LLMConfig.ollama(model_name="llama3.2:70b")
)

# Use different chunk size
crawler_with_larger_chunks = crawler.with_chunking(
    ChunkingConfig.create(chunk_size=2000)
)

# Chain modifications
custom_crawler = (crawler
    .with_llm(LLMConfig.ollama(model_name="llama3.2:70b"))
    .with_chunking(ChunkingConfig.create(chunk_size=2000))
)
```

### Security Groups

Control document access with security groups:

```python
config = CrawlerConfig.create(
    # ... other config
    security_groups=["engineering", "research"]
)

# Documents will be tagged with security groups for access control
```

## Testing

Run the test suite:

```bash
# From crawler directory
make test

# Or with pytest directly
pytest src/crawler/tests/
```

## Development

### Project Structure

```
crawler/
├── src/
│   └── crawler/
│       ├── chunker/          # Text chunking
│       ├── config/           # Configuration management
│       ├── converter/        # Document conversion
│       ├── document/         # Document data model
│       ├── extractor/        # Metadata extraction
│       ├── llm/             # LLM and embedding interfaces
│       ├── vector_db/       # Vector database integration
│       └── main.py          # Main Crawler class
├── examples/                # Usage examples
├── tests/                   # Test suite
└── README.md               # This file
```

### Building

```bash
# Build package
make build

# Run linting
make lint

# Run tests
make test
```

## Documentation

- [src/overview.md](src/overview.md) - Complete source code overview
- [Crawler.md](Crawler.md) - Detailed API reference and workflow
- Module documentation:
  - [chunker/overview.md](src/crawler/chunker/overview.md)
  - [config/overview.md](src/crawler/config/overview.md)
  - [converter/overview.md](src/crawler/converter/overview.md)
  - [document/overview.md](src/crawler/document/overview.md)
  - [extractor/overview.md](src/crawler/extractor/overview.md)
  - [llm/overview.md](src/crawler/llm/overview.md)
  - [vector_db/overview.md](src/crawler/vector_db/overview.md)
- [examples/OVERVIEW.md](examples/OVERVIEW.md) - Example usage patterns

## Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```
RuntimeError: Error calling Ollama model 'llama3.2:3b'
```

Solution: Ensure Ollama is running and the model is pulled:
```bash
ollama serve
ollama pull llama3.2:3b
```

**2. Milvus Connection Error**
```
Failed to connect to Milvus at localhost:19530
```

Solution: Start Milvus with Docker Compose:
```bash
cd database
docker-compose up -d
```

**3. Missing Model**
```
Model 'all-minilm:v2' not found
```

Solution: Pull the embedding model:
```bash
ollama pull all-minilm:v2
```

**4. Schema Validation Error**
```
ValidationError: schema.type must be 'object'
```

Solution: Ensure your metadata schema has `"type": "object"`:
```python
metadata_schema = {
    "type": "object",  # Required
    "properties": {...}
}
```

### Getting Help

1. Check the [documentation](src/overview.md)
2. Review [examples](examples/OVERVIEW.md)
3. Check module-specific documentation
4. Review error messages and stack traces

## Performance Tips

1. **Choose appropriate chunk sizes**: 500-1000 chars for precise search, 1000-2000 for balanced, 2000+ for context-heavy applications
2. **Use caching**: Enable `use_cache=True` to avoid reprocessing documents
3. **Batch processing**: Process multiple documents in one `crawl()` call
4. **Optimize LLM calls**: Use faster models for extraction when possible
5. **Database partitions**: Use Milvus partitions to organize large collections

## Requirements

- Python >= 3.10
- pymilvus >= 2.6.0
- ollama >= 0.5.3
- pydantic >= 2.0
- pymupdf >= 1.24.0
- pymupdf4llm >= 0.0.17
- tqdm >= 4.66.0
- httpx

See [pyproject.toml](pyproject.toml) for complete dependency list.

## License

See [LICENSE](../LICENSE) file for details.

## Contributing

Contributions are welcome! The system is designed to be extensible:

- Add new converters by implementing the `Converter` interface
- Add new LLM providers by implementing the `LLM` interface
- Add new database backends by implementing the `DatabaseClient` interface
- Improve chunking strategies by extending the `Chunker` class

## Version

Current version: 0.1.0

See `__init__.py` for version information.
