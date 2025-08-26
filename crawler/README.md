# Crawler: Document Processing and Vector Database System

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A flexible and extensible document processing system designed to convert various file formats, extract structured metadata, generate embeddings, and store everything in a vector database for retrieval and analysis.

## Features

- **Multiple Document Converters**: Support for PDF, DOCX, and other formats using PyMuPDF, MarkItDown, or Docling
- **AI-Powered Metadata Extraction**: Extract structured information using LLMs and JSON schemas
- **Vision Language Model Integration**: Describe images and diagrams within documents
- **Vector Database Support**: Store and search through document chunks with embeddings
- **Modular Architecture**: Easily swap out components for different technologies
- **Comprehensive Benchmarking**: Test and evaluate search performance
- **Production Ready**: Robust error handling, logging, and configuration management

## Installation

### Requirements

- Python 3.13+
- Access to a vector database (Milvus recommended)
- LLM API access (Ollama, OpenAI, or VLLM)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd crawler

# Install with uv (recommended)
uv pip install -e .

# Or install with pip
pip install -e .
```

### Dependencies

The crawler package includes the following key dependencies:

- **docling**: Advanced PDF processing with VLM integration
- **pymupdf**: PDF manipulation and text extraction
- **markitdown**: General document conversion
- **langchain-ollama**: LLM and embedding integration
- **pymilvus**: Vector database client
- **jsonschema**: Metadata validation
- **numpy**: Numerical operations

## Quick Start

### Basic Usage

```python
from crawler import Crawler, CrawlerConfig

# Configure the crawler
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
    },
    "llm": {
        "model_name": "llama3.2",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "collection": "documents",
        "recreate": True,
    },
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["title"]
    }
}

# Create and run the crawler
config = CrawlerConfig.from_dict(config_dict)
crawler = Crawler(config)
crawler.crawl("/path/to/your/documents")
```

### Advanced Configuration

```python
# Full configuration with all options
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
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
        "collection": "my_documents",
        "partition": "research_papers",
        "recreate": False,
    },
    "llm": {
        "model_name": "llama3.2",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "ctx_length": 32000,
        "default_timeout": 300.0,
    },
    "converter": {
        "type": "pymupdf",
        "metadata": {
            "preserve_formatting": True,
            "include_page_numbers": True,
            "extract_tables": True,
            "image_description_prompt": "Describe this image for document indexing.",
        }
    },
    "extractor": {
        "type": "basic",
        "llm": {
            "model_name": "llama3.2",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        }
    },
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 500},
            "author": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string", "maxLength": 2000}
        },
        "required": ["title", "summary"]
    },
    "utils": {
        "chunk_size": 1000,
        "temp_dir": "tmp/",
        "benchmark": True,
        "generate_benchmark_questions": True,
        "num_benchmark_questions": 5,
    },
    "log_level": "INFO"
}

config = CrawlerConfig.from_dict(config_dict)
crawler = Crawler(config)
```

## Architecture

### Core Components

The crawler follows a modular architecture with clear separation of concerns:

1. **Document Conversion**: Transform various file formats into standardized Markdown
2. **Metadata Extraction**: Use LLMs to extract structured information from documents
3. **Text Chunking**: Split documents into manageable pieces for embedding
4. **Embedding Generation**: Convert text chunks into vector representations
5. **Vector Storage**: Store embeddings and metadata in a searchable database

### Processing Pipeline

```
Raw Document → Converter → Markdown → Extractor → Metadata + Chunks → Embedder → Vectors → Database
```

## Configuration

### Configuration Options

#### Embeddings Configuration

```python
"embeddings": {
    "provider": "ollama",  # "ollama", "openai", or "vllm"
    "model": "all-minilm:v2",  # Model name
    "base_url": "http://localhost:11434",  # API endpoint
    "api_key": "ollama",  # API key (if required)
    "dimension": 384  # Embedding dimension (optional)
}
```

#### Database Configuration

```python
"database": {
    "provider": "milvus",  # Currently only "milvus" supported
    "host": "localhost",
    "port": 19530,
    "username": "root",
    "password": "123456",
    "collection": "documents",  # Collection name
    "partition": "optional_partition",  # Optional partition
    "recreate": False,  # Recreate collection if exists
    "collection_description": "Document collection for search"
}
```

#### LLM Configuration

```python
"llm": {
    "model_name": "llama3.2",
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "system_prompt": "You are a helpful assistant...",  # Optional system prompt
    "ctx_length": 32000,  # Context window size
    "default_timeout": 300.0  # Request timeout in seconds
}
```

#### Converter Configuration

```python
"converter": {
    "type": "pymupdf",  # "pymupdf", "markitdown", or "docling"
    "metadata": {
        "preserve_formatting": True,
        "include_page_numbers": True,
        "extract_tables": True,
        "image_description_prompt": "Describe this image...",
        "image_describer": {
            "type": "ollama",
            "model": "llava",
            "base_url": "http://localhost:11434"
        }
    }
}
```

#### Extractor Configuration

```python
"extractor": {
    "type": "basic",  # "basic" or "multi_schema"
    "llm": {...},  # LLM config for extraction
    "metadata_schema": [...]  # JSON schemas for extraction
}
```

## API Reference

### Main Classes

#### `Crawler`

The main orchestrator class that manages the entire document processing pipeline.

**Methods:**
- `__init__(config, **kwargs)`: Initialize with configuration
- `crawl(path)`: Process files or directories
- `benchmark()`: Run performance benchmarks

**Parameters:**
- `config`: `CrawlerConfig` object
- `path`: File path, directory path, or list of paths

#### `CrawlerConfig`

Configuration dataclass containing all system settings.

**Key Attributes:**
- `embeddings`: Embedding model configuration
- `llm`: Language model configuration
- `database`: Database connection settings
- `converter`: Document converter settings
- `extractor`: Metadata extractor settings
- `metadata_schema`: JSON schema for metadata validation
- `chunk_size`: Text chunking size
- `temp_dir`: Temporary file directory

### Utility Functions

#### `create_converter(type, config)`

Factory function to create document converters.

**Parameters:**
- `type`: Converter type ("pymupdf", "markitdown", "docling")
- `config`: `ConverterConfig` object

**Returns:** Configured converter instance

#### `get_llm(config)`

Factory function to create LLM clients.

**Parameters:**
- `config`: `LLMConfig` object

**Returns:** Configured LLM instance

## Examples

### Processing a Single PDF

```python
from crawler import Crawler, CrawlerConfig

# Simple configuration for PDF processing
config = CrawlerConfig.from_dict({
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434",
    },
    "llm": {
        "model_name": "llama3.2",
        "provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "database": {
        "provider": "milvus",
        "collection": "pdf_collection",
    },
    "converter": {
        "type": "pymupdf",
    },
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"}
        }
    }
})

crawler = Crawler(config)
crawler.crawl("document.pdf")
```

### Processing Multiple Documents with Custom Schema

```python
from crawler import Crawler, CrawlerConfig

# Custom schema for research papers
research_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "abstract": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "publication_year": {"type": "integer"},
        "unique_terms": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title", "abstract"]
}

config = CrawlerConfig.from_dict({
    "embeddings": {"provider": "ollama", "model": "all-minilm:v2", "base_url": "http://localhost:11434"},
    "llm": {"model_name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
    "database": {"provider": "milvus", "collection": "research_papers"},
    "metadata_schema": research_schema
})

crawler = Crawler(config)
crawler.crawl("/path/to/research/papers/")
```

### Using Vision Language Models

```python
from crawler import Crawler, CrawlerConfig

# Configuration with VLM for image descriptions
config = CrawlerConfig.from_dict({
    "embeddings": {"provider": "ollama", "model": "all-minilm:v2", "base_url": "http://localhost:11434"},
    "vision_llm": {"model_name": "llava", "provider": "ollama", "base_url": "http://localhost:11434"},
    "llm": {"model_name": "llama3.2", "provider": "ollama", "base_url": "http://localhost:11434"},
    "database": {"provider": "milvus", "collection": "vlm_processed_docs"},
    "converter": {
        "type": "pymupdf",
        "metadata": {
            "extract_tables": True,
            "image_description_prompt": "Describe this diagram in detail for technical documentation."
        }
    },
    "metadata_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "diagrams": {"type": "array", "items": {"type": "string"}}
        }
    }
})

crawler = Crawler(config)
crawler.crawl("technical_document.pdf")
```

## Performance Benchmarking

The crawler includes built-in benchmarking capabilities to evaluate search performance:

- **Search Time Distribution**: Measure query response times
- **Placement Distribution**: Track where relevant results appear in search rankings
- **Distance Distribution**: Analyze embedding similarity scores
- **Top-K Accuracy**: Measure percentage of queries with relevant results in top K positions

### Running Benchmarks

```python
# Enable benchmarking in configuration
config = CrawlerConfig.from_dict({
    "database": {"provider": "milvus", "collection": "test_collection"},
    "embeddings": {"provider": "ollama", "model": "all-minilm:v2", "base_url": "http://localhost:11434"},
    "utils": {"benchmark": True, "generate_benchmark_questions": True}
})

crawler = Crawler(config)
crawler.crawl("path/to/documents")
results = crawler.benchmark()

# Results include performance metrics and visualizations
print(f"Top-1 Accuracy: {results.percent_in_top_k[1]:.2f}%")
```

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/crawler.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `python -m pytest`
5. Make your changes and add tests
6. Submit a pull request

### Code Style

- Use Black for code formatting
- Follow PEP 8 conventions
- Add type hints for new functions
- Write comprehensive docstrings
- Add unit tests for new features

### Adding New Components

#### Custom Converter

```python
from crawler.processing.converter import Converter, ConverterConfig

class MyCustomConverter(Converter):
    def __init__(self, config: ConverterConfig):
        super().__init__(config)
        # Your initialization code

    def convert(self, filepath: str) -> str:
        # Your conversion logic
        return markdown_content
```

#### Custom Extractor

```python
from crawler.processing.extractor import Extractor

class MyCustomExtractor(Extractor):
    def __init__(self, config: dict):
        super().__init__()
        # Your initialization

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        # Your extraction logic
        return metadata
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Ollama/Milvus services are running
2. **Out of Memory**: Reduce batch sizes or chunk sizes
3. **Schema Validation Errors**: Check that extracted metadata matches your schema
4. **Empty Results**: Verify that documents contain extractable text

### Logging

Enable detailed logging for debugging:

```python
config = CrawlerConfig.from_dict({
    "log_level": "DEBUG",
    "log_file": "crawler.log"
})
```

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review example configurations in the `examples/` directory
