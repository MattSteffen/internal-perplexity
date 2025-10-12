# Examples Directory Overview

This directory contains example configurations and usage patterns for the Crawler package. All examples have been updated to use the new Pydantic-based configuration system for better type safety and validation.

## Purpose

The examples directory demonstrates how to configure and use the Crawler package for different document processing scenarios, including:

- Academic paper processing (ArXiv)
- Religious document processing (Church documents)
- Internal R&D document processing (IRADS)
- Specialized technical documentation (X-Midas)

## Files

### arxiv.py

**Purpose:** Demonstrates processing of academic papers from ArXiv with comprehensive metadata extraction.

**Key Features:**
- Multi-schema metadata extraction (basic metadata + summary points)
- PyMuPDF converter with vision LLM for image description
- Type-safe configuration using factory methods
- Benchmark functionality for search performance
- Search demonstration with example queries

**Configuration Highlights:**
- Embedding model: `all-minilm:v2` (Ollama)
- Main LLM: `gpt-oss:20b` (Ollama, tools mode)
- Vision LLM: `granite3.2-vision:latest` (Ollama)
- Database: Milvus collection `arxiv3`
- Chunk size: 1000
- Extractor: Multi-schema with custom schemas

**Schemas:**
1. **schema1**: Core document properties (title, author, date, keywords, unique_words, description)
2. **schema2**: Summary points (summary_item_1, summary_item_2, summary_item_3)

**Functions:**
- `create_arxiv_config()`: Type-safe configuration factory
- `search_louvain_clustering()`: Demonstrates search functionality
- `main()`: Processes documents and runs benchmarks

**Usage:**
```python
python examples/arxiv.py
```

---

### church.py

**Purpose:** Processes religious and church documents with specialized metadata extraction.

**Key Features:**
- Multi-schema metadata extraction optimized for religious content
- PyMuPDF converter with image descriptions for religious documents
- Both type-safe and dictionary-based configuration options
- Configurable for document recreation

**Configuration Highlights:**
- Embedding model: `all-minilm:v2` (Ollama)
- Main LLM: `gemma3` (Ollama, tools mode)
- Vision LLM: `gemma3:latest` (Ollama)
- Database: Milvus collection `church_documents`
- Chunk size: 1000
- Extractor: Multi-schema with custom schemas
- Temp directory: `/tmp/church`

**Schemas:**
1. **schema1**: Core document properties (title, author, date, keywords, unique_words)
2. **schema2**: Summary points for multi-topic documents

**Functions:**
- `create_church_config()`: Type-safe configuration factory
- `main()`: Processes church documents

**Configuration Methods:**
```python
# Method 1: Type-safe (recommended)
config = create_church_config()

# Method 2: Dictionary-based (backward compatibility)
config = CrawlerConfig.from_dict(church_config_dict)
```

**Usage:**
```python
python examples/church.py
```

---

### irads.py

**Purpose:** Processes internal research and development (IRADS) documents with technical metadata extraction.

**Key Features:**
- Multi-schema metadata extraction for technical content
- Custom document library context for specialized understanding
- PyMuPDF converter with technical image descriptions
- Support for both type-safe and dictionary configurations

**Configuration Highlights:**
- Embedding model: `nomic-embed-text` (Ollama)
- Main LLM: `gemma3` (Ollama, tools mode)
- Vision LLM: `gemma3:latest` (Ollama)
- Database: Milvus collection `irad_documents`
- Chunk size: 1000
- Extractor: Multi-schema with custom context
- Temp directory: `/tmp/irads`

**Document Library Context:**
Custom context describes the collection as internal research documents focused on signal processing, machine learning, and development initiatives.

**Schemas:**
1. **schema1**: Core document properties (title, author, date, keywords, unique_words)
2. **schema2**: Summary points for multi-faceted technical documents

**Functions:**
- `create_irad_config()`: Type-safe configuration factory
- `main()`: Processes IRADS documents

**Configuration Methods:**
```python
# Method 1: Type-safe (recommended)
config = create_irad_config()

# Method 2: Dictionary-based (backward compatibility)
config = CrawlerConfig.from_dict(irad_config_dict)
```

**Usage:**
```python
python examples/irads.py
```

---

### xmidas.py

**Purpose:** Processes X-Midas technical documentation and Q&A data from multiple sources with preprocessing.

**Key Features:**
- Multiple data source support (LearnXM, XM Docs, Q&A)
- Preprocessing pipeline for JSON data conversion
- Partition-based organization in Milvus
- Comprehensive error handling and progress reporting
- Batch processing for large datasets

**Configuration Highlights:**
- Embedding model: `nomic-embed-text` (Ollama)
- Main LLM: `mistral-small3.2` (Ollama, tools mode)
- Vision LLM: `mistral-small3.2:latest` (Ollama)
- Database: Milvus collection `xmidas` with partitions
- Chunk size: 10000
- Extractor: Basic with custom library context
- Remote Ollama server: `http://ollama.a1.autobahn.rinconres.com`
- Remote Milvus: `10.43.210.111:19530`

**Data Sources:**
1. **LearnXM** (`learnxm.json`): Learning documentation with subjects, descriptions, tags, URLs
2. **XM Docs** (`xm_docs.json`): Technical documentation with help files and examples
3. **Q&A** (`processed_xm_qa.json`): Community discussions and technical support

**Schemas:**
1. **learnxm_schema**: Subject, description, URL, tags
2. **xm_docs_schema**: Subject, description, xm_path, tags
3. **qa_schema**: Question, answer, context, users, time

**Functions:**
- `create_xmidas_config()`: Type-safe configuration factory with partition support
- `preprocess_learnxm_data()`: Converts LearnXM JSON to crawler format
- `preprocess_xm_docs_data()`: Converts XM Docs JSON to crawler format
- `preprocess_qa_data()`: Converts Q&A JSON to crawler format
- `crawl_data_source()`: Generic crawler for any data source
- `crawl_learnxm()`: Process LearnXM documentation
- `crawl_xm_docs()`: Process XM Docs documentation
- `crawl_qa()`: Process Q&A data
- `crawl_all()`: Process all data sources with comprehensive reporting
- `main()`: Entry point for X-Midas processing pipeline

**Preprocessing Flow:**
1. Load JSON data from file
2. Convert each entry to crawler-compatible JSON format
3. Save individual JSON files to temp directory
4. Crawl temp directory with appropriate schema

**Usage:**
```python
# Process all data sources
python examples/xmidas.py

# Or import and process specific sources
from examples.xmidas import crawl_learnxm, crawl_xm_docs, crawl_qa

crawl_learnxm()
crawl_xm_docs()
crawl_qa()
```

---

## Common Configuration Pattern

All examples follow a consistent type-safe configuration pattern using Pydantic models:

### 1. Import Configuration Classes

```python
from crawler import Crawler, CrawlerConfig
from crawler.processing import (
    ExtractorConfig,
    ConverterConfig,
    EmbedderConfig,
    LLMConfig,
)
from crawler.storage import DatabaseClientConfig
```

### 2. Create Component Configurations

```python
# Embeddings
embeddings = EmbedderConfig.ollama(
    model="all-minilm:v2",
    base_url="http://localhost:11434"
)

# Main LLM
llm = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434",
    structured_output="tools"
)

# Vision LLM
vision_llm = LLMConfig.ollama(
    model_name="llava:latest",
    base_url="http://localhost:11434"
)

# Database
database = DatabaseClientConfig.milvus(
    collection="my_collection",
    host="localhost",
    port=19530,
    username="root",
    password="Milvus",
    recreate=False,
)

# Extractor
extractor = ExtractorConfig.multi_schema(
    schemas=[schema1, schema2],
    llm=llm,
    document_library_context="Custom context"
)

# Converter
converter = ConverterConfig.pymupdf(
    vision_llm=vision_llm,
    metadata={...}
)
```

### 3. Create Complete Configuration

```python
config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,
    extractor=extractor,
    chunk_size=1000,
    metadata_schema=combined_schema,
    temp_dir="/tmp/crawler",
    benchmark=False,
    log_level="INFO",
)
```

### 4. Run Crawler

```python
mycrawler = Crawler(config)
mycrawler.crawl(file_paths)

if config.benchmark:
    mycrawler.benchmark()
```

---

## Backward Compatibility

All examples support dictionary-based configuration for backward compatibility:

```python
config_dict = {
    "embeddings": {...},
    "llm": {...},
    "vision_llm": {...},
    "database": {...},
    "converter": {...},
    "extractor": {...},
    "utils": {
        "chunk_size": 1000,
        "temp_dir": "/tmp/crawler"
    },
    "metadata_schema": {...}
}

config = CrawlerConfig.from_dict(config_dict)
```

---

## Pydantic Benefits in Examples

The examples demonstrate the benefits of Pydantic-based configuration:

1. **Type Safety**: Catch configuration errors at initialization time
2. **Validation**: Automatic validation of all configuration parameters
3. **Documentation**: Self-documenting configuration with field descriptions
4. **Serialization**: Easy conversion to/from dictionaries and JSON
5. **IDE Support**: Better autocomplete and type hints
6. **Factory Methods**: Convenient configuration creation with sensible defaults

---

## Schema Design Best Practices

The examples demonstrate effective schema design:

1. **Multi-Schema Approach**: Separate core metadata from domain-specific fields
2. **Required vs Optional**: Mark critical fields as required
3. **MaxLength Constraints**: Prevent database issues with sensible limits
4. **Descriptive Properties**: Clear descriptions for LLM understanding
5. **JSON Schema Compliance**: Use standard JSON Schema format

---

## Performance Considerations

The examples show various performance optimizations:

1. **Caching**: All examples use temp directories for caching processed documents
2. **Chunk Sizes**: Varied based on document type and content density
3. **Batch Processing**: X-Midas example demonstrates large-scale batch processing
4. **Partitions**: X-Midas uses partitions for efficient data organization
5. **Benchmarking**: ArXiv example shows how to enable benchmarking

---

## Testing Examples

Before running examples, ensure:

1. **Services Running**:
   - Ollama server with required models
   - Milvus vector database
   
2. **Data Available**:
   - Update file paths in examples
   - Ensure source documents exist
   
3. **Configuration**:
   - Verify model names match your Ollama installation
   - Check database connection details
   - Adjust chunk sizes for your use case

4. **Environment**:
   - Sufficient disk space for temp files
   - Network access to remote services (if applicable)
   - Appropriate permissions for file operations

---

## Extending Examples

To create a new example:

1. Copy an existing example as a template
2. Define custom JSON schemas for your metadata
3. Create a type-safe configuration function
4. Update document processing logic if needed
5. Add custom preprocessing if required
6. Test with sample documents
7. Document your example in this OVERVIEW.md

---

## Migration Guide

For existing dictionary-based configurations:

### Before (Dictionary-based):
```python
config = {
    "embeddings": {"provider": "ollama", "model": "..."},
    "llm": {"model_name": "...", "provider": "ollama"},
    # ... more config
}
crawler_config = CrawlerConfig.from_dict(config)
```

### After (Type-safe, recommended):
```python
embeddings = EmbedderConfig.ollama(model="...")
llm = LLMConfig.ollama(model_name="...")
# ... more components

config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    # ... more parameters
)
```

---

## Troubleshooting

Common issues and solutions:

1. **ValidationError**: Check that all required fields are provided and have correct types
2. **Model not found**: Ensure model names match your Ollama installation
3. **Connection refused**: Verify Ollama and Milvus services are running
4. **File not found**: Update file paths to match your local system
5. **Import errors**: Ensure crawler package is properly installed

---

## Additional Resources

- Main Crawler Documentation: `../README.md`
- Processing Module Overview: `../src/crawler/processing/OVERVIEW.md`
- Storage Module Overview: `../src/crawler/storage/OVERVIEW.md`
- Config Module Overview: `../src/crawler/config/OVERVIEW.md`

---

**Last Updated:** After Pydantic migration (2024)
**Maintained By:** Crawler Development Team

