# Type-Safe Configuration Guide

This guide explains how to use the new type-safe configuration system for the document crawler, which replaces the previous JSON/dictionary-based approach with direct Python object instantiation.

## Benefits of Type-Safe Configuration

1. **Full IDE Support**: Autocompletion, type checking, and inline documentation
2. **Compile-time Validation**: Catch configuration errors before runtime
3. **Better Error Messages**: Clear, specific validation errors
4. **No Typos**: Type checking prevents common dictionary key mistakes
5. **Refactoring Safety**: IDE can safely rename configuration fields

## Quick Start

Instead of this (old way):

```python
config = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434"
    },
    "llm": {
        "model_name": "llama3.2:3b",
        "provider": "ollama",
        "base_url": "http://localhost:11434"
    },
    # ... more nested dictionaries
}

crawler_config = CrawlerConfig.from_dict(config)
```

Do this (new way):

```python
from crawler import CrawlerConfig, EmbedderConfig, LLMConfig, DatabaseClientConfig

# Create configurations directly
embeddings = EmbedderConfig.ollama(
    model="all-minilm:v2",
    base_url="http://localhost:11434"
)

llm = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434"
)

database = DatabaseClientConfig.milvus(collection="documents")

# Create crawler config
crawler_config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=llm,  # Can reuse the same LLM for vision tasks
    database=database
)
```

## Configuration Classes

### EmbedderConfig

Creates embedding model configurations.

```python
# Ollama embeddings
embeddings = EmbedderConfig.ollama(
    model="all-minilm:v2",
    base_url="http://localhost:11434",
    dimension=384  # Optional: pre-configured dimension
)

# OpenAI embeddings
embeddings = EmbedderConfig.openai(
    model="text-embedding-3-small",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"  # Optional
)

# Direct instantiation
embeddings = EmbedderConfig(
    model="custom-model",
    base_url="http://custom:8080",
    provider="custom",
    api_key="key-if-needed"
)
```

### LLMConfig

Creates language model configurations.

```python
# Ollama LLM
llm = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434",
    system_prompt="You are a helpful assistant",  # Optional
    ctx_length=32000,  # Optional
    default_timeout=300.0  # Optional
)

# OpenAI LLM
llm = LLMConfig.openai(
    model_name="gpt-4",
    api_key="your-api-key"
)

# vLLM
llm = LLMConfig.vllm(
    model_name="mistral-7b",
    base_url="http://localhost:8000",
    api_key="optional-key"
)

# Direct instantiation
llm = LLMConfig(
    model_name="custom-model",
    base_url="http://custom:8080",
    provider="custom",
    api_key="key-if-needed"
)
```

### ConverterConfig

Creates document converter configurations.

```python
# MarkItDown converter (requires vision LLM)
vision_llm = LLMConfig.ollama(model_name="llava:latest")
converter = ConverterConfig.markitdown(vision_llm=vision_llm)

# Docling converter (requires vision LLM)
converter = ConverterConfig.docling(vision_llm=vision_llm)

# Docling VLM converter (uses default VLM)
converter = ConverterConfig.docling_vlm()

# PyMuPDF converter (vision LLM optional)
converter = ConverterConfig.pymupdf(
    vision_llm=vision_llm,  # Optional
    metadata={
        "extract_tables": True,
        "include_metadata": True,
        "image_description_prompt": "Describe this image..."
    }
)

# Direct instantiation
converter = ConverterConfig(
    type="markitdown",
    vision_llm=vision_llm,
    metadata={"custom": "options"}
)
```

### ExtractorConfig

Creates metadata extractor configurations.

```python
# Basic extractor
llm = LLMConfig.ollama(model_name="llama3.2:3b")
extractor = ExtractorConfig.basic(
    llm=llm,
    metadata_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 512},
            "author": {"type": "string", "maxLength": 256}
        }
    }
)

# Multi-schema extractor
schemas = [
    {"type": "object", "properties": {"title": {"type": "string"}}},
    {"type": "object", "properties": {"year": {"type": "integer"}}}
]
extractor = ExtractorConfig.multi_schema(schemas=schemas, llm=llm)

# Direct instantiation
extractor = ExtractorConfig(
    type="basic",
    llm=llm,
    metadata_schema=schema_dict
)
```

### DatabaseClientConfig

Creates database client configurations.

```python
# Milvus database
database = DatabaseClientConfig.milvus(
    collection="documents",
    host="localhost",
    port=19530,
    username="root",
    password="Milvus",
    partition="optional_partition",  # Optional
    recreate=False,  # Optional
    collection_description="My document collection"  # Optional
)

# Direct instantiation
database = DatabaseClientConfig(
    provider="milvus",
    collection="documents",
    host="localhost",
    port=19530
)
```

### CrawlerConfig

Creates the main crawler configuration.

```python
# Using create() method
config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,  # Optional
    extractor=extractor,  # Optional
    chunk_size=10000,  # Optional
    temp_dir="tmp/",  # Optional
    benchmark=False,  # Optional
    log_level="INFO",  # Optional
    log_file=None  # Optional
)

# Using default_ollama() helper
config = CrawlerConfig.default_ollama(
    collection="my_docs",
    embed_model="all-minilm:v2",
    llm_model="llama3.2:3b",
    vision_model="llava:latest",
    base_url="http://localhost:11434",
    host="localhost",
    port=19530
)

# Direct instantiation
config = CrawlerConfig(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database
)
```

## Complete Examples

### Basic Ollama Setup

```python
from crawler import (
    CrawlerConfig, EmbedderConfig, LLMConfig,
    DatabaseClientConfig, ConverterConfig, ExtractorConfig
)

# Create all configurations
embeddings = EmbedderConfig.ollama(model="all-minilm:v2")
llm = LLMConfig.ollama(model_name="llama3.2:3b")
vision_llm = LLMConfig.ollama(model_name="llava:latest")
database = DatabaseClientConfig.milvus(collection="documents")
converter = ConverterConfig.markitdown(vision_llm=vision_llm)
extractor = ExtractorConfig.basic(llm=llm)

# Create crawler
config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,
    extractor=extractor
)

crawler = Crawler(config)
```

### Advanced OpenAI Setup

```python
# OpenAI configuration
embeddings = EmbedderConfig.openai(
    model="text-embedding-3-small",
    api_key="your-openai-key"
)

llm = LLMConfig.openai(
    model_name="gpt-4",
    api_key="your-openai-key"
)

vision_llm = LLMConfig.openai(
    model_name="gpt-4-vision-preview",
    api_key="your-openai-key"
)

database = DatabaseClientConfig.milvus(
    collection="research_papers",
    host="your-milvus-host",
    port=19530,
    username="user",
    password="pass"
)

# Advanced converter with custom options
converter = ConverterConfig.pymupdf(
    vision_llm=vision_llm,
    metadata={
        "extract_tables": True,
        "preserve_formatting": True,
        "image_description_prompt": "Analyze this image for research context"
    }
)

# Multi-schema extractor
schemas = [
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "abstract": {"type": "string"}
        }
    }
]

extractor = ExtractorConfig.multi_schema(schemas=schemas, llm=llm)

config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database,
    converter=converter,
    extractor=extractor,
    chunk_size=8000,  # Smaller chunks for research papers
    log_level="DEBUG"
)
```

## Migration Guide

### From JSON/Dictionary Configuration

**Old way:**

```python
config_dict = {
    "embeddings": {
        "provider": "ollama",
        "model": "all-minilm:v2",
        "base_url": "http://localhost:11434"
    },
    "llm": {
        "model_name": "llama3.2:3b",
        "provider": "ollama",
        "base_url": "http://localhost:11434"
    },
    "vision_llm": {
        "model_name": "llava:latest",
        "provider": "ollama",
        "base_url": "http://localhost:11434"
    },
    "database": {
        "provider": "milvus",
        "collection": "documents",
        "host": "localhost",
        "port": 19530
    }
}

crawler_config = CrawlerConfig.from_dict(config_dict)
```

**New way:**

```python
embeddings = EmbedderConfig.ollama(model="all-minilm:v2")
llm = LLMConfig.ollama(model_name="llama3.2:3b")
vision_llm = LLMConfig.ollama(model_name="llava:latest")
database = DatabaseClientConfig.milvus(collection="documents")

crawler_config = CrawlerConfig.create(
    embeddings=embeddings,
    llm=llm,
    vision_llm=vision_llm,
    database=database
)
```

### Benefits of Migration

1. **Type Safety**: IDE will catch typos and type mismatches
2. **Validation**: Configuration errors caught at creation time
3. **Documentation**: Factory methods provide clear usage examples
4. **Maintainability**: Easier to refactor and modify configurations
5. **Readability**: More readable and self-documenting code

## Error Handling

The new configuration system provides clear, specific error messages:

```python
# This will raise ValueError: "Embedder model cannot be empty"
embeddings = EmbedderConfig(model="")

# This will raise ValueError: "LLM model_name cannot be empty"
llm = LLMConfig(model_name="")

# This will raise ValueError: "Database collection cannot be empty"
database = DatabaseClientConfig.milvus(collection="")

# This will raise ValueError: "Port must be between 1 and 65535"
database = DatabaseClientConfig.milvus(collection="test", port=99999)
```

## Best Practices

1. **Use Factory Methods**: Prefer `EmbedderConfig.ollama()` over direct instantiation
2. **Validate Early**: Create configurations at startup to catch errors early
3. **Use Type Hints**: Let your IDE help you with autocompletion
4. **Document Custom Configs**: Add docstrings for custom configuration classes
5. **Test Configurations**: Create unit tests for your configuration setups

## Advanced Usage

### Custom Configuration Classes

```python
from dataclasses import dataclass
from typing import Optional
from crawler.processing import LLMConfig

@dataclass
class CustomEmbedderConfig:
    model: str
    custom_param: Optional[str] = None

    def __post_init__(self):
        if not self.model:
            raise ValueError("Model cannot be empty")

    @classmethod
    def my_provider(cls, model: str, custom_param: str = None):
        return cls(model=model, custom_param=custom_param)
```

### Environment Variables

```python
import os

# Use environment variables for sensitive data
embeddings = EmbedderConfig.openai(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

database = DatabaseClientConfig.milvus(
    collection="documents",
    password=os.getenv("MILVUS_PASSWORD")
)
```

This new type-safe configuration system makes your code more robust, maintainable, and developer-friendly!
