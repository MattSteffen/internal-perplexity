# LLM Module Overview

This module provides unified interfaces for interacting with Large Language Models (LLMs) and embedding models across different providers. It supports structured output generation via JSON schemas and function calling, enabling reliable metadata extraction and text embedding.

## Files in This Module

### `__init__.py`

Exports the public API for the LLM module:
- `LLM`, `LLMConfig`, `OllamaLLM`, `VllmLLM`, `get_llm` - LLM interfaces and implementations
- `schema_to_openai_tools` - Utility for converting JSON schemas to OpenAI tools format

### `llm.py`

Contains LLM configuration and implementations for text generation.

**Key Components:**
- **`LLMConfig`**: Pydantic model for LLM configuration with factory methods
- **`LLM`**: Abstract base class defining the LLM interface
- **`OllamaLLM`**: Implementation for Ollama-hosted models
- **`VllmLLM`**: Implementation for vLLM-hosted models using OpenAI-compatible API
- **`get_llm()`**: Factory function that returns appropriate LLM implementation
- **`schema_to_openai_tools()`**: Converts JSON schema to OpenAI tools format

### `embeddings.py`

Contains embedding model configuration and implementations.

**Key Components:**
- **`EmbedderConfig`**: Pydantic model for embedder configuration with factory methods
- **`Embedder`**: Abstract base class defining the embedder interface
- **`OllamaEmbedder`**: Implementation for Ollama embedding models
- **`get_embedder()`**: Factory function that returns appropriate embedder implementation

## Core Concepts

### LLM Configuration

The `LLMConfig` class provides type-safe configuration for LLM providers with automatic validation:

**Fields:**
- `model_name` (str, required): Name of the LLM model
- `base_url` (str): API endpoint URL (default: "http://localhost:11434")
- `system_prompt` (str | None): Optional system prompt to set model behavior
- `ctx_length` (int): Context window size (default: 32000)
- `default_timeout` (float): API timeout in seconds (default: 300.0)
- `provider` (str): Provider name ("ollama", "openai", "vllm")
- `api_key` (str): API key for authentication (if required)
- `structured_output` (str): Mode for structured output ("response_format" or "tools")

**Factory Methods:**
- `LLMConfig.ollama(model_name, base_url, ...)` - Create Ollama configuration
- `LLMConfig.openai(model_name, api_key, ...)` - Create OpenAI configuration
- `LLMConfig.vllm(model_name, base_url, ...)` - Create vLLM configuration

### Embedder Configuration

The `EmbedderConfig` class provides type-safe configuration for embedding models:

**Fields:**
- `model` (str, required): Name of the embedding model
- `base_url` (str, required): API endpoint URL
- `api_key` (str): API key for authentication
- `provider` (str): Provider name (default: "ollama")
- `dimension` (int | None): Optional pre-configured embedding dimension

**Factory Methods:**
- `EmbedderConfig.ollama(model, base_url, dimension)` - Create Ollama configuration
- `EmbedderConfig.openai(model, api_key, base_url, dimension)` - Create OpenAI configuration

### Structured Output Modes

The LLM implementations support two modes for generating structured JSON output:

**1. Response Format Mode (`response_format`)**
- Uses JSON schema directly in the API call
- More reliable for models that support native JSON mode
- Example: Ollama's `format` parameter

**2. Tools Mode (`tools`)**
- Uses OpenAI-style function calling
- Better for models optimized for tool use
- Automatically extracts function arguments as JSON

The mode is configured via `LLMConfig.structured_output` field.

## Usage Examples

### Basic LLM Usage

```python
from crawler.llm import LLMConfig, get_llm

# Create LLM configuration
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434",
    ctx_length=32000,
    default_timeout=300.0
)

# Get LLM instance
llm = get_llm(llm_config)

# Simple text generation
response = llm.invoke("What is the capital of France?")
print(response)  # "Paris"
```

### Structured Output with Response Format

```python
from crawler.llm import LLMConfig, get_llm

# Configure LLM with response_format mode
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434",
    structured_output="response_format"
)
llm = get_llm(llm_config)

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "year": {"type": "integer"}
    },
    "required": ["title", "author"]
}

# Get structured output
prompt = "Extract metadata from this document: 'The Great Gatsby by F. Scott Fitzgerald, published in 1925.'"
result = llm.invoke(prompt, response_format=schema)
print(result)  # {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925}
```

### Structured Output with Tools Mode

```python
from crawler.llm import LLMConfig, get_llm, schema_to_openai_tools

# Configure LLM with tools mode
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    structured_output="tools"
)
llm = get_llm(llm_config)

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"}
    },
    "required": ["sentiment"]
}

# Convert schema to tools format
tools = schema_to_openai_tools(schema)

# Get structured output
prompt = "Analyze the sentiment of: 'This product is amazing!'"
result = llm.invoke(prompt, tools=tools)
print(result)  # {"sentiment": "positive", "confidence": 0.95}
```

### Message History

```python
from crawler.llm import LLMConfig, get_llm

llm_config = LLMConfig.ollama(model_name="llama3.2:3b")
llm = get_llm(llm_config)

# Use message history for conversations
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What is that number squared?"}
]

response = llm.invoke(messages)
print(response)  # "16"
```

### System Prompts

```python
from crawler.llm import LLMConfig, get_llm

# Configure with system prompt
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    system_prompt="You are a helpful assistant that always responds in JSON format."
)
llm = get_llm(llm_config)

# System prompt is automatically included in all calls
response = llm.invoke("What is Python?")
```

### Basic Embedder Usage

```python
from crawler.llm import EmbedderConfig, get_embedder

# Create embedder configuration
embed_config = EmbedderConfig.ollama(
    model="all-minilm:v2",
    base_url="http://localhost:11434"
)

# Get embedder instance
embedder = get_embedder(embed_config)

# Embed single text
embedding = embedder.embed("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")  # 384

# Get dimension
dim = embedder.get_dimension()
print(f"Model dimension: {dim}")  # 384
```

### Batch Embeddings

```python
from crawler.llm import EmbedderConfig, get_embedder

embed_config = EmbedderConfig.ollama(model="all-minilm:v2")
embedder = get_embedder(embed_config)

# Embed multiple texts at once
texts = [
    "First document text",
    "Second document text",
    "Third document text"
]

embeddings = embedder.embed_batch(texts)
print(f"Created {len(embeddings)} embeddings")  # 3
print(f"Each embedding has {len(embeddings[0])} dimensions")  # 384
```

### Using Different Providers

```python
from crawler.llm import LLMConfig, EmbedderConfig, get_llm, get_embedder

# Ollama LLM
ollama_llm = get_llm(LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434"
))

# vLLM (OpenAI-compatible)
vllm_llm = get_llm(LLMConfig.vllm(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://vllm-server:8000"
))

# OpenAI
openai_llm = get_llm(LLMConfig.openai(
    model_name="gpt-4",
    api_key="sk-..."
))

# Ollama Embedder
ollama_embedder = get_embedder(EmbedderConfig.ollama(
    model="all-minilm:v2"
))

# OpenAI Embedder
openai_embedder = get_embedder(EmbedderConfig.openai(
    model="text-embedding-3-small",
    api_key="sk-..."
))
```

## Integration with Other Modules

### With Metadata Extractor

The LLM module is used by the extractor for metadata extraction:

```python
from crawler.llm import LLMConfig, get_llm
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig

# Configure LLM
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    structured_output="response_format"
)
llm = get_llm(llm_config)

# Configure extractor
extractor_config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}
        },
        "required": ["title"]
    },
    context="Research papers"
)

# Create extractor with LLM
extractor = MetadataExtractor(llm=llm, config=extractor_config)

# Extract metadata from markdown
metadata = extractor.extract("# My Paper\nBy John Doe\n\nAbstract...")
```

### With Vector Database

The embedder is used to generate embeddings for vector database storage:

```python
from crawler.llm import EmbedderConfig, get_embedder
from crawler.vector_db import DatabaseClientConfig, get_db

# Configure embedder
embed_config = EmbedderConfig.ollama(model="all-minilm:v2")
embedder = get_embedder(embed_config)

# Get embedding dimension for database schema
dimension = embedder.get_dimension()

# Configure database with correct dimension
db_config = DatabaseClientConfig.milvus(
    collection="documents",
    host="localhost",
    port=19530
)

db = get_db(db_config, dimension, {}, "Document collection")

# Generate embeddings for text chunks
chunks = ["chunk 1", "chunk 2", "chunk 3"]
embeddings = embedder.embed_batch(chunks)

# Store in database (simplified)
# db.insert_data([...])
```

### With Crawler

Both LLM and embedder are used in the main crawler:

```python
from crawler import Crawler, CrawlerConfig
from crawler.llm import LLMConfig, EmbedderConfig

config = CrawlerConfig.create(
    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    # ... other config
)

crawler = Crawler(config)
crawler.crawl("documents/")
```

## Error Handling

### Timeout Handling

```python
from crawler.llm import LLMConfig, get_llm

llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    default_timeout=30.0  # 30 second timeout
)
llm = get_llm(llm_config)

try:
    response = llm.invoke("Very long prompt...")
except TimeoutError as e:
    print(f"Request timed out: {e}")
```

### Connection Errors

```python
from crawler.llm import LLMConfig, get_llm

llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    base_url="http://invalid-host:11434"
)
llm = get_llm(llm_config)

try:
    response = llm.invoke("Hello")
except RuntimeError as e:
    print(f"Connection error: {e}")
```

### JSON Parsing Errors

```python
from crawler.llm import LLMConfig, get_llm

llm_config = LLMConfig.ollama(model_name="llama3.2:3b")
llm = get_llm(llm_config)

schema = {"type": "object", "properties": {"name": {"type": "string"}}}

try:
    result = llm.invoke("Say hello", response_format=schema)
except ValueError as e:
    print(f"Failed to parse JSON: {e}")
```

## Best Practices

### 1. Choose the Right Structured Output Mode

- Use `response_format` for models with native JSON support (default)
- Use `tools` for models optimized for function calling
- Test both modes to see which works better for your model

### 2. Set Appropriate Timeouts

- Default timeout is 300 seconds (5 minutes)
- Adjust based on model size and prompt complexity
- Use shorter timeouts for quick operations

### 3. Pre-configure Embedding Dimensions

```python
# Pre-configure dimension to avoid probing
embed_config = EmbedderConfig.ollama(
    model="all-minilm:v2",
    dimension=384  # Known dimension for this model
)
```

### 4. Reuse LLM/Embedder Instances

```python
# Don't create new instances for each call
llm = get_llm(config)  # Create once

# Reuse for multiple calls
for document in documents:
    result = llm.invoke(...)
```

### 5. Use System Prompts for Consistent Behavior

```python
# Set behavior once in config
llm_config = LLMConfig.ollama(
    model_name="llama3.2:3b",
    system_prompt="You are a metadata extraction expert. Always output valid JSON."
)
```

## Supported Providers

### Ollama

- **LLM**: Native Ollama Python client
- **Embedder**: Native Ollama embeddings API
- **Structured Output**: Both `response_format` and `tools` modes
- **Requirements**: Ollama server running locally or remotely

### vLLM

- **LLM**: OpenAI-compatible `/v1/chat/completions` endpoint
- **Embedder**: Not currently supported
- **Structured Output**: Both `response_format` and `tools` modes
- **Requirements**: vLLM server with OpenAI-compatible API

### OpenAI

- **LLM**: Via vLLM implementation (OpenAI-compatible API)
- **Embedder**: Via configuration (not yet implemented in get_embedder)
- **Structured Output**: Both modes
- **Requirements**: Valid OpenAI API key

## Advanced Usage

### Custom Structured Output Tools

```python
from crawler.llm import schema_to_openai_tools

# Define custom schema with description
schema = {
    "type": "object",
    "description": "Extract key information from academic papers",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "abstract": {"type": "string"}
    },
    "required": ["title"]
}

# Convert to tools format
tools = schema_to_openai_tools(schema)
print(tools)
# [{"type": "function", "function": {"name": "extract_metadata", ...}}]
```

### Dynamic Context Length

```python
from crawler.llm import LLMConfig, get_llm

# Adjust context length based on prompt size
def get_llm_with_context(prompt_length: int):
    ctx_length = min(32000, prompt_length + 4096)  # Buffer for output
    config = LLMConfig.ollama(
        model_name="llama3.2:3b",
        ctx_length=ctx_length
    )
    return get_llm(config)
```

## Dependencies

- `ollama>=0.5.3` - Ollama Python client
- `httpx` - HTTP client for vLLM API calls
- `pydantic>=2.0` - Configuration validation

## Related Documentation

- [Extractor Module](../extractor/overview.md) - Uses LLMs for metadata extraction
- [Crawler Configuration](../config/overview.md) - LLM/Embedder configuration in crawler
- [Examples](../../../examples/OVERVIEW.md) - Real-world usage examples

