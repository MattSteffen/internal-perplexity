# Extractor Module Overview

This module provides functionality for extracting structured metadata from documents using Large Language Models (LLMs) with JSON Schema validation. It supports automatic schema enforcement, benchmark question generation, and flexible structured output modes.

## Files in This Module

### `__init__.py`

Exports the public API for the extractor module:
- `MetadataExtractor` - Main extractor class
- `MetadataExtractorConfig` - Pydantic configuration model

### `extractor.py`

Contains the metadata extraction implementation with the following components:

**Key Classes:**
- **`MetadataExtractorConfig`**: Pydantic model for configuring metadata extraction
- **`MetadataExtractionResult`**: Pydantic model for extraction results
- **`MetadataExtractor`**: Main class for extracting structured metadata from documents

**Key Functions:**
- `extract(markdown: str) -> dict[str, Any]` - Extract metadata from markdown text
- `run(document: Document) -> MetadataExtractionResult` - Extract metadata and optional benchmark questions
- `generate_benchmark_questions(markdown: str, n: int) -> list[str]` - Generate benchmark questions

## Core Concepts

### Metadata Extraction

The extractor uses LLMs to extract structured information from documents according to a JSON Schema. It ensures type safety, validates output, and handles common LLM output issues automatically.

### Configuration

The `MetadataExtractorConfig` class provides comprehensive configuration options:

**Fields:**
- `json_schema` (dict, required): JSON Schema defining the metadata structure (must be type="object")
- `context` (str): Optional context for disambiguation (e.g., "Research papers on machine learning")
- `structured_output` (Literal["json_schema", "tools"]): How to request structured output from LLM (default: "json_schema")
- `include_benchmark_questions` (bool): Whether to generate benchmark questions (default: False)
- `num_benchmark_questions` (int): Number of questions to generate (default: 3, range: 1-20)
- `truncate_document_chars` (int): Maximum document text length for prompts (default: 4000, range: 512-32000)
- `strict` (bool): Whether to enforce schema strictly (default: True)

### Strict Mode

When `strict=True` (default), the extractor:
1. Drops extra keys not defined in the schema
2. Fills missing required fields with "Unknown"
3. Coerces simple types (arrays, strings) to match schema
4. Validates output with jsonschema if available

When `strict=False`, the extractor passes through LLM output with minimal processing.

### Structured Output Modes

The extractor supports two modes for getting structured output from LLMs:

**1. JSON Schema Mode (`json_schema`)**
- Sends schema via `response_format` parameter
- More reliable for models with native JSON support
- Default and recommended mode

**2. Tools Mode (`tools`)**
- Sends schema as OpenAI-style function/tool definition
- Better for models optimized for function calling
- Automatically extracts function arguments

## Usage Examples

### Basic Metadata Extraction

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import LLMConfig, get_llm

# Define JSON schema for metadata
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 500},
        "author": {"type": "array", "items": {"type": "string"}},
        "publication_date": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title"]
}

# Configure extractor
extractor_config = MetadataExtractorConfig(
    json_schema=schema,
    context="Academic research papers"
)

# Create LLM and extractor
llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=extractor_config)

# Extract metadata from markdown
markdown = """
# Deep Learning for Computer Vision

By Jane Smith and John Doe
Published: 2024

This paper explores...
"""

metadata = extractor.extract(markdown)
print(metadata)
# {
#     "title": "Deep Learning for Computer Vision",
#     "author": ["Jane Smith", "John Doe"],
#     "publication_date": "2024",
#     "keywords": ["deep learning", "computer vision"]
# }
```

### Extraction with Document Pipeline

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig
from crawler.document import Document

# Configure extractor
config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string", "maxLength": 1000}
        },
        "required": ["title"]
    }
)

# Create extractor
llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=config)

# Create document (assume it's been converted to markdown)
doc = Document.create(source="document.pdf")
doc.markdown = "# My Document\n\nContent goes here..."

# Extract metadata using run() method
result = extractor.run(doc)

# Access extracted metadata
print(result.metadata)
# {"title": "My Document", "summary": "Content goes here..."}

# Update document
doc.metadata = result.metadata
```

### Generating Benchmark Questions

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

# Configure extractor with benchmark questions enabled
config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "topic": {"type": "string"}
        },
        "required": ["title"]
    },
    include_benchmark_questions=True,
    num_benchmark_questions=5
)

llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=config)

# Extract with benchmark questions
markdown = """
# Machine Learning Basics

This document covers fundamental concepts in machine learning,
including supervised learning, unsupervised learning, and reinforcement learning.
"""

result = extractor.run(Document.create(source="doc.md"))
result.metadata  # {"title": "Machine Learning Basics", "topic": "machine learning"}
result.benchmark_questions  # ["What are the main types of machine learning?", ...]
```

### Using Different Structured Output Modes

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import LLMConfig, get_llm

# JSON Schema mode (default)
config_json = MetadataExtractorConfig(
    json_schema={...},
    structured_output="json_schema"
)

# Tools mode (for function-calling optimized models)
config_tools = MetadataExtractorConfig(
    json_schema={...},
    structured_output="tools"
)

# Create extractors with different modes
llm_config = LLMConfig.ollama(model_name="llama3.2:3b")
llm = get_llm(llm_config)

extractor_json = MetadataExtractor(llm=llm, config=config_json)
extractor_tools = MetadataExtractor(llm=llm, config=config_tools)
```

### Non-Strict Mode

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

# Allow extra fields and don't backfill missing fields
config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"}
        },
        "required": ["title"]
    },
    strict=False  # Disable strict mode
)

llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=config)

# LLM might return extra fields or omit optional ones
metadata = extractor.extract("# My Document")
# Passes through whatever LLM returns (minimal processing)
```

### Context for Disambiguation

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

# Provide context to help LLM understand document domain
config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "language": {"type": "string"},
            "framework": {"type": "string"}
        },
        "required": ["title"]
    },
    context="Technical documentation for software libraries and frameworks. "
            "Focus on programming languages and tools."
)

llm = get_llm(LLMConfig.ollama(model_name="llama3.2:3b"))
extractor = MetadataExtractor(llm=llm, config=config)

# Context helps LLM correctly identify technical terms
metadata = extractor.extract("# React Documentation\n\nReact is a JavaScript library...")
# {"title": "React Documentation", "language": "JavaScript", "framework": "React"}
```

## JSON Schema Design Best Practices

### 1. Use Clear Property Names

```python
# Good: Clear, unambiguous names
schema = {
    "type": "object",
    "properties": {
        "document_title": {"type": "string"},
        "primary_author": {"type": "string"},
        "publication_year": {"type": "integer"}
    }
}

# Less clear: Ambiguous or abbreviated names
schema = {
    "type": "object",
    "properties": {
        "t": {"type": "string"},  # What does 't' mean?
        "auth": {"type": "string"},  # Author? Authorization?
        "yr": {"type": "integer"}  # Year? Year-round?
    }
}
```

### 2. Add Descriptions for Complex Fields

```python
schema = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of relevant terms or phrases that categorize "
                          "the document's subject matter"
        },
        "unique_words": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Domain-specific or technical terms that might not be "
                          "common knowledge"
        }
    }
}
```

### 3. Set Appropriate Constraints

```python
schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 500  # Prevent extremely long titles
        },
        "publication_year": {
            "type": "integer",
            "minimum": 1900,  # Reasonable bounds
            "maximum": 2100
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20  # Limit array size
        }
    }
}
```

### 4. Mark Required vs Optional Fields

```python
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},  # Essential - mark as required
        "author": {"type": "string"},  # Essential - mark as required
        "isbn": {"type": "string"},  # Not always present - keep optional
        "edition": {"type": "integer"}  # Not always present - keep optional
    },
    "required": ["title", "author"]  # Only truly required fields
}
```

### 5. Use Arrays for Multiple Values

```python
# Good: Array for multiple values
schema = {
    "type": "object",
    "properties": {
        "authors": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# Less flexible: Single value when multiple are possible
schema = {
    "type": "object",
    "properties": {
        "author": {"type": "string"}  # What if there are multiple authors?
    }
}
```

### 6. Consider Multi-Schema Extraction

For complex documents with multiple aspects:

```python
# Schema 1: Core metadata
core_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "array", "items": {"type": "string"}},
        "date": {"type": "string"}
    },
    "required": ["title"]
}

# Schema 2: Content analysis
content_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 2000},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "methodology": {"type": "string"}
    }
}

# Combine for final schema
combined_schema = {
    "type": "object",
    "required": core_schema["required"],
    "properties": {
        **core_schema["properties"],
        **content_schema["properties"]
    }
}
```

## Integration with Other Modules

### With Document Pipeline

```python
from crawler.document import Document
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

# Document flows through converter first
doc = Document.create(source="paper.pdf")
# ... converter populates doc.markdown ...

# Then extractor processes the markdown
config = MetadataExtractorConfig(json_schema={...})
extractor = MetadataExtractor(llm=get_llm(LLMConfig.ollama(model_name="llama3.2:3b")), config=config)

result = extractor.run(doc)
doc.metadata = result.metadata
doc.benchmark_questions = result.benchmark_questions
```

### With Crawler

```python
from crawler import Crawler, CrawlerConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig

# Configure extractor as part of crawler config
config = CrawlerConfig.create(
    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    extractor=MetadataExtractorConfig(
        json_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title"]
        },
        context="Technical documentation"
    ),
    database=DatabaseClientConfig.milvus(collection="docs"),
    # ... other config
)

crawler = Crawler(config)
crawler.crawl("documents/")
```

### With Benchmarking

```python
from crawler.extractor import MetadataExtractorConfig

# Enable benchmark question generation
config = MetadataExtractorConfig(
    json_schema={...},
    include_benchmark_questions=True,
    num_benchmark_questions=5
)

# Questions are stored in document and can be used for search benchmarking
# The benchmark system uses these questions to test search accuracy
```

## Error Handling

### Schema Validation Errors

```python
from crawler.extractor import MetadataExtractorConfig
from pydantic import ValidationError

try:
    # Invalid schema (missing type)
    config = MetadataExtractorConfig(
        json_schema={
            "properties": {"title": {"type": "string"}}
            # Missing "type": "object"
        }
    )
except ValidationError as e:
    print(f"Schema validation error: {e}")
```

### LLM Response Parsing

```python
from crawler.extractor import MetadataExtractor, MetadataExtractorConfig
from crawler.llm import get_llm, LLMConfig

config = MetadataExtractorConfig(json_schema={...})
extractor = MetadataExtractor(llm=get_llm(LLMConfig.ollama(model_name="llama3.2:3b")), config=config)

try:
    metadata = extractor.extract("Document text...")
except ValueError as e:
    print(f"Failed to parse LLM response: {e}")
```

### Missing Required Fields (Strict Mode)

```python
# With strict=True (default), missing required fields are filled with "Unknown"
config = MetadataExtractorConfig(
    json_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"]
    },
    strict=True
)

# If LLM doesn't return title, it gets set to "Unknown"
metadata = extractor.extract("...")
# {"title": "Unknown"}
```

## Advanced Usage

### Custom Extraction Prompt

The extractor uses a built-in prompt template that emphasizes:
- Exact schema conformance
- Required vs optional fields
- Type normalization (dates, arrays, strings)
- No extra keys

For advanced use cases, you can subclass `MetadataExtractor` and override `_build_metadata_prompt()`.

### Validation with jsonschema

If the `jsonschema` package is installed, the extractor automatically validates output:

```bash
pip install jsonschema
```

```python
# Validation happens automatically if jsonschema is installed
config = MetadataExtractorConfig(json_schema={...})
extractor = MetadataExtractor(llm=llm, config=config)

try:
    metadata = extractor.extract("...")
except ValueError as e:
    print(f"Schema validation failed: {e}")
```

### Document Truncation

For very long documents, configure truncation:

```python
config = MetadataExtractorConfig(
    json_schema={...},
    truncate_document_chars=2000  # Only first 2000 chars sent to LLM
)

# Reduces cost and latency for long documents
# Extract from beginning where metadata is usually located
```

## Performance Considerations

### 1. Truncation Strategy

```python
# Default truncation works for most documents (metadata at start)
config = MetadataExtractorConfig(
    json_schema={...},
    truncate_document_chars=4000
)

# Increase for documents with metadata spread throughout
config = MetadataExtractorConfig(
    json_schema={...},
    truncate_document_chars=8000
)
```

### 2. Schema Complexity

- Simpler schemas = more reliable extraction
- Limit number of properties to 10-15 for best results
- Break complex schemas into multiple extraction passes if needed

### 3. LLM Selection

- Larger models (7B+) = better metadata extraction
- Models trained on instruction following work best
- Test different models to find the best balance of speed/quality

### 4. Batch Processing

The extractor processes one document at a time. For batch processing, use the Crawler:

```python
from crawler import Crawler, CrawlerConfig

config = CrawlerConfig.create(
    extractor=MetadataExtractorConfig(...),
    # ... other config
)

crawler = Crawler(config)
crawler.crawl("documents/")  # Processes all documents
```

## Dependencies

- `pydantic>=2.0` - Configuration validation and data models
- `jsonschema` (optional) - Schema validation
- Parent modules: `crawler.llm`, `crawler.document`

## Related Documentation

- [LLM Module](../llm/overview.md) - LLM configuration and usage
- [Document Module](../document/overview.md) - Document data structure
- [Crawler Configuration](../config/overview.md) - Complete crawler setup
- [Examples](../../../examples/OVERVIEW.md) - Real-world extraction examples

