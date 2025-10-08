# Processing Module Overview

This module provides a type-safe, modular document processing pipeline for converting, extracting, embedding, and analyzing documents. All configuration models use Pydantic BaseModels for automatic validation and type safety.

## Files in This Module

### `__init__.py`
Exports the public API for the processing module. Provides clean imports for:
- `Converter`, `ConverterConfig`, and implementations (MarkItDown, Docling, PyMuPDF)
- `Extractor`, `ExtractorConfig`, and implementations (Basic, MultiSchema)
- `LLM`, `LLMConfig`, and implementations (Ollama, vLLM)
- `Embedder`, `EmbedderConfig`, and implementations (Ollama)
- Factory functions: `create_converter`, `create_extractor`, `get_llm`, `get_embedder`

### `converter.py`
Document conversion implementations that transform various file formats to Markdown.

**Pydantic Models:**
- `ConverterConfig` - Type-safe configuration for document converters
  - Fields: type, vision_llm (optional), metadata (optional)
  - Factory methods: `markitdown()`, `docling()`, `docling_vlm()`, `pymupdf()`
  - Validates converter type and vision LLM requirements
  
- `ExtractedImage` - Data model for extracted images from documents
  - Fields: page_number (≥0), image_index (≥0), bbox, image_data (bytes), image_ext, description
  - Validates page numbers and image indices are non-negative
  - Supports arbitrary types (bytes) for image data

**Abstract Base Class:**
- `Converter` - Interface for all converter implementations
  - Methods: `convert(filepath)` - converts file to markdown

**Implementations:**
- `MarkItDownConverter` - Uses MarkItDown library with vision model support
  - Supports various document formats
  - Uses OpenAI-compatible API for vision processing
  - Comprehensive logging and statistics
  
- `DoclingConverter` - Specialized PDF converter with VLM integration
  - Uses Docling library with custom VLM pipeline
  - Configurable vision model parameters (timeout, scale, prompt)
  - Advanced PDF processing with layout analysis
  
- `DoclingVLMConverter` - Docling with default VLM configuration
  - Simplified version using Docling's built-in VLM
  - No custom VLM configuration required
  
- `PyMuPDFConverter` - Comprehensive PDF converter
  - Extracts text, images, and tables
  - Uses PyMuPDF for detailed PDF analysis
  - AI-powered image description via `ImageDescriptionInterface`
  - Configurable table extraction strategies
  - Preserves reading order and formatting options

**Image Description Services:**
- `ImageDescriptionInterface` - Abstract interface for image description
- `OllamaImageDescriber` - Ollama-based image description
- `DummyImageDescriber` - Placeholder for testing

**Factory Function:**
- `create_converter(type, config)` - Creates converter instances based on type

### `extractor.py`
Metadata extraction and text chunking implementations.

**Pydantic Model:**
- `ExtractorConfig` - Type-safe configuration for metadata extractors
  - Fields: type, llm (optional), metadata_schema (optional), document_library_context
  - Factory methods: `basic()`, `multi_schema()`
  - Validates extractor type

**Abstract Base Class:**
- `Extractor` - Interface for all extractor implementations
  - Methods: `extract_metadata(text)`, `chunk_text(text, chunk_size)`

**Implementations:**
- `BasicExtractor` - Standard metadata extraction using LLM
  - Uses JSON schema for structured extraction
  - Supports benchmark question generation
  - Comprehensive logging of extraction process
  - Validates required fields and reports missing data
  
- `MultiSchemaExtractor` - Applies multiple schemas sequentially
  - Processes document with multiple extraction schemas
  - Aggregates results from all schemas
  - Tracks success/failure for each schema
  - Detailed statistics on extraction performance

**Helper Functions:**
- `create_extractor(config, llm)` - Factory function for creating extractors
- `generate_benchmark_questions(llm, text, n)` - Generates test questions for documents

**Prompts:**
- `extract_metadata_prompt` - Template for metadata extraction with strict output contract

### `llm.py`
Large Language Model interaction layer with support for multiple providers.

**Pydantic Model:**
- `LLMConfig` - Type-safe configuration for LLM providers
  - Fields: model_name, base_url, system_prompt, ctx_length (>0), default_timeout (>0), provider, api_key, structured_output
  - Factory methods: `ollama()`, `openai()`, `vllm()`
  - Validates context length and timeout are positive
  - Validates structured_output mode ('response_format' or 'tools')

**Abstract Base Class:**
- `LLM` - Interface for all LLM implementations
  - Methods: `invoke(prompt_or_messages, response_format, tools)`
  - Supports both text and structured JSON output

**Implementations:**
- `OllamaLLM` - Ollama provider implementation
  - Uses official Ollama Python client
  - Supports response_format and tools modes for structured output
  - Comprehensive logging and performance tracking
  - Handles timeouts and errors gracefully
  - Automatic JSON parsing with cleanup
  
- `VllmLLM` - vLLM provider implementation
  - Uses OpenAI-compatible /v1/chat/completions API
  - Supports json_schema response_format and tools
  - Similar logging and error handling as OllamaLLM
  - Uses httpx for HTTP requests

**Helper Functions:**
- `get_llm(config)` - Factory function for creating LLM instances
- `schema_to_openai_tools(schema)` - Converts JSON schema to OpenAI tools format

### `embeddings.py`
Vector embedding generation for text chunks.

**Pydantic Model:**
- `EmbedderConfig` - Type-safe configuration for embedding providers
  - Fields: model, base_url, api_key, provider, dimension (optional, >0)
  - Factory methods: `ollama()`, `openai()`
  - Validates model and base_url are non-empty
  - Validates dimension is positive if provided

**Abstract Base Class:**
- `Embedder` - Interface for all embedder implementations
  - Methods: `embed(query)`, `get_dimension()`

**Implementation:**
- `OllamaEmbedder` - Ollama embedding provider
  - Uses langchain_ollama for embedding generation
  - Caches dimension after first query
  - Supports batch embedding with progress tracking
  - Comprehensive logging of embedding statistics
  - Error handling and retry logic

**Factory Function:**
- `get_embedder(config)` - Creates embedder instances based on provider

### `processing.md`
Comprehensive documentation for the processing pipeline:
- Overview of the document processing stages
- Detailed API documentation for each module
- Error handling strategies and best practices
- Configuration management guidelines
- Examples of custom implementations
- Troubleshooting guide for common issues
- Performance optimization tips

## Design Decisions

### Pydantic for Type Safety
All configuration models use Pydantic BaseModel instead of dataclasses to provide:
- Automatic validation at creation and assignment time
- Field-level constraints (min_length, gt, ge)
- Clear error messages for invalid data
- Runtime type checking
- Easy serialization/deserialization
- IDE autocomplete support
- Factory methods with validated parameters

### Modular Architecture
The pipeline is designed as independent, composable stages:
1. **Conversion** - File format → Markdown
2. **Extraction** - Markdown → Structured metadata + chunks
3. **LLM Interaction** - Structured API for language models
4. **Embedding** - Text → Vector representations

Each stage can be swapped or customized without affecting others.

### Provider-Agnostic Interfaces
Abstract base classes define contracts for:
- Converters (`Converter`)
- Extractors (`Extractor`)
- LLMs (`LLM`)
- Embedders (`Embedder`)

This allows easy addition of new providers (e.g., Azure OpenAI, Anthropic) without changing pipeline code.

### Comprehensive Logging
All implementations include:
- Start/end markers with emoji indicators
- Detailed statistics (processing time, throughput, success rates)
- Progress bars for long operations (tqdm)
- Error messages with context
- Debug information for troubleshooting

### Structured Output Modes
LLM implementations support two modes for getting structured JSON:
1. **response_format** - JSON schema passed directly to model
2. **tools** - OpenAI-style function calling

Both modes are supported across Ollama and vLLM implementations.

### Vision Model Integration
Document converters support vision models for:
- Image analysis and description
- Complex layout understanding
- Table and diagram extraction
- Optical character recognition (OCR)

Vision models are configured via LLMConfig and used consistently across converters.

## Configuration Examples

### Full Pipeline Configuration

```python
from crawler.processing import (
    ConverterConfig, ExtractorConfig, LLMConfig, EmbedderConfig
)

# Configure LLM for metadata extraction
llm_config = LLMConfig.ollama(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    ctx_length=32000,
    structured_output="response_format"
)

# Configure vision LLM for document conversion
vision_llm_config = LLMConfig.ollama(
    model_name="llava",
    base_url="http://localhost:11434"
)

# Configure converter
converter_config = ConverterConfig.pymupdf(
    vision_llm=vision_llm_config,
    metadata={
        "extract_tables": True,
        "table_strategy": "lines_strict",
        "image_describer": {
            "type": "ollama",
            "model": "granite3.2-vision:latest",
            "base_url": "http://localhost:11434"
        }
    }
)

# Configure extractor
extractor_config = ExtractorConfig.basic(
    llm=llm_config,
    metadata_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string"}
        },
        "required": ["title"]
    },
    document_library_context="Research papers from arXiv"
)

# Configure embedder
embedder_config = EmbedderConfig.ollama(
    model="all-minilm:v2",
    base_url="http://localhost:11434",
    dimension=384
)
```

### Validation Examples

All models validate inputs automatically:

```python
# This will raise ValidationError
try:
    bad_config = LLMConfig(
        model_name="",  # Empty string not allowed
        base_url="http://localhost:11434"
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# This will raise ValidationError
try:
    bad_embed = EmbedderConfig(
        model="test",
        base_url="http://test.com",
        dimension=-1  # Must be positive
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Dependencies

- `pydantic>=2.0` - Data validation and settings management
- `pymupdf>=1.26.4` - PDF processing
- `markitdown>=0.1.2` - Document conversion
- `docling>=2.47.1` - Advanced PDF conversion
- `ollama>=0.5.3` - Ollama API client
- `openai>=1.101.0` - OpenAI-compatible client
- `langchain-ollama>=0.2.0` - Ollama embeddings
- `httpx` - HTTP client for vLLM
- `tqdm>=4.66.0` - Progress bars
- `requests>=2.32.5` - HTTP requests for image description

## Usage Examples

See `processing.md` for detailed usage examples, or refer to:
- `examples/arxiv.py` - Processing academic papers
- `examples/xmidas.py` - XMIDAS document processing
- `examples/church.py` - Church document processing

