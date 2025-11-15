# Converter Package Overview

This package provides a unified interface for document conversion with support for multiple backends including MarkItDown and PyMuPDF4LLM. It features type-safe configuration, rich result objects, and integration with the Document pipeline.

## Package Structure

```
converter/
├── __init__.py          # Public API exports
├── base.py              # Converter abstract base class
├── types.py             # DocumentInput, ConvertOptions, results, events
├── factory.py           # create_converter(config: ConverterConfig)
├── markitdown.py        # MarkItDownConverter and MarkItDownConfig
├── pymupdf4llm.py       # PyMuPDF4LLMConverter, PyMuPDF4LLMConfig, and VLM interfaces
└── overview.md          # This file
```

## Core Components

### 1. Base Interface (`base.py`)
- **Converter**: Abstract base class defining the converter interface
  - `name` (property): Human-friendly name for the converter backend
  - `convert(doc: DocumentInput) -> ConvertedDocument`: Convert a document from DocumentInput
  - `convert_document(document: Document) -> None`: Convert a Document in place (preferred method for Document pipeline)
- All converters must implement the abstract `convert()` method
- The `convert_document()` method is provided by the base class and handles Document integration

### 2. Type System (`types.py`)
- **DocumentInput**: Unified input representation supporting:
  - `from_path(p: str | Path)`: Create from file path
  - `from_bytes(data: bytes, filename: str, mime_type: str)`: Create from bytes
  - `from_fileobj(f: IO[bytes], filename: str, mime_type: str)`: Create from file-like object
  - `from_document(document: Document)`: Create from Document object
- **ConvertOptions**: Configuration for conversion behavior (include_metadata, include_images, describe_images, extract_tables, etc.)
- **ConvertedDocument**: Rich result object with:
  - `markdown`: Converted markdown text
  - `source_name`: Source filename
  - `images`: List of ImageAsset objects
  - `tables`: List of TableAsset objects
  - `stats`: ConversionStats with performance metrics
  - `warnings`: List of warning messages
  - `metadata`: Dictionary of extracted metadata
- **ImageAsset**: Extracted image with page_number, bbox, ext, data (bytes), description
- **TableAsset**: Extracted table with page_number, bbox, rows, cols, markdown
- **ConversionStats**: Statistics including total_pages, processed_pages, text_blocks, images, images_described, tables, total_time_sec
- **ProgressEvent**: Progress tracking events (stage, page, total_pages, message, metrics)
- **Capabilities**: Converter capability descriptions

### 3. Configuration (`factory.py`)
- **ConverterConfig**: Discriminated union type (using Pydantic Field discriminator) of:
  - `MarkItDownConfig`
  - `PyMuPDF4LLMConfig`
- **create_converter(config: ConverterConfig) -> Converter**: Factory function that creates the appropriate converter based on config type

### 4. Converter Implementations

#### MarkItDownConverter (`markitdown.py`)
- **Purpose**: AI-powered document conversion using the MarkItDown library
- **Supports**: PDF, DOCX, HTML, plain text, and other formats supported by MarkItDown
- **Features**: 
  - Vision model integration via LLM configuration
  - Plugin support (optional)
  - Handles various file formats automatically
- **Configuration** (`MarkItDownConfig`):
  - `type`: Literal["markitdown"]
  - `llm_config`: LLMConfig for vision processing
  - `enable_plugins`: bool (default: False)

#### PyMuPDF4LLMConverter (`pymupdf4llm.py`)
- **Purpose**: Comprehensive PDF processing using PyMuPDF and pymupdf4llm with VLM image description
- **Supports**: PDF only
- **Features**: 
  - Text extraction using pymupdf4llm.to_markdown()
  - Image extraction with base64 data-URI embedding
  - AI-powered image description using Vision Language Models (VLM)
  - Concurrent image description processing
  - Paragraph newline fixing for better markdown formatting
  - Image deduplication by content hash
- **Configuration** (`PyMuPDF4LLMConfig`):
  - `type`: Literal["pymupdf4llm"]
  - `vlm_config`: Optional LLMConfig for VLM (if None, images are not described)
  - `image_prompt`: Optional custom prompt for image description
  - `max_workers`: int (1-32, default: 4) for concurrent image description
  - `to_markdown_kwargs`: dict for additional pymupdf4llm.to_markdown() parameters
- **VLM Interfaces**:
  - `VLMInterface`: Abstract interface for image description
  - `OllamaVLM`: Implementation using Ollama API
  - `DummyVLM`: Fallback implementation for testing

## Usage Examples

### Basic Usage with DocumentInput
```python
from crawler.converter import create_converter, MarkItDownConfig, DocumentInput
from crawler.llm import LLMConfig

# Create converter configuration
llm_config = LLMConfig.ollama(
    model_name="llava",
    base_url="http://localhost:11434"
)
config = MarkItDownConfig(
    type="markitdown",
    llm_config=llm_config
)
converter = create_converter(config)

# Convert document from file path
doc_input = DocumentInput.from_path("document.pdf")
result = converter.convert(doc_input)
print(result.markdown)
print(f"Processed {result.stats.total_pages} pages")
```

### Using with Document Pipeline (Recommended)
```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.document import Document
from crawler.llm import LLMConfig

# Create converter with VLM for image description
vlm_config = LLMConfig.ollama(
    model_name="granite3.2-vision:latest",
    base_url="http://localhost:11434"
)
config = PyMuPDF4LLMConfig(
    type="pymupdf4llm",
    vlm_config=vlm_config,
    max_workers=4
)
converter = create_converter(config)

# Create and convert document
doc = Document.create(source="document.pdf")
converter.convert_document(doc)  # Modifies doc in place

# Access results
print(doc.markdown)
print(f"Found {len(doc.images)} images")
print(f"Found {len(doc.tables)} tables")
print(doc.stats)
```

### Converting from Bytes
```python
from crawler.converter import create_converter, MarkItDownConfig, DocumentInput
from crawler.llm import LLMConfig

converter = create_converter(
    MarkItDownConfig(
        type="markitdown",
        llm_config=LLMConfig.ollama(model_name="llava")
    )
)

# Read file as bytes
with open("document.pdf", "rb") as f:
    data = f.read()

# Convert from bytes
doc_input = DocumentInput.from_bytes(
    data=data,
    filename="document.pdf",
    mime_type="application/pdf"
)
result = converter.convert(doc_input)
```

### PyMuPDF4LLM with Custom Image Prompt
```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.llm import LLMConfig

config = PyMuPDF4LLMConfig(
    type="pymupdf4llm",
    vlm_config=LLMConfig.ollama(
        model_name="llava:latest",
        base_url="http://localhost:11434"
    ),
    image_prompt="Describe this image focusing on any text, diagrams, or technical content.",
    max_workers=8  # More workers for faster processing
)
converter = create_converter(config)

doc = Document.create(source="technical_document.pdf")
converter.convert_document(doc)
```

## Key Features

### Type Safety
- Pydantic-based configuration with discriminated unions
- Comprehensive type hints throughout
- Runtime validation of inputs and outputs
- Type-safe factory function with automatic converter selection

### Document Pipeline Integration
- `convert_document()` method modifies Document objects in place
- Automatically populates Document fields (markdown, images, tables, stats, warnings)
- Seamless integration with the crawler pipeline

### Rich Results
- Structured result objects with markdown, assets, and statistics
- Separate handling of images, tables, and metadata
- Performance metrics and conversion statistics
- Image and table extraction with bounding boxes and page numbers

### Image Description
- VLM integration for AI-powered image description
- Concurrent processing for multiple images
- Image deduplication by content hash
- Configurable prompts and VLM backends

### Error Handling
- Graceful handling of unsupported formats
- Exception handling for file I/O errors
- Detailed error messages and warnings

## Input/Output

### Input Types
Converters accept `DocumentInput` which can be created from:
- File paths: `DocumentInput.from_path("file.pdf")`
- Bytes data: `DocumentInput.from_bytes(data, filename="file.pdf")`
- File objects: `DocumentInput.from_fileobj(file_obj, filename="file.pdf")`
- Document objects: `DocumentInput.from_document(document)`

### Output Types
- `ConvertedDocument`: Contains markdown, images, tables, stats, warnings, metadata
- For Document pipeline: Modifies Document object in place via `convert_document()`

## Dependencies

- **Core**: pydantic, pathlib, typing
- **MarkItDown**: markitdown, openai (for MarkItDownConverter)
- **PyMuPDF**: pymupdf, pymupdf4llm (for PyMuPDF4LLMConverter)
- **VLM**: requests (for OllamaVLM integration)

## Design Principles

1. **Type Safety First**: Comprehensive type hints and Pydantic validation
2. **Unified Interface**: Consistent API across all converter implementations
3. **Rich Results**: Structured output with metadata, assets, and statistics
4. **Document Integration**: Native support for Document pipeline workflow
5. **Extensibility**: Easy to add new converter types by implementing Converter interface
6. **Error Handling**: Graceful failure with detailed error information
7. **Performance**: Concurrent image processing for faster conversion