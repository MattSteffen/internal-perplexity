# Converter Package Overview

This package provides a unified interface for document conversion using PyMuPDF4LLM. It features type-safe configuration and direct integration with the Document pipeline.

## Package Structure

```
converter/
├── __init__.py          # Public API exports
├── base.py              # Converter abstract base class
├── types.py             # Conversion stats, events, capabilities
├── factory.py           # create_converter(config: ConverterConfig)
├── pymupdf4llm.py       # PyMuPDF4LLMConverter, PyMuPDF4LLMConfig, and VLM interfaces
└── overview.md          # This file
```

## Core Components

### 1. Base Interface (`base.py`)
- **Converter**: Abstract base class defining the converter interface
  - `name` (property): Human-friendly name for the converter backend
  - `convert(document: Document) -> None`: Convert a Document in place, populating converter fields
- All converters must implement the abstract `convert()` method
- Converters modify Document objects directly, populating: `content`, `markdown`, `stats`, `source_name`, and `warnings`

### 2. Type System (`types.py`)
- **ConversionStats**: Statistics including total_pages, processed_pages, text_blocks, images, images_described, tables, total_time_sec
- **ProgressEvent**: Progress tracking events (stage, page, total_pages, message, metrics)
- **Capabilities**: Converter capability descriptions

### 3. Configuration (`factory.py`)
- **ConverterConfig**: Type alias for PyMuPDF4LLMConfig
- **create_converter(config: ConverterConfig) -> Converter**: Factory function that creates a PyMuPDF4LLMConverter instance

### 4. Converter Implementation

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

### Basic Usage with Document
```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.document import Document

# Create converter configuration
config = PyMuPDF4LLMConfig(type="pymupdf4llm")
converter = create_converter(config)

# Create and convert document
doc = Document.create(source="document.pdf")
converter.convert(doc)  # Modifies doc in place

# Access results
print(doc.markdown)
if doc.stats:
    print(f"Processed {doc.stats.total_pages} pages")
```

### Using with VLM for Image Description
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
converter.convert(doc)  # Modifies doc in place

# Access results
print(doc.markdown)
print(doc.stats)
```

### Converting Document with Pre-loaded Content
```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.document import Document

converter = create_converter(PyMuPDF4LLMConfig(type="pymupdf4llm"))

# Read file as bytes and create document with content
with open("document.pdf", "rb") as f:
    data = f.read()

doc = Document.create(source="document.pdf")
doc.content = data  # Pre-load content
converter.convert(doc)  # Uses existing content
```

### PyMuPDF4LLM with Custom Image Prompt
```python
from crawler.converter import create_converter, PyMuPDF4LLMConfig
from crawler.document import Document
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
converter.convert(doc)
```

## Key Features

### Type Safety
- Pydantic-based configuration
- Comprehensive type hints throughout
- Runtime validation of inputs and outputs
- Type-safe factory function

### Document Pipeline Integration
- `convert()` method modifies Document objects in place
- Automatically populates Document fields (content, markdown, stats, source_name, warnings)
- Seamless integration with the crawler pipeline
- Handles both documents with pre-loaded content and documents that need content read from source path

### Rich Results
- Direct modification of Document objects with all conversion results
- Performance metrics and conversion statistics stored in Document.stats
- Image description with VLM integration
- Warning messages collected during conversion

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
Converters accept `Document` objects directly:
- Documents can be created with `Document.create(source="file.pdf")`
- If `document.content` is set, converter uses those bytes directly
- Otherwise, converter reads from `document.source` as a file path

### Output
- Converters modify Document objects in place, populating:
  - `content`: Raw binary content (if not already set)
  - `markdown`: Converted markdown text
  - `stats`: ConversionStats with performance metrics
  - `source_name`: Source filename (if not already set)
  - `warnings`: List of warning messages

## Dependencies

- **Core**: pydantic, pathlib, typing
- **PyMuPDF**: pymupdf, pymupdf4llm
- **VLM**: requests (for OllamaVLM integration)

## Design Principles

1. **Type Safety First**: Comprehensive type hints and Pydantic validation
2. **Unified Interface**: Consistent API for converter operations
3. **Rich Results**: Structured output with metadata, assets, and statistics
4. **Document Integration**: Native support for Document pipeline workflow
5. **Error Handling**: Graceful failure with detailed error information
6. **Performance**: Concurrent image processing for faster conversion
