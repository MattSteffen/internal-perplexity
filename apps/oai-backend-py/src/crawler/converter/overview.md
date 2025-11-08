# Converter Package Overview

This package provides a unified interface for document conversion with support for multiple backends including MarkItDown, Docling, and PyMuPDF. It features type-safe configuration, progress tracking, and rich result objects.

## Package Structure

```
converter/
├── __init__.py          # Public API exports
├── base.py              # Converter interface, exceptions
├── types.py             # DocumentInput, ConvertOptions, results, events
├── configs.py           # Discriminated union config models
├── factory.py           # create_converter(config: ConverterConfig)
├── registry.py          # Plugin registry for converters by name
├── markitdown.py        # MarkItDownConverter
├── docling.py           # DoclingConverter
├── docling_vlm.py       # DoclingVLMConverter
├── pymupdf.py           # PyMuPDFConverter and ImageDescriber impls
└── overview.md          # This file
```

## Core Components

### 1. Base Interface (`base.py`)
- **Converter**: Abstract base class defining the converter interface
- **ProgressCallback**: Type alias for progress event handlers
- Provides sync/async conversion methods with progress tracking
- Supports batch conversion with configurable concurrency

### 2. Type System (`types.py`)
- **DocumentInput**: Unified input representation (path, bytes, fileobj)
- **ConvertOptions**: Configuration for conversion behavior
- **ConvertedDocument**: Rich result object with markdown, assets, and stats
- **ProgressEvent**: Progress tracking events
- **Capabilities**: Converter capability descriptions
- **ImageAsset/TableAsset**: Extracted content assets
- **ConversionStats**: Performance and content statistics

### 3. Configuration (`configs.py`)
- **BaseConverterConfig**: Base configuration class
- **MarkItDownConfig**: MarkItDown-specific configuration
- **DoclingConfig**: Docling-specific configuration
- **DoclingVLMConfig**: Docling VLM configuration
- **PyMuPDFConfig**: PyMuPDF-specific configuration
- **ConverterConfig**: Discriminated union of all config types

### 4. Factory System (`factory.py`, `registry.py`)
- **create_converter()**: Type-safe converter creation from config
- **registry**: Plugin system for dynamic converter registration
- Automatic type mapping and validation

## Converter Implementations

### MarkItDownConverter (`markitdown.py`)
- **Purpose**: AI-powered document conversion with vision models
- **Supports**: PDF, DOCX, HTML, plain text
- **Features**: Vision model integration, plugin support
- **Configuration**: LLM base URL, model name, API key, plugins

### DoclingConverter (`docling.py`)
- **Purpose**: Advanced PDF processing with custom VLM integration
- **Supports**: PDF only
- **Features**: Custom vision model configuration, table extraction
- **Configuration**: VLM base URL, model, prompt, timeout, scale

### DoclingVLMConverter (`docling_vlm.py`)
- **Purpose**: PDF processing with Docling's default VLM
- **Supports**: PDF only
- **Features**: Uses Docling's built-in VLM configuration
- **Configuration**: Minimal (relies on defaults)

### PyMuPDFConverter (`pymupdf.py`)
- **Purpose**: Comprehensive PDF processing with content extraction
- **Supports**: PDF only
- **Features**: 
  - Text extraction with reading order preservation
  - Image extraction and AI-powered description
  - Table detection and markdown conversion
  - Metadata extraction
  - Page-by-page processing with progress tracking
- **Configuration**: Image describer settings, table strategy

## Usage Examples

### Basic Usage
```python
from crawler.converter import create_converter, MarkItDownConfig, DocumentInput

# Create converter
config = MarkItDownConfig(
    type="markitdown",
    llm_base_url="http://localhost:11434",
    llm_model="llava"
)
converter = create_converter(config)

# Convert document
doc = DocumentInput.from_path("document.pdf")
result = converter.convert(doc)
print(result.markdown)
```

### With Progress Tracking
```python
from crawler.converter import PyMuPDFConfig, ConvertOptions, ProgressEvent

def on_progress(event: ProgressEvent) -> None:
    print(f"[{event.stage}] page={event.page}/{event.total_pages}: {event.message}")

config = PyMuPDFConfig(
    type="pymupdf",
    image_describer={"type": "ollama", "model": "granite3.2-vision:latest"}
)
converter = create_converter(config)

options = ConvertOptions(
    extract_tables=True,
    describe_images=True,
    page_range=(1, 10),
)

result = converter.convert(
    DocumentInput.from_path("file.pdf"),
    options=options,
    on_progress=on_progress,
)
```

### Async Batch Processing
```python
import asyncio
from crawler.converter import create_converter, MarkItDownConfig

async def main():
    converter = create_converter(
        MarkItDownConfig(
            type="markitdown",
            llm_base_url="http://localhost:11434",
            llm_model="llava"
        )
    )
    
    docs = [
        DocumentInput.from_path("a.pdf"),
        DocumentInput.from_path("b.docx"),
    ]
    
    results = await converter.aconvert_many(docs, concurrency=2)
    print([r.stats for r in results])

asyncio.run(main())
```

### Registry Usage
```python
from crawler.converter import registry

# Create converter by name
converter = registry.create(
    "pymupdf", 
    image_describer={"type": "dummy"}
)

result = converter.convert(DocumentInput.from_path("document.pdf"))
```

## Key Features

### Type Safety
- Pydantic-based configuration with discriminated unions
- Comprehensive type hints throughout
- Runtime validation of inputs and outputs

### Progress Tracking
- Optional progress callbacks for long-running operations
- Rich progress events with metrics and status information
- Page-by-page progress for PDF processing

### Rich Results
- Structured result objects with markdown, assets, and statistics
- Separate handling of images, tables, and metadata
- Performance metrics and conversion statistics

### Extensibility
- Plugin registry system for adding new converters
- Abstract interfaces for image description services
- Configurable options for all conversion aspects

### Error Handling
- Comprehensive exception hierarchy
- Graceful handling of unsupported formats
- Detailed logging and error reporting

## Migration from Legacy Code

The new converter package maintains compatibility with the existing converter.py while providing:

1. **Better Type Safety**: Pydantic models replace manual validation
2. **Unified Interface**: Consistent API across all converter types
3. **Rich Results**: Structured output with assets and statistics
4. **Progress Tracking**: Built-in support for progress callbacks
5. **Async Support**: Native async/await support for all operations
6. **Plugin System**: Easy extension with new converter types

## Dependencies

- **Core**: pydantic, pathlib, typing
- **MarkItDown**: markitdown, openai
- **Docling**: docling
- **PyMuPDF**: pymupdf, requests (for Ollama integration)
- **Async**: asyncio (built-in)

## Design Principles

1. **Type Safety First**: Comprehensive type hints and validation
2. **Unified Interface**: Consistent API across all implementations
3. **Rich Results**: Structured output with metadata and assets
4. **Progress Awareness**: Built-in progress tracking capabilities
5. **Extensibility**: Easy to add new converter types and features
6. **Error Handling**: Graceful failure with detailed error information
7. **Performance**: Efficient processing with configurable concurrency