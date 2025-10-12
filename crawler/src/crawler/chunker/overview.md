# Chunker Module Overview

This module provides text chunking functionality for the crawler system.

## Files

- `__init__.py` - Module initialization and exports
- `chunker.py` - Main chunker implementation with Chunker class and ChunkingConfig
- `overview.md` - This documentation file

## Classes

### ChunkingConfig
Pydantic configuration class for chunking parameters:
- `chunk_size`: Maximum size of each chunk in characters (default: 1000)
- `overlap`: Number of characters to overlap between consecutive chunks (default: 200)
- `strategy`: Chunking strategy to use (default: "naive")
- `preserve_paragraphs`: Whether to try to preserve paragraph boundaries (default: True)
- `min_chunk_size`: Minimum size of a chunk (default: 100)

### Chunker
Main chunker class that implements text chunking:
- `chunk_text(text: str) -> List[str]`: Split text into chunks
- `get_chunk_count(text: str) -> int`: Estimate number of chunks for given text

## Chunking Strategies

Currently supports:
- **naive**: Simple character-based chunking with optional paragraph/sentence boundary preservation

## Usage

```python
from crawler.chunker import Chunker, ChunkingConfig

# Create configuration
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=200,
    preserve_paragraphs=True
)

# Create chunker
chunker = Chunker(config)

# Chunk text
chunks = chunker.chunk_text("Your text here...")
```
