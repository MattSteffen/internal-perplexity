# Chunker Module Overview

This module provides text chunking functionality for splitting documents into smaller, manageable pieces suitable for embedding and storage in vector databases. It supports configurable chunk sizes, overlap strategies, and boundary preservation.

## Files in This Module

### `__init__.py`

Exports the public API for the chunker module:
- `Chunker` - Main chunker class for splitting text
- `ChunkingConfig` - Pydantic configuration model

### `chunker.py`

Contains the text chunking implementation with the following components:

**Key Classes:**
- **`ChunkingConfig`**: Pydantic model for configuring text chunking behavior
- **`Chunker`**: Main class for splitting text into chunks

**Key Methods:**
- `chunk_text(document: Document) -> list[str]` - Split document text into chunks
- `get_chunk_count(text: str) -> int` - Estimate the number of chunks for given text

## Core Concepts

### Why Chunking?

Text chunking is essential for:
1. **Embedding Models**: Most embedding models have token limits (e.g., 512-8192 tokens)
2. **Search Precision**: Smaller chunks improve retrieval accuracy by reducing noise
3. **Context Management**: Chunks provide manageable context windows for LLMs
4. **Memory Efficiency**: Smaller chunks reduce memory requirements for processing

### Chunking Configuration

The `ChunkingConfig` class provides comprehensive configuration:

**Fields:**
- `chunk_size` (int): Maximum chunk size in characters (default: 1000, must be > 0)
- `overlap` (int): Characters to overlap between chunks (default: 200, must be >= 0)
- `strategy` (str): Chunking strategy to use (default: "naive")
- `preserve_paragraphs` (bool): Try to preserve paragraph boundaries (default: True)
- `min_chunk_size` (int): Minimum chunk size in characters (default: 100, must be > 0)

**Factory Method:**
- `ChunkingConfig.create(...)` - Create config with custom parameters

### Chunking Strategies

Currently, the chunker implements the **"naive"** strategy:

**Naive Strategy:**
- Splits text by character count
- Respects paragraph boundaries when `preserve_paragraphs=True`
- Falls back to sentence boundaries if no paragraph breaks found
- Falls back to word boundaries if no sentence breaks found
- Always respects minimum chunk size
- Creates overlapping chunks based on `overlap` setting

Future strategies may include:
- `semantic` - Split based on semantic similarity
- `recursive` - Recursive splitting with multiple delimiters
- `sentence` - Split at sentence boundaries only

### Overlap Strategy

Chunks can overlap to maintain context continuity:

```
Text: [AAAAA BBBBB CCCCC DDDDD EEEEE]

chunk_size=10, overlap=3:
Chunk 1: [AAAAA BBB]
Chunk 2:    [BBB CCCCC]
Chunk 3:        [CCC DDDDD]
Chunk 4:            [DDD EEEEE]
```

Overlapping helps:
- Preserve context across chunk boundaries
- Improve search recall (relevant text appears in multiple chunks)
- Reduce edge effects from splitting

## Usage Examples

### Basic Chunking

```python
from crawler.chunker import Chunker, ChunkingConfig
from crawler.document import Document

# Create default configuration
config = ChunkingConfig.create(chunk_size=1000, overlap=200)
chunker = Chunker(config)

# Create document with markdown
doc = Document.create(source="document.pdf")
doc.markdown = """
# Introduction

This is a long document that needs to be split into smaller chunks
for processing and embedding...

## Section 1

More content here...
"""

# Chunk the text
chunks = chunker.chunk_text(doc)
print(f"Created {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (length: {len(chunk)}): {chunk[:100]}...")
```

### Custom Chunk Sizes

```python
from crawler.chunker import ChunkingConfig, Chunker

# Small chunks for precise search
small_config = ChunkingConfig.create(
    chunk_size=500,
    overlap=100,
    min_chunk_size=50
)

# Medium chunks for balanced search/context
medium_config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=200,
    min_chunk_size=100
)

# Large chunks for more context
large_config = ChunkingConfig.create(
    chunk_size=2000,
    overlap=400,
    min_chunk_size=200
)

# Use appropriate configuration
chunker = Chunker(medium_config)
```

### Preserving Paragraph Boundaries

```python
from crawler.chunker import ChunkingConfig, Chunker
from crawler.document import Document

# Enable paragraph preservation (default)
config = ChunkingConfig.create(
    chunk_size=1000,
    preserve_paragraphs=True  # Try to break at paragraph boundaries
)
chunker = Chunker(config)

doc = Document.create(source="doc.md")
doc.markdown = """
Paragraph 1 with some content that explains the first concept.

Paragraph 2 with different content about another topic.

Paragraph 3 continues with more information.
"""

chunks = chunker.chunk_text(doc)
# Chunks will try to respect paragraph breaks (double newlines)
```

### Disabling Paragraph Preservation

```python
from crawler.chunker import ChunkingConfig, Chunker

# Strict character-based chunking
config = ChunkingConfig.create(
    chunk_size=1000,
    preserve_paragraphs=False  # Pure character count splitting
)
chunker = Chunker(config)

# Chunks split at exact character boundaries
```

### Estimating Chunk Count

```python
from crawler.chunker import Chunker, ChunkingConfig

config = ChunkingConfig.create(chunk_size=1000, overlap=200)
chunker = Chunker(config)

text = "Your long document text here..." * 100

# Estimate chunks before processing
estimated_count = chunker.get_chunk_count(text)
print(f"Estimated chunks: {estimated_count}")

# Actual chunking
chunks = chunker.chunk_text(document)
print(f"Actual chunks: {len(chunks)}")
```

### No Overlap Configuration

```python
from crawler.chunker import ChunkingConfig, Chunker

# Chunks with no overlap
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=0  # No overlap between chunks
)
chunker = Chunker(config)

# Each chunk is completely separate
```

## Choosing Chunk Sizes

### Small Chunks (300-700 characters)

**Best For:**
- Precise search and retrieval
- Question-answering systems
- Code documentation
- Structured data

**Pros:**
- High precision in search results
- Less noise in retrieved context
- Better for specific queries

**Cons:**
- May split related information
- Requires more chunks (more storage/processing)
- Context fragmentation

**Example:**
```python
config = ChunkingConfig.create(
    chunk_size=500,
    overlap=100,
    min_chunk_size=50
)
```

### Medium Chunks (800-1500 characters)

**Best For:**
- General-purpose document processing
- Research papers
- Technical documentation
- Blog posts and articles

**Pros:**
- Good balance of precision and context
- Reasonable chunk count
- Works well with most embedding models

**Cons:**
- May not be optimal for all use cases
- Moderate context window

**Example:**
```python
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=200,
    min_chunk_size=100
)
```

### Large Chunks (1500-3000 characters)

**Best For:**
- Long-form content analysis
- Maintaining narrative flow
- Documents where context is crucial
- Summary generation

**Pros:**
- Preserves more context
- Fewer chunks to manage
- Better for understanding relationships

**Cons:**
- May reduce search precision
- Larger embeddings to store
- May exceed some model limits

**Example:**
```python
config = ChunkingConfig.create(
    chunk_size=2000,
    overlap=400,
    min_chunk_size=200
)
```

### Very Large Chunks (3000+ characters)

**Best For:**
- Full document embeddings
- Coarse-grained search
- Document classification

**Pros:**
- Maximum context preservation
- Minimal chunks

**Cons:**
- Lower search precision
- May exceed embedding model limits
- Higher storage requirements

**Example:**
```python
config = ChunkingConfig.create(
    chunk_size=4000,
    overlap=800,
    min_chunk_size=500
)
```

## Overlap Configuration

### Low Overlap (0-10% of chunk size)

```python
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=100  # 10% overlap
)
```

**Use When:**
- Storage is a concern
- Documents have clear section boundaries
- Processing speed is important

### Medium Overlap (15-25% of chunk size)

```python
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=200  # 20% overlap (default)
)
```

**Use When:**
- General-purpose applications
- Want to balance storage and context
- Standard recommendation

### High Overlap (30-50% of chunk size)

```python
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=400  # 40% overlap
)
```

**Use When:**
- Context continuity is critical
- Search recall is more important than storage
- Documents have continuous narrative

## Integration with Other Modules

### With Document Pipeline

```python
from crawler.document import Document
from crawler.chunker import Chunker, ChunkingConfig

# Document comes from converter with markdown populated
doc = Document.create(source="paper.pdf")
doc.markdown = "..."  # Populated by converter

# Chunk the markdown
config = ChunkingConfig.create(chunk_size=1000)
chunker = Chunker(config)
doc.chunks = chunker.chunk_text(doc)

print(f"Created {len(doc.chunks)} chunks")
```

### With Crawler

```python
from crawler import Crawler, CrawlerConfig
from crawler.chunker import ChunkingConfig
from crawler.llm import LLMConfig, EmbedderConfig
from crawler.vector_db import DatabaseClientConfig

# Configure chunking as part of crawler config
config = CrawlerConfig.create(
    embeddings=EmbedderConfig.ollama(model="all-minilm:v2"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    chunking=ChunkingConfig.create(
        chunk_size=1000,
        overlap=200,
        preserve_paragraphs=True
    ),
    database=DatabaseClientConfig.milvus(collection="docs"),
    # ... other config
)

crawler = Crawler(config)
crawler.crawl("documents/")
```

### With Embedder

```python
from crawler.chunker import Chunker, ChunkingConfig
from crawler.llm import EmbedderConfig, get_embedder
from crawler.document import Document

# Chunk text
config = ChunkingConfig.create(chunk_size=1000)
chunker = Chunker(config)

doc = Document.create(source="doc.md")
doc.markdown = "..."
doc.chunks = chunker.chunk_text(doc)

# Embed chunks
embedder = get_embedder(EmbedderConfig.ollama(model="all-minilm:v2"))
doc.text_embeddings = embedder.embed_batch(doc.chunks)

print(f"Created {len(doc.text_embeddings)} embeddings for {len(doc.chunks)} chunks")
```

## Advanced Usage

### Custom Chunking Strategy

For advanced use cases, you can subclass `Chunker` and implement custom strategies:

```python
from crawler.chunker import Chunker, ChunkingConfig

class SemanticChunker(Chunker):
    def chunk_text(self, document):
        # Custom semantic chunking logic
        # Could use sentence embeddings to group similar sentences
        text = document.markdown
        # ... implement semantic chunking ...
        return chunks

# Use custom chunker
config = ChunkingConfig.create(chunk_size=1000, strategy="semantic")
chunker = SemanticChunker(config)
```

### Dynamic Chunk Sizing

Adjust chunk size based on document characteristics:

```python
from crawler.chunker import Chunker, ChunkingConfig
from crawler.document import Document

def get_optimal_chunk_size(document: Document) -> int:
    """Determine optimal chunk size based on document length."""
    text_length = len(document.markdown)
    
    if text_length < 5000:
        return 500  # Small docs: small chunks
    elif text_length < 20000:
        return 1000  # Medium docs: medium chunks
    else:
        return 2000  # Large docs: large chunks

# Apply dynamic chunking
doc = Document.create(source="doc.pdf")
doc.markdown = "..."

optimal_size = get_optimal_chunk_size(doc)
config = ChunkingConfig.create(chunk_size=optimal_size)
chunker = Chunker(config)
chunks = chunker.chunk_text(doc)
```

### Handling Edge Cases

```python
from crawler.chunker import Chunker, ChunkingConfig
from crawler.document import Document

config = ChunkingConfig.create(chunk_size=1000, min_chunk_size=100)
chunker = Chunker(config)

# Empty document
doc = Document.create(source="empty.md")
doc.markdown = ""
chunks = chunker.chunk_text(doc)
print(chunks)  # []

# Very short document
doc.markdown = "Short text"
chunks = chunker.chunk_text(doc)
print(chunks)  # ["Short text"]

# Document shorter than chunk_size
doc.markdown = "This is a document that is shorter than the chunk size."
chunks = chunker.chunk_text(doc)
print(len(chunks))  # 1
```

## Performance Considerations

### 1. Chunk Size vs Processing Time

- **Smaller chunks** = More chunks = More embeddings to generate = Longer processing
- **Larger chunks** = Fewer chunks = Fewer embeddings = Faster processing

### 2. Chunk Size vs Storage

- More chunks = More database entries = More storage
- Consider storage costs when choosing chunk sizes

### 3. Chunk Size vs Search Quality

- **Too small**: May miss relevant context, lower recall
- **Too large**: May include irrelevant content, lower precision
- **Optimal**: Depends on your use case and query types

### 4. Overlap Trade-offs

- **More overlap** = Better recall, but more storage and processing
- **Less overlap** = Lower storage, but potential gaps in context
- **Sweet spot**: Usually 15-25% overlap

## Best Practices

### 1. Test Different Configurations

```python
from crawler.chunker import ChunkingConfig, Chunker
from crawler.document import Document

# Test multiple configurations
configs = [
    ChunkingConfig.create(chunk_size=500, overlap=100),
    ChunkingConfig.create(chunk_size=1000, overlap=200),
    ChunkingConfig.create(chunk_size=2000, overlap=400),
]

doc = Document.create(source="test.pdf")
doc.markdown = "..."

for config in configs:
    chunker = Chunker(config)
    chunks = chunker.chunk_text(doc)
    print(f"Config: chunk_size={config.chunk_size}, "
          f"overlap={config.overlap} -> {len(chunks)} chunks")
```

### 2. Consider Document Type

- **Code**: Small chunks (300-500) with low overlap
- **Research papers**: Medium chunks (1000-1500) with medium overlap
- **Books/narratives**: Large chunks (2000+) with high overlap
- **FAQs**: Small chunks (300-500) with minimal overlap

### 3. Align with Embedding Model Limits

```python
from crawler.chunker import ChunkingConfig

# For model with 512 token limit (~2048 characters)
config = ChunkingConfig.create(
    chunk_size=1800,  # Leave buffer for special tokens
    overlap=300
)

# For model with 8192 token limit (~32000 characters)
config = ChunkingConfig.create(
    chunk_size=4000,
    overlap=800
)
```

### 4. Preserve Semantic Units

Always enable paragraph preservation for better semantic coherence:

```python
config = ChunkingConfig.create(
    chunk_size=1000,
    preserve_paragraphs=True  # Default, but explicit is better
)
```

### 5. Set Appropriate Minimum Size

```python
# Avoid very small chunks that lack context
config = ChunkingConfig.create(
    chunk_size=1000,
    min_chunk_size=100  # Don't create chunks smaller than 100 chars
)
```

## Troubleshooting

### Chunks Too Large

**Problem**: Chunks exceed embedding model limits

**Solution**:
```python
# Reduce chunk_size
config = ChunkingConfig.create(
    chunk_size=500,  # Smaller chunks
    overlap=100
)
```

### Too Many Chunks

**Problem**: Document splits into excessive number of chunks

**Solution**:
```python
# Increase chunk_size
config = ChunkingConfig.create(
    chunk_size=2000,  # Larger chunks
    overlap=300
)
```

### Poor Context Preservation

**Problem**: Related information split across chunks

**Solution**:
```python
# Increase overlap and preserve paragraphs
config = ChunkingConfig.create(
    chunk_size=1000,
    overlap=400,  # Higher overlap
    preserve_paragraphs=True
)
```

### Uneven Chunk Sizes

**Problem**: Some chunks much smaller than others

**Solution**:
```python
# Adjust min_chunk_size
config = ChunkingConfig.create(
    chunk_size=1000,
    min_chunk_size=200,  # Higher minimum
    preserve_paragraphs=True
)
```

## Dependencies

- `pydantic>=2.0` - Configuration validation
- Parent modules: `crawler.document`

## Related Documentation

- [Document Module](../document/overview.md) - Document data structure
- [LLM Module](../llm/overview.md) - Embeddings for chunks
- [Crawler Configuration](../config/overview.md) - Complete crawler setup
- [Examples](../../../examples/OVERVIEW.md) - Real-world chunking examples

