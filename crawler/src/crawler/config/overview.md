# Config Package Overview

This package contains the main `CrawlerConfig` class that orchestrates all crawler subsystem configurations.

## Files

### `crawler_config.py`
Contains the `CrawlerConfig` Pydantic model that serves as the central configuration orchestrator for the entire crawler system. This class:

- Aggregates all subsystem configurations (embeddings, LLMs, database, converter, extractor, chunking)
- Provides type-safe configuration management with Pydantic validation
- Supports serialization/deserialization for storage in Milvus collection descriptions
- Includes factory methods for creating configs from dictionaries and collection descriptions

### `__init__.py`
Package initialization file that exports `CrawlerConfig` for convenient imports.

## Usage

### Basic Import

```python
from crawler.config import CrawlerConfig
# Or from the main package:
from crawler import CrawlerConfig
```

### Creating a Config

```python
from crawler.config import CrawlerConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig
from crawler.converter import ConverterConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.chunker import ChunkingConfig

config = CrawlerConfig.create(
    name="my_crawler",
    embeddings=EmbedderConfig.ollama(model="nomic-embed-text"),
    llm=LLMConfig.ollama(model_name="llama3.2:3b"),
    vision_llm=LLMConfig.ollama(model_name="llava:latest"),
    database=DatabaseClientConfig.milvus(collection="docs"),
    converter=ConverterConfig(...),
    extractor=MetadataExtractorConfig(...),
    chunking=ChunkingConfig.create(chunk_size=1000),
)
```

### Loading from Dictionary

```python
config_dict = {
    "name": "my_crawler",
    "embeddings": {"provider": "ollama", "model": "nomic-embed-text", ...},
    "llm": {"model_name": "llama3.2:3b", ...},
    # ... other configs
}
config = CrawlerConfig.from_dict(config_dict)
```

### Loading from Collection Description

```python
from crawler.vector_db import CollectionDescription, DatabaseClientConfig

# Get collection description from Milvus
description = CollectionDescription.from_json(collection_description_json)
database_config = DatabaseClientConfig.milvus(collection="my_collection")

# Restore config from collection
config = CrawlerConfig.from_collection_description(description, database_config)
```

## Design Decisions

- **Centralized Orchestration**: `CrawlerConfig` is the single source of truth for all crawler settings
- **Sub-configs Stay in Modules**: Individual config classes (`EmbedderConfig`, `LLMConfig`, etc.) remain in their respective modules to maintain modularity
- **Milvus Storage**: The full config is stored in `CollectionDescription.collection_config_json` for persistence and restoration
- **Type Safety**: All configs use Pydantic models for validation and type checking

## Related Modules

- `crawler.main`: Uses `CrawlerConfig` to initialize the `Crawler` class
- `crawler.vector_db`: Stores and restores configs from Milvus collection descriptions
- Individual subsystem modules: Provide their respective config classes that are aggregated by `CrawlerConfig`

