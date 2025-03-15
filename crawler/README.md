# Document Crawler and Vector Database System

This project is a document crawler that processes files from specified directories, extracts text and metadata, generates embeddings, and stores them in a vector database (Milvus) for semantic search.

## Configuration System

The project uses a flexible, layered configuration system that allows for both global settings and directory-specific overrides.

### Configuration Structure

1. **Base Configuration** (`config/base_config.yaml`):

   - Global settings that apply to the entire application
   - Milvus connection parameters
   - Default embedding settings
   - Processing parameters
   - Logging configuration

2. **Collection Templates** (`config/collection_template.yaml`):

   - Defines the schema and settings for vector database collections
   - Field definitions
   - Index parameters
   - Search configuration

3. **Directory-Specific Configurations** (`config/directories/*.yaml`):
   - Settings for specific data directories
   - Can override base and collection settings
   - Specifies the path to the data directory
   - Custom processing parameters
   - Collection overrides

### How Configuration Works

The configuration system uses a hierarchical approach:

1. Base configuration provides default settings
2. Collection templates define the structure for vector collections
3. Directory-specific configurations can override both base and collection settings

This allows for flexible configuration where you can:

- Use the same embedding model across all directories
- Have different chunk sizes for different types of documents
- Use different collections with custom schemas for different data sources
- Apply specific filters or processing steps to certain directories

### Adding a New Directory

To add a new directory to crawl:

1. Create a new YAML file in `config/directories/` (e.g., `my_new_data.yaml`)
2. Specify the path to the directory and any custom settings
3. Run the crawler with `python main.py --directory my_new_data`

Example directory configuration:

```yaml
# Directory-specific configuration
path: "/path/to/my/data"

# Collection to use
collection: "my_collection"

# Processing overrides
processing:
  chunk_size: 500
  extractors:
    - type: "json"
      enabled: true
```

## Dependencies

- Python 3.8+
- pymilvus
- sentence-transformers
- pyyaml

## Usage

```bash
# Process a specific directory configuration
python main.py --directory conference

# Process all configured directories
python main.py
```
