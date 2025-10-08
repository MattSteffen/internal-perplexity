# Config Module Overview

This module provides centralized configuration management, validation, and default settings for the crawler system. All configuration models use Pydantic BaseModels from the processing and storage modules for type safety and automatic validation.

## Files in This Module

### `__init__.py`
Exports the public API for the config module. Provides clean imports for:
- Default configurations: `DEFAULT_OLLAMA_EMBEDDINGS`, `DEFAULT_OLLAMA_LLM`, `DEFAULT_OLLAMA_VISION_LLM`, `DEFAULT_MILVUS_CONFIG`, etc.
- Validation: `ConfigValidator`, `ValidationError`
- Loading utilities: `load_config_from_file`, `load_config_from_env`

### `config_defaults.py`
Centralized default configurations for all providers and components.

**Purpose:**
Provides default configurations that can be easily imported and used throughout the crawler system, ensuring consistency and reducing duplication.

**Default Configurations:**

**Ollama Provider Defaults:**
- `DEFAULT_OLLAMA_BASE_URL` - Base URL for Ollama services (http://localhost:11434)
- `DEFAULT_OLLAMA_TIMEOUT` - Default timeout for Ollama API calls (300.0s)
- `DEFAULT_OLLAMA_CTX_LENGTH` - Default context length for models (32000 tokens)

**Model Configurations (using Pydantic factory methods):**
- `DEFAULT_OLLAMA_EMBEDDINGS` - EmbedderConfig for all-minilm:v2
- `DEFAULT_OLLAMA_LLM` - LLMConfig for llama3.2:3b
- `DEFAULT_OLLAMA_VISION_LLM` - LLMConfig for llava:latest (vision model)

**Database Configurations:**
- `DEFAULT_MILVUS_CONFIG` - DatabaseClientConfig for local Milvus instance
  - Collection: "documents"
  - Host: localhost:19530
  - Username: root / Password: Milvus

**Crawler Settings:**
- `DEFAULT_CHUNK_SIZE` - Text chunk size (10000 characters)
- `DEFAULT_TEMP_DIR` - Temporary directory path (tmp/)
- `DEFAULT_BENCHMARK` - Whether to run benchmarks (False)
- `DEFAULT_METADATA_SCHEMA` - Default JSON schema for metadata extraction

**Converter & Extractor:**
- `DEFAULT_CONVERTER_CONFIG` - ConverterConfig using MarkItDown
- `DEFAULT_EXTRACTOR_CONFIG` - ExtractorConfig using basic extraction

**Key Features:**
- All configurations use type-safe Pydantic factory methods
- Ensures consistency across the entire system
- Easy to override with custom values
- Centralized location for all defaults

### `loader.py`
Configuration loading utilities for various sources.

**Functions:**

**`load_config_from_file(config_path: str) -> CrawlerConfig`**
- Loads configuration from a JSON file
- Validates JSON structure and creates Pydantic models
- Raises FileNotFoundError if config file doesn't exist
- Raises json.JSONDecodeError if file is not valid JSON
- Raises ValueError if configuration is invalid

**`load_config_from_env() -> CrawlerConfig`**
- Loads configuration from environment variables
- Uses CRAWLER_ prefix for all environment variables
- Supported variables:
  - CRAWLER_EMBEDDING_MODEL, CRAWLER_EMBEDDING_BASE_URL
  - CRAWLER_LLM_MODEL, CRAWLER_LLM_BASE_URL
  - CRAWLER_VISION_MODEL, CRAWLER_VISION_BASE_URL
  - CRAWLER_DATABASE_HOST, CRAWLER_DATABASE_PORT
  - CRAWLER_DATABASE_USERNAME, CRAWLER_DATABASE_PASSWORD
  - CRAWLER_DATABASE_COLLECTION
  - CRAWLER_CHUNK_SIZE, CRAWLER_TEMP_DIR, CRAWLER_BENCHMARK
  - CRAWLER_LOG_LEVEL
- Provides sensible defaults for all unset variables

**`load_config_from_dict(config_dict: Dict[str, Any]) -> CrawlerConfig`**
- Loads configuration from a dictionary
- Useful for programmatic configuration
- Creates Pydantic models from dictionary data
- Raises ValueError if configuration is invalid

**`save_config_to_file(config: CrawlerConfig, config_path: str) -> None`**
- Saves configuration to a JSON file
- Ensures parent directory exists
- Converts Pydantic models to dictionary format
- Creates human-readable JSON with indentation

**`create_example_config(output_path: str = "example_config.json") -> None`**
- Creates an example configuration file
- Useful for getting started quickly
- Shows all available configuration options
- Creates file with sensible defaults

**Key Features:**
- Multiple configuration sources (file, env, dict)
- Automatic Pydantic validation for all configs
- Support for partial configurations with defaults
- Type-safe configuration objects

### `validator.py`
Comprehensive configuration validation system.

**Pydantic Model:**
- `ValidationResult` - Type-safe result of a validation check
  - Fields: test_name, success, message, details (optional), duration (optional, ≥0)
  - Validates duration is non-negative
  - Provides structured validation results

**Exception:**
- `ValidationError` - Raised when configuration validation fails

**Class: ConfigValidator**
Main validation class that performs comprehensive checks on crawler configuration.

**Methods:**

**`__init__(log_level: str = "INFO")`**
- Initializes the validator
- Sets up logging at specified level
- Prepares for validation tests

**`validate_all(config: CrawlerConfig) -> List[ValidationResult]`**
- Runs all validation tests on the configuration
- Tests performed:
  1. Basic Configuration - validates structure, chunk size, temp directory
  2. LLM Connectivity - tests LLM with a simple prompt
  3. Embedding Model - tests embedding generation and dimension
  4. Vision LLM - tests vision model connectivity
  5. Database Connection - tests database connection and basic operations
  6. Image Describer - tests image description service
  7. Metadata Schema - validates JSON schema structure
  8. Converter Configuration - validates converter type and requirements
  9. Extractor Configuration - validates extractor type and LLM config
- Each test is timed and results are collected
- Continues even if individual tests fail
- Returns list of ValidationResult objects

**Validation Tests (Internal Methods):**

**`_validate_basic_config(config)`**
- Checks required configuration fields exist
- Validates chunk size is positive
- Ensures temp directory exists or creates it
- Returns details about chunk size, temp directory, log level

**`_validate_llm_connectivity(config)`**
- Creates LLM instance from config
- Sends test prompt to verify connectivity
- Checks if LLM responds correctly
- Returns model name, provider, base URL

**`_validate_embedding_connectivity(config)`**
- Creates embedder instance from config
- Generates test embedding
- Verifies embedding dimension matches
- Returns model name, dimension, provider

**`_validate_vision_llm_connectivity(config)`**
- Creates vision LLM instance
- Tests with vision-related prompt
- Verifies vision capability
- Returns vision model details

**`_validate_database_connection(config)`**
- Creates database client
- Tests collection creation
- Tests duplicate checking functionality
- Returns collection name, host, port, provider

**`_validate_image_describer_connectivity(config)`**
- Tests image description service
- Uses structured output to verify capability
- Supports both response_format and tools modes
- Returns vision model and base URL

**`_validate_metadata_schema(config)`**
- Validates JSON schema structure
- Checks required keys (type, properties)
- Tests schema against sample metadata
- Uses jsonschema for validation
- Returns property count and required fields

**`_validate_converter_config(config)`**
- Validates converter type is supported
- Checks vision LLM requirements for converter types
- Verifies markitdown/docling have vision_llm configured
- Returns converter type and vision LLM status

**`_validate_extractor_config(config)`**
- Validates extractor type is supported
- Checks LLM configuration is present
- Verifies metadata schema if provided
- Returns extractor type and LLM status

**`print_summary() -> None`**
- Prints formatted summary of all validation results
- Shows pass/fail status for each test
- Displays test details and metrics
- Provides overall success/failure message
- User-friendly output with emojis

**Key Features:**
- Comprehensive validation of all system components
- Actual connectivity tests (not just config checks)
- Detailed timing information for each test
- Continues testing even if some tests fail
- Structured results using Pydantic models
- Human-readable output with clear error messages

## Design Decisions

### Pydantic for Type Safety

The config module leverages Pydantic BaseModels from processing and storage modules:
- All configuration objects are type-safe
- Automatic validation at creation time
- Clear error messages for invalid configurations
- Easy serialization/deserialization

The `ValidationResult` model uses Pydantic to ensure:
- Test names are always strings
- Success status is boolean
- Duration is non-negative (if provided)
- Details are properly typed dictionaries

### Centralized Defaults

All default configurations are defined in one place (`config_defaults.py`):
- Single source of truth for default values
- Easy to update defaults across the system
- Consistent configuration across all components
- Reduces code duplication

### Multiple Configuration Sources

Support for various configuration sources:
- **Files**: JSON configuration files
- **Environment**: Environment variables with CRAWLER_ prefix
- **Dictionaries**: Programmatic configuration
- **Defaults**: Fallback to sensible defaults

This flexibility allows the system to work in different environments:
- Development: Use defaults or config files
- Production: Use environment variables
- Testing: Use programmatic dictionaries

### Comprehensive Validation

The validator performs real connectivity tests:
- Not just schema validation
- Actual API calls to verify services are running
- Tests with real data (embeddings, prompts)
- Database operations to verify connectivity

This ensures the configuration will actually work when used, not just that it's structurally valid.

### Fail-Safe Validation

Validation continues even if individual tests fail:
- All tests run regardless of failures
- Complete picture of what works and what doesn't
- Each test is independent
- Results are collected for analysis

This helps identify all issues at once rather than one at a time.

## Usage Examples

### Using Default Configurations

```python
from crawler.config import (
    DEFAULT_OLLAMA_LLM,
    DEFAULT_OLLAMA_EMBEDDINGS,
    DEFAULT_MILVUS_CONFIG
)

# Use defaults directly
llm = get_llm(DEFAULT_OLLAMA_LLM)
embedder = get_embedder(DEFAULT_OLLAMA_EMBEDDINGS)
db = get_db(DEFAULT_MILVUS_CONFIG, 384, {})
```

### Loading from File

```python
from crawler.config import load_config_from_file

# Load configuration from JSON file
config = load_config_from_file("config.json")

# All fields are validated automatically
print(f"LLM model: {config.llm.model_name}")
print(f"Database: {config.database.collection}")
```

### Loading from Environment

```python
import os
from crawler.config import load_config_from_env

# Set environment variables
os.environ["CRAWLER_LLM_MODEL"] = "llama3.2:3b"
os.environ["CRAWLER_DATABASE_HOST"] = "milvus.example.com"

# Load from environment
config = load_config_from_env()
```

### Validating Configuration

```python
from crawler.config import ConfigValidator

# Create validator
validator = ConfigValidator(log_level="INFO")

# Run all validation tests
results = validator.validate_all(config)

# Print summary
validator.print_summary()

# Check if all tests passed
all_passed = all(r.success for r in results)
if not all_passed:
    print("Some validation tests failed!")
```

### Creating Example Configuration

```python
from crawler.config.loader import create_example_config

# Create example configuration file
create_example_config("my_config.json")

# Now you can edit the file and load it
config = load_config_from_file("my_config.json")
```

### Saving Configuration

```python
from crawler.config.loader import save_config_to_file

# Save current configuration
save_config_to_file(config, "saved_config.json")
```

## Configuration File Format

Example JSON configuration file:

```json
{
  "embeddings": {
    "model": "all-minilm:v2",
    "base_url": "http://localhost:11434",
    "provider": "ollama"
  },
  "llm": {
    "model": "llama3.2:3b",
    "base_url": "http://localhost:11434",
    "provider": "ollama"
  },
  "vision_llm": {
    "model": "llava:latest",
    "base_url": "http://localhost:11434",
    "provider": "ollama"
  },
  "database": {
    "collection": "documents",
    "host": "localhost",
    "port": 19530,
    "username": "root",
    "password": "Milvus",
    "provider": "milvus"
  },
  "converter": {
    "type": "markitdown"
  },
  "extractor": {
    "type": "basic"
  },
  "chunk_size": 10000,
  "temp_dir": "tmp/",
  "benchmark": false,
  "log_level": "INFO"
}
```

## Environment Variables

All configuration can be set via environment variables:

```bash
# Embedding model
export CRAWLER_EMBEDDING_MODEL="all-minilm:v2"
export CRAWLER_EMBEDDING_BASE_URL="http://localhost:11434"

# LLM model
export CRAWLER_LLM_MODEL="llama3.2:3b"
export CRAWLER_LLM_BASE_URL="http://localhost:11434"

# Vision LLM
export CRAWLER_VISION_MODEL="llava:latest"
export CRAWLER_VISION_BASE_URL="http://localhost:11434"

# Database
export CRAWLER_DATABASE_COLLECTION="documents"
export CRAWLER_DATABASE_HOST="localhost"
export CRAWLER_DATABASE_PORT="19530"
export CRAWLER_DATABASE_USERNAME="root"
export CRAWLER_DATABASE_PASSWORD="Milvus"

# Other settings
export CRAWLER_CHUNK_SIZE="10000"
export CRAWLER_TEMP_DIR="tmp/"
export CRAWLER_BENCHMARK="false"
export CRAWLER_LOG_LEVEL="INFO"
```

## Dependencies

The config module relies on:
- `pydantic>=2.0` - For ValidationResult type safety
- `jsonschema>=4.25.1` - For metadata schema validation
- `httpx` - For HTTP connectivity tests (vLLM)
- `requests>=2.32.5` - For HTTP requests
- Processing module configurations (LLMConfig, EmbedderConfig, etc.)
- Storage module configurations (DatabaseClientConfig)

## Error Handling

The config module provides comprehensive error handling:

**ValidationError**
- Raised when configuration validation fails
- Provides clear error messages
- Includes context about what failed

**FileNotFoundError**
- Raised when config file doesn't exist
- Includes the path that was attempted

**json.JSONDecodeError**
- Raised when config file is not valid JSON
- Includes line and column of error

**ValueError**
- Raised when configuration is structurally invalid
- Raised by Pydantic validators for invalid field values

## Best Practices

1. **Always Validate Configuration**
   - Use ConfigValidator before starting the crawler
   - Ensures all services are accessible
   - Catches configuration errors early

2. **Use Environment Variables in Production**
   - Keep sensitive data out of code
   - Easy to change without code updates
   - Works well with container orchestration

3. **Start with Default Configurations**
   - Use provided defaults for development
   - Override only what you need to change
   - Reduces configuration boilerplate

4. **Create Example Configs**
   - Use `create_example_config()` for new projects
   - Provides template with all options
   - Easy to customize for your needs

5. **Save Working Configurations**
   - Use `save_config_to_file()` to persist configs
   - Version control your config files
   - Document any custom settings

6. **Check Validation Results**
   - Review failed tests carefully
   - Fix connectivity issues before running crawler
   - Use DEBUG log level for troubleshooting

## Testing

The config module is designed to be testable:
- ValidationResult is a Pydantic model with automatic validation
- All functions accept config objects for easy mocking
- Validators return structured results for assertions
- Support for both integration and unit tests

