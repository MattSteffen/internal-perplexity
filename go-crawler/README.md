# Go Crawler

A high-performance document processing and vector database system written in Go. This is a complete rewrite of the Python crawler in Go, providing better performance, type safety, and concurrency.

## Features

- **Multiple Document Converters**: Support for PDF, DOCX, and other formats
- **AI-Powered Metadata Extraction**: Extract structured information using LLMs and JSON schemas
- **Vision Language Model Integration**: Describe images and diagrams within documents
- **Vector Database Support**: Store and search through document chunks with embeddings
- **Modular Architecture**: Easily swap out components for different technologies
- **High Performance**: Native Go performance with goroutines for concurrent processing
- **Type Safety**: Full type safety with Go's type system
- **REST API**: HTTP server for crawler management and monitoring

## Architecture

The crawler follows a modular architecture with clear separation of concerns:

1. **Document Conversion**: Transform various file formats into standardized Markdown
2. **Metadata Extraction**: Use LLMs to extract structured information from documents
3. **Text Chunking**: Split documents into manageable pieces for embedding
4. **Embedding Generation**: Convert text chunks into vector representations
5. **Vector Storage**: Store embeddings and metadata in a searchable database

### Processing Pipeline

```
Raw Document → Converter → Markdown → Extractor → Metadata + Chunks → Embedder → Vectors → Database
```

## Quick Start

### Prerequisites

- Go 1.21+
- Access to a vector database (Milvus recommended)
- LLM API access (Ollama, OpenAI, or VLLM)

### Installation

```bash
git clone <repository-url>
cd go-crawler
go mod download
```

### Basic Usage

```go
package main

import (
    "context"
    "log"

    "go-crawler/pkg/crawler"
    "go-crawler/internal/config"
)

func main() {
    cfg := &config.CrawlerConfig{
        Embeddings: config.EmbedderConfig{
            Provider: "ollama",
            Model:    "all-minilm:v2",
            BaseURL:  "http://localhost:11434",
        },
        LLM: config.LLMConfig{
            ModelName: "llama3.2",
            Provider:  "ollama",
            BaseURL:   "http://localhost:11434",
        },
        Database: config.DatabaseConfig{
            Provider:   "milvus",
            Host:       "localhost",
            Port:       19530,
            Collection: "documents",
        },
        MetadataSchema: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "title": map[string]interface{}{"type": "string"},
                "author": map[string]interface{}{"type": "string"},
            },
            "required": []string{"title"},
        },
    }

    crawler, err := crawler.New(cfg)
    if err != nil {
        log.Fatal(err)
    }

    if err := crawler.Crawl(context.Background(), "/path/to/documents"); err != nil {
        log.Fatal(err)
    }
}
```

## Usage Examples

### CLI Usage

The CLI tool provides commands for common operations:

```bash
# Initialize configuration
./bin/crawler config init

# Validate configuration
./bin/crawler config validate crawler-config.json

# Crawl documents
./bin/crawler crawl /path/to/documents --verbose

# Run benchmarks
./bin/crawler benchmark

# Show statistics
./bin/crawler stats
```

### REST API Usage

The HTTP server provides a REST API for integration:

```bash
# Start the server
./bin/crawler-server

# Health check
curl http://localhost:8080/api/v1/health

# Start crawling
curl -X POST http://localhost:8080/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documents"}'

# Get statistics
curl http://localhost:8080/api/v1/stats

# Run benchmarks
curl -X POST http://localhost:8080/api/v1/benchmark
```

### Docker Usage

Use Docker Compose for complete deployment:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f crawler-api

# Stop services
docker-compose down
```

## Configuration

### Configuration Options

#### Embeddings Configuration

```go
config.EmbedderConfig{
    Provider: "ollama",  // "ollama", "openai", or "vllm"
    Model:    "all-minilm:v2",  // Model name
    BaseURL:  "http://localhost:11434",  // API endpoint
    APIKey:   "ollama",  // API key (if required)
}
```

#### Database Configuration

```go
config.DatabaseConfig{
    Provider:   "milvus",  // Currently only "milvus" supported
    Host:       "localhost",
    Port:       19530,
    Username:   "root",
    Password:   "123456",
    Collection: "documents",  // Collection name
    Partition:  "optional_partition",  // Optional partition
    Recreate:   false,  // Recreate collection if exists
}
```

#### LLM Configuration

```go
config.LLMConfig{
    ModelName:     "llama3.2",
    Provider:      "ollama",
    BaseURL:       "http://localhost:11434",
    SystemPrompt:  "You are a helpful assistant...",  // Optional system prompt
    ContextLength: 32000,  // Context window size
    Timeout:       300.0,  // Request timeout in seconds
}
```

## API Reference

### Main Classes

#### `crawler.Crawler`

The main orchestrator class that manages the entire document processing pipeline.

**Methods:**

- `New(config *config.CrawlerConfig) (*Crawler, error)`: Initialize with configuration
- `Crawl(ctx context.Context, path string) error`: Process files or directories

#### `config.CrawlerConfig`

Configuration struct containing all system settings.

**Key Fields:**

- `Embeddings`: Embedding model configuration
- `LLM`: Language model configuration
- `Database`: Database connection settings
- `Converter`: Document converter settings
- `Extractor`: Metadata extractor settings
- `MetadataSchema`: JSON schema for metadata validation
- `ChunkSize`: Text chunking size
- `TempDir`: Temporary file directory

## Development

### Building

```bash
# Build the CLI tool
go build -o bin/crawler cmd/crawler/main.go

# Build the HTTP server
go build -o bin/crawler-server cmd/server/main.go
```

### Testing

```bash
go test ./...
```

### Running

```bash
# CLI usage
./bin/crawler crawl /path/to/documents

# Start HTTP server
./bin/crawler-server
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

### Building

```bash
# Build all binaries
make build-all

# Build CLI tool only
make build

# Build HTTP server only
make build-server
```

### Testing

```bash
# Run all tests
make test

# Run tests with coverage
make coverage

# Run specific package tests
make test-pkg PKG=./pkg/crawler
```

### Code Quality

```bash
# Format code
make fmt

# Vet code
make vet

# Lint code (requires golangci-lint)
make lint

# Run all quality checks
make check
```

### Docker

```bash
# Build Docker image
make docker-build

# Run with Docker
make docker-run
```

## Project Structure

```
go-crawler/
├── cmd/
│   ├── crawler/          # CLI tool
│   └── server/           # HTTP API server
├── pkg/
│   ├── crawler/          # Main crawler package
│   └── interfaces/       # Core interfaces
├── internal/
│   ├── config/           # Configuration management
│   ├── llm/             # LLM implementations
│   ├── embeddings/      # Embedding implementations
│   ├── storage/         # Database implementations
│   └── processing/      # Document processing (TODO)
├── docs/                # Documentation
├── docker-compose.yml   # Docker deployment
├── Dockerfile          # Docker build
├── Makefile           # Build automation
└── README.md          # This file
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Run `make check` to ensure code quality
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow Go naming conventions
- Add comprehensive tests for new features
- Update documentation for API changes
- Use `make check` before committing
- Follow the existing code style

## Roadmap

### Completed ✅

- [x] Project structure and configuration system
- [x] Core interfaces and type definitions
- [x] Ollama LLM and embedding implementations
- [x] Milvus vector database client
- [x] Main crawler orchestration logic
- [x] CLI tool with comprehensive commands
- [x] REST API server with health checks
- [x] Docker deployment configuration
- [x] Makefile for build automation
- [x] Comprehensive documentation

### In Progress 🚧

- [ ] Document processing components (PyMuPDF, MarkItDown, Docling)
- [ ] Metadata extraction with LLM-driven JSON schema validation
- [ ] Benchmarking and evaluation system

### Planned 📋

- [ ] Additional LLM providers (OpenAI, VLLM)
- [ ] More vector database backends
- [ ] Web UI for crawler management
- [ ] Advanced chunking strategies
- [ ] Plugin system for custom processors
- [ ] Performance monitoring and metrics
- [ ] Authentication and authorization
- [ ] Rate limiting and request throttling

## License

MIT License - see LICENSE file for details.
