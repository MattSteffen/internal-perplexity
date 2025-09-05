# Agent Server Package

This package provides the main HTTP server implementation for the agent server.

## Overview

The server package is responsible for:
- Initializing all dependencies (LLM provider, tools, agents)
- Setting up HTTP routes and middleware
- Managing server lifecycle (start/stop)
- Providing access to internal components for testing

## Usage

### Basic Usage

```go
package main

import (
    "log"
    "internal-perplexity/server/api/server"
)

func main() {
    // Create server with default config
    srv, err := server.NewServer(nil)
    if err != nil {
        log.Fatalf("Failed to create server: %v", err)
    }

    // Start server (blocks until shutdown)
    if err := srv.Start(); err != nil {
        log.Fatalf("Server error: %v", err)
    }
}
```

### Custom Configuration

```go
config := &server.Config{
    Port:         ":9090",
    LLMBaseURL:   "http://localhost:11434/v1",
    LLMModel:     "gpt-oss:20b",
    ReadTimeout:  60 * time.Second,
    WriteTimeout: 60 * time.Second,
    IdleTimeout:  120 * time.Second,
}

srv, err := server.NewServer(config)
```

## Configuration

The `Config` struct supports the following options:

- `Port`: Server port (default: ":8080")
- `LLMBaseURL`: Base URL for LLM provider (default: "http://localhost:11434/v1")
- `LLMModel`: LLM model name (default: "gpt-oss:20b")
- `ReadTimeout`: HTTP read timeout (default: 30s)
- `WriteTimeout`: HTTP write timeout (default: 30s)
- `IdleTimeout`: HTTP idle timeout (default: 60s)

## Available Endpoints

The server automatically sets up the following endpoints:

- `POST /agents/{name}` - Execute main agents
- `POST /sub-agents/{name}` - Execute sub-agents
- `POST /tools/{name}` - Execute tools
- `GET /tools` - List available tools
- `GET /health` - Health check

## Architecture

The server follows a clean architecture pattern:

1. **Server Creation**: `NewServer()` initializes all dependencies
2. **Dependency Injection**: LLM provider, tools, and agents are created and wired together
3. **Route Setup**: HTTP handlers are created and routes are configured
4. **Middleware**: CORS and logging middleware are applied
5. **Lifecycle Management**: Server can be started and stopped gracefully

## Testing

The server provides getter methods for testing:

```go
llm := srv.GetLLMProvider()
registry := srv.GetToolRegistry()
agent := srv.GetPrimaryAgent()
```

## Error Handling

The server handles errors gracefully:
- Failed dependency initialization returns an error
- Server startup errors are propagated
- Graceful shutdown with timeout
- Proper signal handling for SIGINT/SIGTERM
