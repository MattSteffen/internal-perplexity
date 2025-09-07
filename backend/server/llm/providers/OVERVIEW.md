# Models Package

Provider-specific LLM integrations with a unified interface, shared types, and transport utilities. Providers implement the same contract for completions and streaming, with standardized options, messages, usage, errors, and capability flags.

## Architecture Overview

The Models package provides a unified interface for multiple LLM providers (OpenAI, Anthropic, Ollama) with:

- **Unified LLMProvider Interface**: Consistent API across all providers
- **Cross-Provider Compatibility**: Standardized types and error handling
- **Transport Layer**: HTTP clients, rate limiting, and streaming utilities
- **Provider Registry**: Factory pattern for provider instantiation
- **Comprehensive Testing**: Fake provider for unit tests

## Directory Structure

### shared/
```
types.go           # Unified types (Message, CompletionRequest/Response, etc.)
provider.go        # LLMProvider interface + shared utilities
```

**Contents:**
- Unified `LLMProvider` interface for all providers
- Standardized message, request, and response types
- Error normalization and validation utilities
- Cross-provider type definitions

### transport/
```
http.go            # HTTP client with retries and timeouts
rate_limit.go      # Token-bucket rate limiting
streaming.go       # SSE streaming and chunk processing
```

**Contents:**
- Tuned HTTP client with retry logic and connection pooling
- Rate limiting with provider-specific configurations
- Streaming utilities for Server-Sent Events
- Chunk processing and normalization

### registry/
```
registry.go        # Provider registry and factory
```

**Contents:**
- Provider factory pattern implementation
- Registry for managing provider instances
- Configuration-driven provider instantiation

### openai/
```
provider.go        # OpenAI provider implementation
map.go             # OpenAI <-> shared type conversion
```

**Contents:**
- OpenAI-compatible API implementation
- Support for BaseURL override (vLLM, Groq, OpenRouter)
- Tool calling and JSON mode support
- Streaming with normalized chunk format

### anthropic/
```
provider.go        # Anthropic provider (skeleton)
```

**Contents:**
- Anthropic Messages API skeleton
- Placeholder for future implementation
- Same interface as other providers

### ollama/
```
provider.go        # Ollama provider (skeleton)
```

**Contents:**
- Local Ollama API skeleton
- HTTP-based communication with localhost:11434
- Placeholder for future implementation

### test/
```
fake_provider.go   # In-memory fake for unit tests
```

**Contents:**
- Programmable fake provider for testing
- Mock responses and streaming
- Error simulation capabilities

## Key Interfaces

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    StreamComplete(ctx context.Context, req *CompletionRequest) (<-chan *StreamChunk, func(), error)
    CountTokens(messages []Message, model string) (int, error)
    GetModelCapabilities(model string) ModelCapabilities
    Name() string
}

type Message struct {
    Role    Role               `json:"role"`
    Content string             `json:"content,omitempty"`
    ToolCalls      []ToolCall      `json:"tool_calls,omitempty"`
    ToolInvocation *ToolInvocation `json:"tool_invocation,omitempty"`
}

type CompletionRequest struct {
    Messages []Message
    Options  CompletionOptions
    System   string
}
```

## Provider Registry

```go
// Create provider via registry
provider, err := registry.NewProvider(registry.ProviderConfig{
    Name:   "openai",
    APIKey: os.Getenv("OPENAI_API_KEY"),
    // BaseURL can point to OpenAI-compatible endpoints
})

// Or use OpenAI-compatible endpoint
provider, err := registry.NewProvider(registry.ProviderConfig{
    Name:    "openai-compatible",
    APIKey:  apiKey,
    BaseURL: "https://api.groq.com/openai/v1",
})
```

## Usage Examples

### Basic Completion
```go
provider, _ := registry.NewProvider(registry.ProviderConfig{
    Name:   "openai",
    APIKey: os.Getenv("OPENAI_API_KEY"),
})

req := &shared.CompletionRequest{
    Messages: []shared.Message{
        {Role: shared.RoleUser, Content: "Hello, how are you?"},
    },
    Options: shared.CompletionOptions{
        Model:       "gpt-4o-mini",
        Temperature: 0.7,
    },
}

resp, err := provider.Complete(ctx, req)
if err != nil {
    log.Fatal(err)
}
fmt.Println(resp.Content)
```

### Streaming Completion
```go
ch, cancel, err := provider.StreamComplete(ctx, req)
if err != nil {
    log.Fatal(err)
}
defer cancel()

for chunk := range ch {
    if chunk.Done {
        break
    }
    if chunk.DeltaText != "" {
        fmt.Print(chunk.DeltaText)
    }
}
```

### Provider Capabilities
```go
caps := provider.GetModelCapabilities("gpt-4")
fmt.Printf("Supports tools: %v\n", caps.Tools)
fmt.Printf("Max context: %d tokens\n", caps.MaxContextTokens)
```

## Error Handling

All providers normalize errors to `ProviderError`:

```go
resp, err := provider.Complete(ctx, req)
if err != nil {
    if pe, ok := err.(*shared.ProviderError); ok {
        switch pe.Code {
        case shared.ErrRateLimited:
            // Handle rate limiting
        case shared.ErrAuth:
            // Handle authentication error
        case shared.ErrContextLength:
            // Handle context length exceeded
        }
    }
}
```

## Transport Layer

The transport layer provides:

- **HTTP Client**: Connection pooling, retries, timeouts
- **Rate Limiting**: Token-bucket per-provider rate limiting
- **Streaming**: SSE reader with chunk normalization
- **Error Handling**: HTTP status code normalization

```go
httpClient := transport.NewHTTPClient(shared.ClientOptions{
    BaseURL:     "https://api.openai.com/v1",
    APIKey:      apiKey,
    Timeout:     30 * time.Second,
    RetryMax:    3,
    RetryBackoff: time.Second,
})
```

## Testing

Use the fake provider for unit tests:

```go
fake := test.NewFakeProvider()
fake.AddResponse("test prompt", &shared.CompletionResponse{
    Content: "test response",
})

// Use fake as LLMProvider in tests
```
