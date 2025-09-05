# LLM Models - MVP Tasks

## Overview
Implement OpenAI ChatCompletion integration with localhost:11434 (Ollama) fallback for testing, focusing on core functionality needed for agent orchestration.

## Core Tasks

### 1. OpenAI Client Setup (`openai/`)
- [ ] Create `client.go` using github.com/sashabaranov/go-openai
- [ ] Implement OpenAI client wrapper with localhost:11434 fallback
- [ ] Add configuration for API keys and base URLs
- [ ] Implement basic error handling and retries
- [ ] Test connection to localhost:11434 with gpt-oss:20b

```go
type ChatCompletionClient struct {
    client      *openai.Client
    baseURL     string
    apiKey      string
    model       string
    fallbackURL string // localhost:11434
}

func (c *ChatCompletionClient) Complete(ctx context.Context, messages []openai.ChatCompletionMessage) (*openai.ChatCompletionResponse, error) {
    // Try OpenAI first, fallback to localhost:11434
    req := openai.ChatCompletionRequest{
        Model:    c.model,
        Messages: messages,
    }

    resp, err := c.client.CreateChatCompletion(ctx, req)
    if err != nil {
        // Fallback to localhost:11434
        fallbackClient := openai.NewClient("ollama").WithBaseURL(c.fallbackURL)
        resp, err = fallbackClient.CreateChatCompletion(ctx, req)
    }
    return &resp, err
}
```

### 2. Shared Interfaces (`shared/`)
- [ ] Define `LLMProvider` interface for unified access
- [ ] Create `Message` struct for cross-provider compatibility
- [ ] Implement `CompletionOptions` for standardized parameters
- [ ] Add `TokenUsage` tracking structure
- [ ] Document API exchange patterns (OpenAI ↔ Anthropic) without implementation
- [ ] Create basic orchestration types

```go
type LLMProvider interface {
    Complete(ctx context.Context, messages []Message, opts CompletionOptions) (*CompletionResponse, error)
    CountTokens(messages []Message) (int, error)
}

type Message struct {
    Role    string `json:"role"`    // system, user, assistant
    Content string `json:"content"` // message content
}
```

### 3. Provider Abstraction
- [ ] Create provider factory for different backends
- [ ] Implement configuration loading from environment
- [ ] Add health check methods for providers
- [ ] Support multiple models (gpt-4, gpt-oss:20b)

```go
type ProviderFactory struct {
    providers map[string]LLMProvider
}

func (f *ProviderFactory) GetProvider(name string) (LLMProvider, error) {
    // Return OpenAI or Ollama provider
}
```

## Testing Tasks

### 4. Unit Tests
- [ ] Test OpenAI client with mock responses
- [ ] Test localhost:11434 connection and fallback logic
- [ ] Test token counting functionality
- [ ] Test error handling for network failures
- [ ] Test streaming responses (if implemented)

### 5. Integration Tests
- [ ] Test real completion with gpt-oss:20b on localhost:11434
- [ ] Test provider switching between OpenAI and Ollama
- [ ] Test token usage tracking across providers
- [ ] Test concurrent requests to different providers

```go
func TestOllamaIntegration(t *testing.T) {
    client := NewChatCompletionClient(Config{
        BaseURL: "http://localhost:11434/v1",
        Model:   "gpt-oss:20b",
    })

    resp, err := client.Complete(context.Background(), []Message{
        {Role: "user", Content: "Hello, test message"},
    }, CompletionOptions{})

    assert.NoError(t, err)
    assert.NotEmpty(t, resp.Content)
}
```

## Configuration

### 6. Environment Setup
- [ ] Add environment variable support
- [ ] Create config struct for provider settings
- [ ] Add validation for required settings
- [ ] Support both OpenAI API key and Ollama URLs

```yaml
llm:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4"
  ollama:
    base_url: "http://localhost:11434/v1"
    model: "gpt-oss:20b"
    fallback: true
```

## Implementation Priority

### Phase 1: Basic Functionality
1. [ ] Implement OpenAI client with localhost:11434 fallback
2. [ ] Create basic Message and Completion types
3. [ ] Add simple completion method
4. [ ] Test with localhost:11434

### Phase 2: Provider Abstraction
1. [ ] Create LLMProvider interface
2. [ ] Implement provider factory
3. [ ] Add configuration support
4. [ ] Test provider switching

### Phase 3: Advanced Features
1. [ ] Add streaming support
2. [ ] Implement token counting
3. [ ] Add retry logic
4. [ ] Comprehensive error handling

## Success Criteria
- [ ] Can complete prompts using localhost:11434 (gpt-oss:20b)
- [ ] Fallback to OpenAI works when Ollama unavailable
- [ ] Token usage is tracked correctly
- [ ] Error handling is robust
- [ ] Configuration is flexible and well-documented

## Files to Create
- `llm/models/openai/client.go` - OpenAI client using sashabaranov/go-openai
- `llm/models/openai/types.go` - OpenAI-specific wrappers and types
- `llm/models/shared/provider.go` - LLMProvider interface and implementations
- `llm/models/shared/types.go` - Shared types and API exchange documentation
- `llm/models/shared/factory.go` - Provider factory for different backends
- `llm/models/openai/client_test.go` - OpenAI client tests
- `llm/models/shared/provider_test.go` - Provider interface tests
