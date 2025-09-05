# Models Package

Provider-specific LLM integrations and shared orchestration types using github.com/sashabaranov/go-openai.

## Directory Structure

### openai/
```
client.go         # OpenAI client implementation using sashabaranov/go-openai
types.go          # OpenAI-specific types and wrappers
```

**Contents:**
- OpenAI client using `github.com/sashabaranov/go-openai`
- `openai.ChatCompletionRequest` and `openai.ChatCompletionResponse` types
- `openai.ChatCompletionMessage` types (system, user, assistant)
- Localhost:11434 (Ollama) fallback configuration
- Token usage tracking from OpenAI responses
- Error handling for both OpenAI API and Ollama

### [provider]/
```
[endpoint].go  # Provider-specific endpoint types
```

**Contents:**
- Provider-specific request/response types
- Authentication handling
- Rate limiting structures
- Provider-specific error types

### shared/
**Contents:**
- `LLMProvider` interface for unified provider access
- `Message` struct for cross-provider compatibility
- `CompletionOptions` for standardized parameters
- `TokenUsage` tracking
- `ModelCapabilities` enum
- Orchestration framework types

**API Exchange Documentation:**
This package documents patterns for exchanging between different LLM providers (OpenAI ↔ Anthropic ↔ etc.) but does not implement the exchanges. The patterns include:

- **Message Format Conversion**: Converting between provider-specific message formats
- **Parameter Mapping**: Translating completion options between providers
- **Error Normalization**: Standardizing error types across providers
- **Capability Detection**: Identifying provider-specific features and limitations
- **Fallback Strategies**: Implementing provider failover and load balancing

## Key Interfaces

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    StreamComplete(ctx context.Context, req *CompletionRequest) (<-chan *StreamChunk, error)
    CountTokens(messages []Message) (int, error)
    GetModelCapabilities() ModelCapabilities
}

type Orchestrator interface {
    ExecuteTask(ctx context.Context, task *Task) (*TaskResult, error)
    RegisterAgent(name string, agent Agent)
    GetAgent(name string) (Agent, error)
}
```

## Usage

```go
// Initialize provider
provider := openai.NewProvider(apiKey)

// Use shared interface
orchestrator := shared.NewOrchestrator(provider)

// Execute task
result, err := orchestrator.ExecuteTask(ctx, task)
```
