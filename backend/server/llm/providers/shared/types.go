package shared

import (
	"context"
	"time"
)

// Role defines the role of a message in a conversation
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a chat message for LLM providers
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content,omitempty"`
	// Optional tool call/invocation metadata for cross-provider parity.
	ToolCalls      []ToolCall      `json:"tool_calls,omitempty"`
	ToolInvocation *ToolInvocation `json:"tool_invocation,omitempty"`
}

// ToolDef defines a tool/function that can be called by the LLM
type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	JSONSchema  map[string]any `json:"json_schema,omitempty"`
}

// ToolCall represents a tool call made by the LLM
type ToolCall struct {
	Name string `json:"name"`
	ID   string `json:"id,omitempty"`
	// Normalized JSON arguments.
	Arguments map[string]any `json:"arguments,omitempty"`
}

// ToolInvocation represents the result of a tool call
type ToolInvocation struct {
	CallID   string            `json:"call_id"`
	Name     string            `json:"name"`
	Result   map[string]any    `json:"result"`
	RawText  string            `json:"raw_text,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ResponseFormat defines the format of the response
type ResponseFormat int

const (
	ResponseFormatText ResponseFormat = iota
	ResponseFormatJSON                // strict JSON mode if provider supports it
)

// CompletionOptions defines parameters for LLM completion requests
type CompletionOptions struct {
	Model            string
	MaxTokens        int
	Temperature      float32
	TopP             float32
	TopK             int
	PresencePenalty  float32
	FrequencyPenalty float32
	Stop             []string
	ResponseFormat   ResponseFormat
	Tools            []ToolDef
	ParallelTools    bool
	// Provider-specific passthrough for future-proofing
	Extra map[string]any
}

// CompletionRequest represents a request to complete
type CompletionRequest struct {
	Messages []Message
	Options  CompletionOptions
	// Optional system prompt when a provider needs top-level system.
	System string
}

// TokenUsage tracks token consumption for billing and monitoring
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// ToolDelta represents partial tool call information during streaming
type ToolDelta struct {
	// Partial tool call information during streaming.
	Call *ToolCall
	// Partial tool invocation result, if provider streams them.
	Invocation *ToolInvocation
}

// StreamChunk represents a chunk of streaming response
type StreamChunk struct {
	DeltaText   string
	ToolDelta   *ToolDelta
	Done        bool
	Usage       *TokenUsage // may only be present on final chunk
	RawProvider any         // optional raw frame for debugging
}

// CompletionResponse represents the response from an LLM completion
type CompletionResponse struct {
	Content     string
	Messages    []Message // full assistant message + tool blocks, if any
	Usage       TokenUsage
	StopReason  string // normalized stop reason (e.g., "stop", "length", "tool")
	RawProvider any    // raw provider response for debugging
}

// ErrorCode defines normalized error codes across providers
type ErrorCode string

const (
	ErrRateLimited        ErrorCode = "rate_limited"
	ErrOverloaded         ErrorCode = "overloaded"
	ErrTimeout            ErrorCode = "timeout"
	ErrAuth               ErrorCode = "auth"
	ErrInvalidRequest     ErrorCode = "invalid_request"
	ErrModelNotFound      ErrorCode = "model_not_found"
	ErrContextLength      ErrorCode = "context_length_exceeded"
	ErrUnavailable        ErrorCode = "service_unavailable"
	ErrUnknown            ErrorCode = "unknown"
	ErrUnsupportedFeature ErrorCode = "unsupported_feature"
)

// ProviderError represents a normalized error from any provider
type ProviderError struct {
	Code    ErrorCode
	Message string
	// Optional: original HTTP status/code and provider payload
	HTTPStatus int
	Raw        any
}

func (e *ProviderError) Error() string { return e.Message }

// ModelCapabilities defines what features a model supports
type ModelCapabilities struct {
	Streaming         bool
	Tools             bool
	ParallelToolCalls bool
	JSONMode          bool
	SystemMessage     bool
	Vision            bool
	// Introspection fields
	SupportsTopK        bool
	SupportsPresencePen bool
	SupportsFreqPen     bool
	MaxContextTokens    int
}

// LLMProvider defines the unified interface for LLM providers
type LLMProvider interface {
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
	StreamComplete(ctx context.Context, req *CompletionRequest) (<-chan *StreamChunk, func(), error)
	CountTokens(messages []Message, model string) (int, error)
	GetModelCapabilities(model string) ModelCapabilities
	Name() string
}

// ProviderType defines the type of LLM provider
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "openai"
	ProviderAnthropic ProviderType = "anthropic"
	ProviderOllama    ProviderType = "ollama"
)

// ModelInfo represents information about a specific model
type ModelInfo struct {
	Name        string       `json:"name"`
	Provider    ProviderType `json:"provider"`
	MaxTokens   int          `json:"max_tokens"`
	Description string       `json:"description,omitempty"`
}

// LLMConfig holds configuration for LLM providers
type LLMConfig struct {
	BaseURL     string  `json:"base_url"`
	APIKey      string  `json:"api_key,omitempty"`
	Model       string  `json:"model"`
	Temperature float32 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

// ClientOptions defines HTTP client configuration
type ClientOptions struct {
	BaseURL      string
	APIKey       string
	OrgID        string
	Headers      map[string]string
	Timeout      time.Duration
	RetryMax     int
	RetryBackoff time.Duration
	MaxIdleConns int
	IdleConnTTL  time.Duration
}
