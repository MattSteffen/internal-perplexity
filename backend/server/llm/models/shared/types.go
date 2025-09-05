package shared

import "context"

// Message represents a chat message for LLM providers
type Message struct {
	Role    string `json:"role"`    // system, user, assistant
	Content string `json:"content"` // message content
}

// CompletionOptions defines parameters for LLM completion requests
type CompletionOptions struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

// CompletionResponse represents the response from an LLM completion
type CompletionResponse struct {
	Content      string     `json:"content"`
	Role         string     `json:"role,omitempty"`
	FinishReason string     `json:"finish_reason,omitempty"`
	Usage        TokenUsage `json:"usage"`
}

// TokenUsage tracks token consumption for billing and monitoring
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LLMProvider defines the interface for LLM providers
type LLMProvider interface {
	Complete(ctx context.Context, messages []Message, opts CompletionOptions) (*CompletionResponse, error)
	CountTokens(messages []Message) (int, error)
}

// LLMConfig holds configuration for LLM providers
type LLMConfig struct {
	BaseURL     string  `json:"base_url"`
	APIKey      string  `json:"api_key,omitempty"`
	Model       string  `json:"model"`
	Temperature float32 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}
