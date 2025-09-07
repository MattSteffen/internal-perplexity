package ollama

import (
	"context"

	"internal-perplexity/server/llm/providers/shared"
)

// Config holds Ollama provider configuration
type Config struct {
	BaseURL string
}

// Provider implements the unified LLMProvider interface for Ollama
type Provider struct {
	config Config
}

// NewProvider creates a new Ollama provider
func NewProvider(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}

	return &Provider{
		config: cfg,
	}, nil
}

// Name returns the provider name
func (p *Provider) Name() string { return "ollama" }

// GetModelCapabilities returns capabilities for the specified model
func (p *Provider) GetModelCapabilities(model string) shared.ModelCapabilities {
	return shared.ModelCapabilities{
		Streaming:           true,
		Tools:               false,
		ParallelToolCalls:   false,
		JSONMode:            false,
		SystemMessage:       true,
		Vision:              false,
		SupportsTopK:        true,
		SupportsPresencePen: false,
		SupportsFreqPen:     false,
		MaxContextTokens:    8192,
	}
}

// CountTokens estimates token count for the given messages and model
func (p *Provider) CountTokens(messages []shared.Message, model string) (int, error) {
	// TODO: Implement proper token counting
	// For now, return a rough estimate
	totalTokens := 0
	for _, msg := range messages {
		totalTokens += len(msg.Content) / 4
		totalTokens += 4 // overhead per message
	}
	return totalTokens, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	if err := shared.ValidateCompletionRequest(req); err != nil {
		return nil, err
	}

	// TODO: Implement Ollama /api/chat endpoint call
	return nil, &shared.ProviderError{
		Code:    shared.ErrUnsupportedFeature,
		Message: "Ollama provider not yet implemented",
	}
}

// StreamComplete performs a streaming completion request
func (p *Provider) StreamComplete(ctx context.Context, req *shared.CompletionRequest) (<-chan *shared.StreamChunk, func(), error) {
	if err := shared.ValidateCompletionRequest(req); err != nil {
		return nil, nil, err
	}

	// TODO: Implement Ollama streaming
	return nil, nil, &shared.ProviderError{
		Code:    shared.ErrUnsupportedFeature,
		Message: "Ollama streaming not yet implemented",
	}
}
