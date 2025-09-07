package openai

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/providers/transport"

	"github.com/sashabaranov/go-openai"
)

// Config holds OpenAI provider configuration
type Config struct {
	APIKey  string
	BaseURL string
	OrgID   string
}

// Provider implements the unified LLMProvider interface for OpenAI
type Provider struct {
	client     *openai.Client
	httpClient *transport.HTTPClient
	config     Config
}

// NewProvider creates a new OpenAI provider
func NewProvider(cfg Config) (*Provider, error) {
	openaiConfig := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		openaiConfig.BaseURL = cfg.BaseURL
	}
	if cfg.OrgID != "" {
		openaiConfig.OrgID = cfg.OrgID
	}

	client := openai.NewClientWithConfig(openaiConfig)

	httpOpts := shared.ClientOptions{
		BaseURL:      cfg.BaseURL,
		APIKey:       cfg.APIKey,
		OrgID:        cfg.OrgID,
		Timeout:      0, // Use defaults
		RetryMax:     3,
		RetryBackoff: 0, // Use defaults
	}

	httpClient := transport.NewHTTPClient(httpOpts)

	return &Provider{
		client:     client,
		httpClient: httpClient,
		config:     cfg,
	}, nil
}

// Name returns the provider name
func (p *Provider) Name() string { return "openai" }

// GetModelCapabilities returns capabilities for the specified model
func (p *Provider) GetModelCapabilities(model string) shared.ModelCapabilities {
	// Conservative defaults; could be refined per model name.
	return shared.ModelCapabilities{
		Streaming:           true,
		Tools:               true,
		ParallelToolCalls:   true,
		JSONMode:            true,
		SystemMessage:       true,
		Vision:              false,
		SupportsTopK:        false,
		SupportsPresencePen: true,
		SupportsFreqPen:     true,
		MaxContextTokens:    128000,
	}
}

// CountTokens estimates token count for the given messages and model
func (p *Provider) CountTokens(messages []shared.Message, model string) (int, error) {
	// TODO: Implement proper token counting using tiktoken-go
	// For now, return a rough estimate
	totalTokens := 0
	for _, msg := range messages {
		// Rough estimation: ~4 characters per token
		totalTokens += len(msg.Content) / 4
		// Add tokens for role and formatting
		totalTokens += 4
	}
	return totalTokens, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	if err := shared.ValidateCompletionRequest(req); err != nil {
		return nil, err
	}

	openaiReq, err := ToOpenAIRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	resp, err := p.client.CreateChatCompletion(ctx, *openaiReq)
	if err != nil {
		return nil, NormalizeOpenAIError(err)
	}

	return FromOpenAIResponse(resp), nil
}

// StreamComplete performs a streaming completion request
func (p *Provider) StreamComplete(ctx context.Context, req *shared.CompletionRequest) (<-chan *shared.StreamChunk, func(), error) {
	if err := shared.ValidateCompletionRequest(req); err != nil {
		return nil, nil, err
	}

	openaiReq, err := ToOpenAIStreamRequest(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to convert stream request: %w", err)
	}

	stream, err := p.client.CreateChatCompletionStream(ctx, *openaiReq)
	if err != nil {
		return nil, nil, NormalizeOpenAIError(err)
	}

	ch := make(chan *shared.StreamChunk, 32)
	cancel := func() { _ = stream.Close() }

	go func() {
		defer close(ch)
		defer cancel()

		for {
			resp, err := stream.Recv()
			if err != nil {
				if IsStreamEOF(err) {
					ch <- &shared.StreamChunk{Done: true}
					return
				}
				// Surface as a final chunk with Done for consumers.
				ch <- &shared.StreamChunk{
					Done:        true,
					RawProvider: map[string]any{"error": err.Error()},
				}
				return
			}
			ch <- FromOpenAIStream(resp)
		}
	}()

	return ch, cancel, nil
}

// IsStreamEOF checks if the error indicates end of stream
func IsStreamEOF(err error) bool {
	if err == nil {
		return false
	}
	return err.Error() == "EOF" || err.Error() == "stream closed"
}

// NormalizeOpenAIError converts OpenAI errors to normalized ProviderError
func NormalizeOpenAIError(err error) *shared.ProviderError {
	if err == nil {
		return nil
	}

	// Handle OpenAI-specific errors
	// This is a simplified version - in practice you'd check error types
	return &shared.ProviderError{
		Code:    shared.ErrUnknown,
		Message: err.Error(),
	}
}
