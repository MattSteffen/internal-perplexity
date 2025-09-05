package openai

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/providers/shared"

	"github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI client with Ollama fallback
type Client struct {
	openaiClient *openai.Client
	ollamaClient *openai.Client
	config       *shared.LLMConfig
}

// NewClient creates a new OpenAI client with Ollama fallback
func NewClient(config *shared.LLMConfig) *Client {
	var openaiClient *openai.Client
	if config.APIKey != "" {
		openaiClient = openai.NewClient(config.APIKey)
	}

	// Create Ollama client with custom config
	ollamaConfig := openai.DefaultConfig("ollama")
	ollamaConfig.BaseURL = "http://localhost:11434/v1"
	ollamaClient := openai.NewClientWithConfig(ollamaConfig)

	return &Client{
		openaiClient: openaiClient,
		ollamaClient: ollamaClient,
		config:       config,
	}
}

// Complete sends a completion request, trying OpenAI first then falling back to Ollama
func (c *Client) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	// Convert messages to OpenAI format
	openaiMessages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Set up request with model from request
	openaiReq := openai.ChatCompletionRequest{
		Model:    req.Model,
		Messages: openaiMessages,
	}

	if req.Options.MaxTokens > 0 {
		openaiReq.MaxTokens = req.Options.MaxTokens
	}
	if req.Options.Temperature > 0 {
		openaiReq.Temperature = req.Options.Temperature
	}
	if req.Options.TopP > 0 {
		openaiReq.TopP = req.Options.TopP
	}
	if req.Options.Stream {
		openaiReq.Stream = req.Options.Stream
	}

	// Try OpenAI first if available and API key provided
	var resp openai.ChatCompletionResponse
	var err error

	if c.openaiClient != nil && req.APIKey != "" {
		// Create client with request-specific API key
		clientWithKey := openai.NewClient(req.APIKey)
		resp, err = clientWithKey.CreateChatCompletion(ctx, openaiReq)
		if err == nil {
			return c.convertResponse(resp), nil
		}
		fmt.Printf("OpenAI request failed, falling back to Ollama: %v\n", err)
	}

	// Fallback to Ollama
	openaiReq.Model = "gpt-oss:20b" // Use gpt-oss:20b for Ollama
	resp, err = c.ollamaClient.CreateChatCompletion(ctx, openaiReq)
	if err != nil {
		return nil, fmt.Errorf("both OpenAI and Ollama failed: %w", err)
	}

	return c.convertResponse(resp), nil
}

// CountTokens estimates token count (simplified implementation)
func (c *Client) CountTokens(messages []shared.Message) (int, error) {
	totalTokens := 0
	for _, msg := range messages {
		// Rough estimation: 1 token per 4 characters
		totalTokens += len(msg.Content) / 4
		// Add tokens for role and formatting
		totalTokens += 4
	}
	return totalTokens, nil
}

// convertResponse converts OpenAI response to shared format
func (c *Client) convertResponse(resp openai.ChatCompletionResponse) *shared.CompletionResponse {
	var content string
	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
	}

	return &shared.CompletionResponse{
		Content: content,
		Role:    "assistant",
		Usage: shared.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

// GetSupportedModels returns the list of supported models for this provider
func (c *Client) GetSupportedModels() []shared.ModelInfo {
	return []shared.ModelInfo{
		{
			Name:        "gpt-4",
			Provider:    shared.ProviderOpenAI,
			MaxTokens:   8192,
			Description: "OpenAI GPT-4 model",
		},
		{
			Name:        "gpt-4-turbo",
			Provider:    shared.ProviderOpenAI,
			MaxTokens:   128000,
			Description: "OpenAI GPT-4 Turbo model",
		},
		{
			Name:        "gpt-3.5-turbo",
			Provider:    shared.ProviderOpenAI,
			MaxTokens:   4096,
			Description: "OpenAI GPT-3.5 Turbo model",
		},
		{
			Name:        "gpt-oss:20b",
			Provider:    shared.ProviderOllama,
			MaxTokens:   4096,
			Description: "Ollama GPT-OSS 20B model (fallback)",
		},
	}
}

// SupportsModel checks if the provider supports the given model
func (c *Client) SupportsModel(model string) bool {
	supportedModels := c.GetSupportedModels()
	for _, m := range supportedModels {
		if m.Name == model {
			return true
		}
	}
	return false
}
