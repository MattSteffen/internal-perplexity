package openai

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/models/shared"

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
func (c *Client) Complete(ctx context.Context, messages []shared.Message, opts shared.CompletionOptions) (*shared.CompletionResponse, error) {
	// Convert messages to OpenAI format
	openaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Set up request
	req := openai.ChatCompletionRequest{
		Model:    c.config.Model,
		Messages: openaiMessages,
	}

	if opts.MaxTokens > 0 {
		req.MaxTokens = opts.MaxTokens
	}
	if opts.Temperature > 0 {
		req.Temperature = opts.Temperature
	}
	if opts.TopP > 0 {
		req.TopP = opts.TopP
	}
	if opts.Stream {
		req.Stream = opts.Stream
	}

	// Try OpenAI first if available
	var resp openai.ChatCompletionResponse
	var err error

	if c.openaiClient != nil {
		resp, err = c.openaiClient.CreateChatCompletion(ctx, req)
		if err == nil {
			return c.convertResponse(resp), nil
		}
		fmt.Printf("OpenAI request failed, falling back to Ollama: %v\n", err)
	}

	// Fallback to Ollama
	req.Model = "gpt-oss:20b" // Use gpt-oss:20b for Ollama
	resp, err = c.ollamaClient.CreateChatCompletion(ctx, req)
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
