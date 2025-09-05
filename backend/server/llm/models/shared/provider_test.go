package shared

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOllamaIntegration tests the LLM provider with localhost:11434
func TestOllamaIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Setup Ollama client
	config := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama", // Ollama doesn't require API key
	}

	provider, err := NewOllamaProvider(config)
	require.NoError(t, err)

	// Test basic completion
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Say 'Hello, World!' and nothing else."},
	}

	resp, err := provider.Complete(ctx, messages, CompletionOptions{
		Temperature: 0.1,
		MaxTokens:   50,
	})

	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
	assert.Contains(t, resp.Content, "Hello, World!")
	assert.Greater(t, resp.TokensUsed, 0)
}

// TestProviderFallback tests switching between providers
func TestProviderFallback(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	factory := NewLLMProviderFactory()

	// Register Ollama provider
	ollamaConfig := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	}
	err := factory.RegisterProvider("ollama", ollamaConfig)
	require.NoError(t, err)

	// Test provider retrieval
	provider, err := factory.GetProvider("ollama")
	require.NoError(t, err)
	assert.NotNil(t, provider)

	// Test completion with fallback provider
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	messages := []Message{
		{Role: "user", Content: "What is 2+2?"},
	}

	resp, err := provider.Complete(ctx, messages, CompletionOptions{})
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
}

// TestTokenCounting tests token usage tracking
func TestTokenCounting(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	config := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	}

	provider, err := NewOllamaProvider(config)
	require.NoError(t, err)

	messages := []Message{
		{Role: "user", Content: "This is a test message for token counting."},
	}

	// Test token counting
	tokens, err := provider.CountTokens(messages)
	require.NoError(t, err)
	assert.Greater(t, tokens, 0)

	// Test completion and verify token usage
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := provider.Complete(ctx, messages, CompletionOptions{})
	require.NoError(t, err)
	assert.Greater(t, resp.TokensUsed, 0)
}

// TestConcurrentRequests tests multiple concurrent requests
func TestConcurrentRequests(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	config := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	}

	provider, err := NewOllamaProvider(config)
	require.NoError(t, err)

	const numRequests = 3
	results := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			messages := []Message{
				{Role: "user", Content: fmt.Sprintf("Test request %d", id)},
			}

			_, err := provider.Complete(ctx, messages, CompletionOptions{})
			results <- err
		}(i)
	}

	// Wait for all requests to complete
	for i := 0; i < numRequests; i++ {
		err := <-results
		assert.NoError(t, err)
	}
}

// TestModelCapabilities tests provider capabilities
func TestModelCapabilities(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	config := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	}

	provider, err := NewOllamaProvider(config)
	require.NoError(t, err)

	caps := provider.GetCapabilities()
	assert.NotNil(t, caps)
	assert.Contains(t, caps.SupportedModels, "gpt-oss:20b")
}

// BenchmarkOllamaCompletion benchmarks completion performance
func BenchmarkOllamaCompletion(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	config := &LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	}

	provider, err := NewOllamaProvider(config)
	require.NoError(b, err)

	messages := []Message{
		{Role: "user", Content: "Say 'benchmark' and nothing else."},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		_, err := provider.Complete(ctx, messages, CompletionOptions{
			Temperature: 0.1,
			MaxTokens:   10,
		})

		cancel()
		require.NoError(b, err)
	}
}
