package shared

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestProviderTypes tests the provider type constants
func TestProviderTypes(t *testing.T) {
	assert.Equal(t, "openai", string(ProviderOpenAI))
	assert.Equal(t, "anthropic", string(ProviderAnthropic))
	assert.Equal(t, "ollama", string(ProviderOllama))
}

// TestMessageStruct tests the Message struct
func TestMessageStruct(t *testing.T) {
	msg := Message{
		Role:    "user",
		Content: "Hello, World!",
	}

	assert.Equal(t, "user", msg.Role)
	assert.Equal(t, "Hello, World!", msg.Content)
}

// TestCompletionOptions tests the CompletionOptions struct
func TestCompletionOptions(t *testing.T) {
	opts := CompletionOptions{
		MaxTokens:   100,
		Temperature: 0.7,
		TopP:        0.9,
		Stream:      true,
	}

	assert.Equal(t, 100, opts.MaxTokens)
	assert.Equal(t, float32(0.7), opts.Temperature)
	assert.Equal(t, float32(0.9), opts.TopP)
	assert.True(t, opts.Stream)
}

// TestCompletionRequest tests the CompletionRequest struct
func TestCompletionRequest(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "Test message"},
	}

	req := &CompletionRequest{
		Messages: messages,
		Options: CompletionOptions{
			MaxTokens: 50,
		},
		Model:  "gpt-4",
		APIKey: "test-key",
	}

	assert.Len(t, req.Messages, 1)
	assert.Equal(t, "gpt-4", req.Model)
	assert.Equal(t, "test-key", req.APIKey)
	assert.Equal(t, 50, req.Options.MaxTokens)
}
