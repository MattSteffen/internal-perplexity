package document_summarizer

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDocumentSummarizerIntegration tests with localhost:11434
func TestDocumentSummarizerIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Setup LLM client for Ollama
	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	summarizer := NewDocumentSummarizer(llmClient)

	// Test document
	testDoc := `The Go programming language was created by Google engineers Robert Griesemer, Rob Pike, and Ken Thompson. It was announced in November 2009 and version 1.0 was released in March 2012. Go is designed for building simple, reliable, and efficient software. It features garbage collection, concurrent programming support, and a rich standard library. Go is particularly well-suited for building web servers, data pipelines, and command-line tools.`

	input := &ToolInput{
		Name: "document_summarizer",
		Data: map[string]interface{}{
			"content":    testDoc,
			"max_length": 50,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := summarizer.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.NotEmpty(t, result.Data)

	// Verify summary structure
	summaryData, ok := result.Data.(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, summaryData, "summary")

	summary, ok := summaryData["summary"].(string)
	require.True(t, ok)
	assert.NotEmpty(t, summary)
	assert.LessOrEqual(t, len(summary), 100) // Should be reasonably short
}

// TestSummarizerInputValidation tests input validation
func TestSummarizerInputValidation(t *testing.T) {
	llmClient := &mockLLMClient{}
	summarizer := NewDocumentSummarizer(llmClient)

	// Test missing content
	input := &ToolInput{
		Data: map[string]interface{}{
			"max_length": 50,
		},
	}

	_, err := summarizer.Execute(context.Background(), input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "content")

	// Test invalid max_length
	input = &ToolInput{
		Data: map[string]interface{}{
			"content":    "test content",
			"max_length": "invalid",
		},
	}

	_, err = summarizer.Execute(context.Background(), input)
	assert.Error(t, err)
}

// TestSummarizerDeterministic tests that tool is deterministic
func TestSummarizerDeterministic(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping deterministic test in short mode")
	}

	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL:     "http://localhost:11434/v1",
		Model:       "gpt-oss:20b",
		APIKey:      "ollama",
		Temperature: 0.1, // Low temperature for consistency
	})

	summarizer := NewDocumentSummarizer(llmClient)

	testDoc := "This is a simple test document for deterministic summarization."

	input := &ToolInput{
		Data: map[string]interface{}{
			"content":    testDoc,
			"max_length": 30,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Run multiple times
	results := make([]string, 3)
	for i := 0; i < 3; i++ {
		result, err := summarizer.Execute(ctx, input)
		require.NoError(t, err)
		require.True(t, result.Success)

		summaryData := result.Data.(map[string]interface{})
		results[i] = summaryData["summary"].(string)
	}

	// All results should be identical (deterministic)
	assert.Equal(t, results[0], results[1])
	assert.Equal(t, results[1], results[2])
}

// TestSummarizerWithDifferentLengths tests different summary lengths
func TestSummarizerWithDifferentLengths(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping length test in short mode")
	}

	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	summarizer := NewDocumentSummarizer(llmClient)

	testDoc := `Go is a statically typed, compiled programming language designed at Google. It is syntactically similar to C, but with memory safety, garbage collection, structural typing, and CSP-style concurrency. The language was announced in November 2009 and the first stable release was in March 2012. Go's designers were primarily motivated by their shared dislike of C++.

Go is a procedural language with some functional programming features. It supports concurrent programming with goroutines and channels. Go has a rich standard library and a powerful toolchain. It is used for web development, cloud services, DevOps tools, and system programming.`

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test different max lengths
	lengths := []int{20, 50, 100}

	for _, maxLength := range lengths {
		input := &ToolInput{
			Data: map[string]interface{}{
				"content":    testDoc,
				"max_length": maxLength,
			},
		}

		result, err := summarizer.Execute(ctx, input)
		require.NoError(t, err)
		require.True(t, result.Success)

		summaryData := result.Data.(map[string]interface{})
		summary := summaryData["summary"].(string)

		// Summary should be reasonably close to requested length
		assert.LessOrEqual(t, len(summary), maxLength*2) // Allow some flexibility
		assert.Greater(t, len(summary), 10)              // Should not be too short
	}
}

// TestSummarizerSchemaValidation tests JSON schema validation
func TestSummarizerSchemaValidation(t *testing.T) {
	summarizer := NewDocumentSummarizer(&mockLLMClient{})

	schema := summarizer.Schema()
	assert.NotNil(t, schema)

	// Verify schema has required fields
	assert.Contains(t, schema, "input")
	assert.Contains(t, schema, "output")

	// Test schema structure
	inputSchema := schema["input"]
	assert.Equal(t, "object", inputSchema.(map[string]interface{})["type"])

	outputSchema := schema["output"]
	assert.Equal(t, "object", outputSchema.(map[string]interface{})["type"])
}

// BenchmarkDocumentSummarizer benchmarks summarization performance
func BenchmarkDocumentSummarizer(b *testing.B) {
	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	summarizer := NewDocumentSummarizer(llmClient)

	testDoc := "This is a benchmark test document for performance measurement."

	input := &ToolInput{
		Data: map[string]interface{}{
			"content":    testDoc,
			"max_length": 30,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		_, err := summarizer.Execute(ctx, input)
		cancel()

		if err != nil {
			b.Fatal(err)
		}
	}
}

// mockLLMClient is a mock implementation for testing
type mockLLMClient struct{}

func (m *mockLLMClient) Complete(ctx context.Context, messages []Message, opts CompletionOptions) (*CompletionResponse, error) {
	return &CompletionResponse{
		Content:    "This is a mock summary response.",
		TokensUsed: 10,
	}, nil
}

func (m *mockLLMClient) CountTokens(messages []Message) (int, error) {
	return 5, nil
}
