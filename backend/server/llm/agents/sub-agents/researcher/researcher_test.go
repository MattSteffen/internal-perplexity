package summary

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSummaryAgentIntegration tests with localhost:11434
func TestSummaryAgentIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Setup dependencies
	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	summary := NewSummaryAgent(llmClient)

	// Test summary request
	input := &AgentInput{
		Data: map[string]interface{}{
			"contents": []string{
				"Go is a programming language developed by Google.",
				"It features garbage collection and concurrent execution.",
			},
			"instructions": "Summarize the key features",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := summary.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.NotNil(t, result.Content)

	// Verify result structure
	summaryResult, ok := result.Content.(*SummaryResult)
	require.True(t, ok)
	assert.NotEmpty(t, summaryResult.Summary)
	assert.Greater(t, summaryResult.ContentCount, 0)

	// Verify stats
	assert.Greater(t, result.TokensUsed.Total, 0)
	assert.Greater(t, result.Duration, time.Duration(0))
}

// TestSummaryAgentWithTools tests agent tool integration
func TestSummaryAgentWithTools(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping tool integration test in short mode")
	}

	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	toolRegistry := NewToolRegistry()
	searchTool := NewWebSearchTool()
	summarizerTool := NewDocumentSummarizer(llmClient)

	toolRegistry.Register("web_search", searchTool)
	toolRegistry.Register("document_summarizer", summarizerTool)

	researcher := NewResearcherAgent(llmClient, toolRegistry)

	input := &AgentInput{
		Data: map[string]interface{}{
			"query": "artificial intelligence basics",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := researcher.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)

	// Verify tools were used
	metadata := result.Metadata
	assert.Contains(t, metadata, "tools_used")
	toolsUsed := metadata["tools_used"].([]string)
	assert.Contains(t, toolsUsed, "web_search")
	assert.Contains(t, toolsUsed, "document_summarizer")
}

// TestResearcherAgentErrorHandling tests error scenarios
func TestResearcherAgentErrorHandling(t *testing.T) {
	llmClient := &mockLLMClient{}
	toolRegistry := NewToolRegistry()

	researcher := NewResearcherAgent(llmClient, toolRegistry)

	// Test with missing tools
	input := &AgentInput{
		Data: map[string]interface{}{
			"query": "test query",
		},
	}

	_, err := researcher.Execute(context.Background(), input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tool")
}

// TestResearcherAgentCapabilities tests agent capabilities
func TestResearcherAgentCapabilities(t *testing.T) {
	llmClient := &mockLLMClient{}
	toolRegistry := NewToolRegistry()
	researcher := NewResearcherAgent(llmClient, toolRegistry)

	caps := researcher.GetCapabilities()
	assert.Contains(t, caps, CapabilityResearch)
	assert.Contains(t, caps, CapabilityWebSearch)
	assert.Contains(t, caps, CapabilitySummarization)
}

// TestResearcherAgentStats tests statistics tracking
func TestResearcherAgentStats(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stats test in short mode")
	}

	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	toolRegistry := NewToolRegistry()
	toolRegistry.Register("web_search", NewWebSearchTool())
	toolRegistry.Register("document_summarizer", NewDocumentSummarizer(llmClient))

	researcher := NewResearcherAgent(llmClient, toolRegistry)

	input := &AgentInput{
		Data: map[string]interface{}{
			"query": "machine learning",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	start := time.Now()
	result, err := researcher.Execute(ctx, input)
	duration := time.Since(start)

	require.NoError(t, err)

	// Verify stats are reasonable
	assert.Greater(t, result.TokensUsed.Total, 0)
	assert.Less(t, result.Duration, duration+(time.Second*5)) // Allow some overhead
	assert.True(t, result.Success)
}

// TestResearcherAgentConcurrentExecution tests concurrent agent execution
func TestResearcherAgentConcurrentExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent test in short mode")
	}

	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	toolRegistry := NewToolRegistry()
	toolRegistry.Register("web_search", NewWebSearchTool())
	toolRegistry.Register("document_summarizer", NewDocumentSummarizer(llmClient))

	const numAgents = 2
	results := make(chan error, numAgents)

	for i := 0; i < numAgents; i++ {
		go func(id int) {
			researcher := NewResearcherAgent(llmClient, toolRegistry)
			input := &AgentInput{
				Data: map[string]interface{}{
					"query": fmt.Sprintf("concurrent test query %d", id),
				},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			_, err := researcher.Execute(ctx, input)
			results <- err
		}(i)
	}

	// Wait for all executions to complete
	for i := 0; i < numAgents; i++ {
		err := <-results
		assert.NoError(t, err)
	}
}

// BenchmarkResearcherAgent benchmarks agent performance
func BenchmarkResearcherAgent(b *testing.B) {
	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	toolRegistry := NewToolRegistry()
	toolRegistry.Register("web_search", NewWebSearchTool())
	toolRegistry.Register("document_summarizer", NewDocumentSummarizer(llmClient))

	researcher := NewResearcherAgent(llmClient, toolRegistry)

	input := &AgentInput{
		Data: map[string]interface{}{
			"query": "benchmark query",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)

		_, err := researcher.Execute(ctx, input)
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
		Content:    "Mock response for testing.",
		TokensUsed: 5,
	}, nil
}

func (m *mockLLMClient) CountTokens(messages []Message) (int, error) {
	return 3, nil
}
