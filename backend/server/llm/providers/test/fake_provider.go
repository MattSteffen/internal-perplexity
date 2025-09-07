package test

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"internal-perplexity/server/llm/providers/shared"
)

// FakeProvider implements LLMProvider for testing purposes
type FakeProvider struct {
	mu              sync.RWMutex
	responses       map[string]*shared.CompletionResponse
	streamResponses map[string][]*shared.StreamChunk
	delays          map[string]time.Duration
	errors          map[string]error
	callCount       int
	lastRequest     *shared.CompletionRequest
}

// NewFakeProvider creates a new fake provider for testing
func NewFakeProvider() *FakeProvider {
	return &FakeProvider{
		responses:       make(map[string]*shared.CompletionResponse),
		streamResponses: make(map[string][]*shared.StreamChunk),
		delays:          make(map[string]time.Duration),
		errors:          make(map[string]error),
	}
}

// AddResponse adds a canned response for a specific prompt
func (fp *FakeProvider) AddResponse(prompt string, response *shared.CompletionResponse) {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.responses[prompt] = response
}

// AddStreamResponse adds a canned streaming response for a specific prompt
func (fp *FakeProvider) AddStreamResponse(prompt string, chunks []*shared.StreamChunk) {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.streamResponses[prompt] = chunks
}

// AddDelay adds a delay for a specific prompt
func (fp *FakeProvider) AddDelay(prompt string, delay time.Duration) {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.delays[prompt] = delay
}

// AddError adds an error for a specific prompt
func (fp *FakeProvider) AddError(prompt string, err error) {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.errors[prompt] = err
}

// GetCallCount returns the number of calls made to the provider
func (fp *FakeProvider) GetCallCount() int {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	return fp.callCount
}

// GetLastRequest returns the last request made to the provider
func (fp *FakeProvider) GetLastRequest() *shared.CompletionRequest {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	return fp.lastRequest
}

// Name returns the provider name
func (fp *FakeProvider) Name() string { return "fake" }

// GetModelCapabilities returns capabilities for the specified model
func (fp *FakeProvider) GetModelCapabilities(model string) shared.ModelCapabilities {
	return shared.ModelCapabilities{
		Streaming:           true,
		Tools:               true,
		ParallelToolCalls:   true,
		JSONMode:            true,
		SystemMessage:       true,
		Vision:              false,
		SupportsTopK:        true,
		SupportsPresencePen: true,
		SupportsFreqPen:     true,
		MaxContextTokens:    128000,
	}
}

// CountTokens returns a mock token count
func (fp *FakeProvider) CountTokens(messages []shared.Message, model string) (int, error) {
	totalTokens := 0
	for _, msg := range messages {
		totalTokens += len(msg.Content) / 4
		totalTokens += 4 // overhead per message
	}
	return totalTokens, nil
}

// Complete performs a mock completion request
func (fp *FakeProvider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	fp.mu.Lock()
	fp.callCount++
	fp.lastRequest = req
	fp.mu.Unlock()

	// Create a key from the first user message
	var key string
	for _, msg := range req.Messages {
		if msg.Role == shared.RoleUser && msg.Content != "" {
			key = msg.Content
			break
		}
	}

	fp.mu.RLock()
	defer fp.mu.RUnlock()

	// Check for errors first
	if err, exists := fp.errors[key]; exists {
		return nil, err
	}

	// Check for delay
	if delay, exists := fp.delays[key]; exists {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}

	// Return canned response if available
	if response, exists := fp.responses[key]; exists {
		return response, nil
	}

	// Return default mock response
	return &shared.CompletionResponse{
		Content: fmt.Sprintf("Mock response for: %s", key),
		Messages: []shared.Message{
			{
				Role:    shared.RoleAssistant,
				Content: fmt.Sprintf("Mock response for: %s", key),
			},
		},
		Usage: shared.TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
		StopReason: "stop",
	}, nil
}

// StreamComplete performs a mock streaming completion request
func (fp *FakeProvider) StreamComplete(ctx context.Context, req *shared.CompletionRequest) (<-chan *shared.StreamChunk, func(), error) {
	fp.mu.Lock()
	fp.callCount++
	fp.lastRequest = req
	fp.mu.Unlock()

	// Create a key from the first user message
	var key string
	for _, msg := range req.Messages {
		if msg.Role == shared.RoleUser && msg.Content != "" {
			key = msg.Content
			break
		}
	}

	ch := make(chan *shared.StreamChunk, 32)
	cancel := func() {
		close(ch)
	}

	go func() {
		defer close(ch)

		fp.mu.RLock()
		chunks, exists := fp.streamResponses[key]
		fp.mu.RUnlock()

		if exists {
			// Send canned chunks
			for _, chunk := range chunks {
				select {
				case <-ctx.Done():
					return
				case ch <- chunk:
				}
			}
		} else {
			// Send default streaming response
			response := fmt.Sprintf("Mock streaming response for: %s", key)
			words := strings.Fields(response)

			for i, word := range words {
				select {
				case <-ctx.Done():
					return
				case ch <- &shared.StreamChunk{
					DeltaText: word + " ",
					Done:      i == len(words)-1,
				}:
				}

				// Simulate streaming delay
				time.Sleep(50 * time.Millisecond)
			}

			// Send final chunk with usage
			select {
			case <-ctx.Done():
				return
			case ch <- &shared.StreamChunk{
				Done: true,
				Usage: &shared.TokenUsage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
			}:
			}
		}
	}()

	return ch, cancel, nil
}
