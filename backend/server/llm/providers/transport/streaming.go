package transport

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"internal-perplexity/server/llm/providers/shared"
)

// SSEReader handles Server-Sent Events streaming
type SSEReader struct {
	scanner *bufio.Scanner
}

// NewSSEReader creates a new SSE reader from an HTTP response
func NewSSEReader(resp *http.Response) *SSEReader {
	return &SSEReader{
		scanner: bufio.NewScanner(resp.Body),
	}
}

// ReadEvent reads the next SSE event
func (r *SSEReader) ReadEvent() (string, error) {
	for r.scanner.Scan() {
		line := r.scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return "", io.EOF
			}
			return data, nil
		}
	}
	if err := r.scanner.Err(); err != nil {
		return "", err
	}
	return "", io.EOF
}

// StreamReader provides a generic interface for reading streaming responses
type StreamReader interface {
	ReadChunk() (string, error)
	Close() error
}

// JSONStreamReader reads JSON chunks from a stream
type JSONStreamReader struct {
	reader StreamReader
}

// NewJSONStreamReader creates a new JSON stream reader
func NewJSONStreamReader(reader StreamReader) *JSONStreamReader {
	return &JSONStreamReader{reader: reader}
}

// ReadJSON reads and unmarshals the next JSON chunk
func (r *JSONStreamReader) ReadJSON(v interface{}) error {
	data, err := r.reader.ReadChunk()
	if err != nil {
		return err
	}

	return json.Unmarshal([]byte(data), v)
}

// ChunkProcessor processes streaming chunks and converts them to unified format
type ChunkProcessor struct {
	provider string
}

// NewChunkProcessor creates a new chunk processor for the specified provider
func NewChunkProcessor(provider string) *ChunkProcessor {
	return &ChunkProcessor{provider: provider}
}

// ProcessChunk processes a raw chunk from the provider and converts it to shared.StreamChunk
func (cp *ChunkProcessor) ProcessChunk(rawChunk interface{}) (*shared.StreamChunk, error) {
	switch cp.provider {
	case "openai":
		return cp.processOpenAIChunk(rawChunk)
	case "anthropic":
		return cp.processAnthropicChunk(rawChunk)
	case "ollama":
		return cp.processOllamaChunk(rawChunk)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", cp.provider)
	}
}

// processOpenAIChunk processes an OpenAI streaming chunk
func (cp *ChunkProcessor) processOpenAIChunk(rawChunk interface{}) (*shared.StreamChunk, error) {
	// This would parse OpenAI's streaming format
	// For now, return a basic implementation
	chunk := &shared.StreamChunk{
		RawProvider: rawChunk,
	}

	// Parse the actual OpenAI chunk format here
	// This is a placeholder implementation
	if data, ok := rawChunk.(map[string]interface{}); ok {
		if choices, ok := data["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok {
						chunk.DeltaText = content
					}
				}
				if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
					chunk.Done = true
				}
			}
		}
		if usage, ok := data["usage"].(map[string]interface{}); ok {
			chunk.Usage = &shared.TokenUsage{}
			// Parse usage fields
			_ = usage
		}
	}

	return chunk, nil
}

// processAnthropicChunk processes an Anthropic streaming chunk
func (cp *ChunkProcessor) processAnthropicChunk(rawChunk interface{}) (*shared.StreamChunk, error) {
	// This would parse Anthropic's streaming format
	// Placeholder implementation
	chunk := &shared.StreamChunk{
		RawProvider: rawChunk,
	}
	return chunk, nil
}

// processOllamaChunk processes an Ollama streaming chunk
func (cp *ChunkProcessor) processOllamaChunk(rawChunk interface{}) (*shared.StreamChunk, error) {
	// This would parse Ollama's streaming format
	// Placeholder implementation
	chunk := &shared.StreamChunk{
		RawProvider: rawChunk,
	}
	return chunk, nil
}

// StreamHandler manages streaming responses
type StreamHandler struct {
	ctx    context.Context
	cancel context.CancelFunc
	ch     chan *shared.StreamChunk
	closed bool
}

// NewStreamHandler creates a new stream handler
func NewStreamHandler(ctx context.Context) (*StreamHandler, func()) {
	ctx, cancel := context.WithCancel(ctx)
	handler := &StreamHandler{
		ctx:    ctx,
		cancel: cancel,
		ch:     make(chan *shared.StreamChunk, 32),
		closed: false,
	}

	return handler, func() {
		handler.Close()
	}
}

// SendChunk sends a chunk to the stream
func (h *StreamHandler) SendChunk(chunk *shared.StreamChunk) bool {
	if h.closed {
		return false
	}

	select {
	case h.ch <- chunk:
		return true
	case <-h.ctx.Done():
		return false
	default:
		return false
	}
}

// Channel returns the channel for receiving chunks
func (h *StreamHandler) Channel() <-chan *shared.StreamChunk {
	return h.ch
}

// Close closes the stream handler
func (h *StreamHandler) Close() {
	if !h.closed {
		h.closed = true
		h.cancel()
		close(h.ch)
	}
}

// IsDone checks if the stream is done
func (h *StreamHandler) IsDone() bool {
	return h.closed
}
