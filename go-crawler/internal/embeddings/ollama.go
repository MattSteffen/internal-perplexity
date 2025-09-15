package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go-crawler/internal/config"
)

// OllamaEmbedder implements the Embedder interface for Ollama
type OllamaEmbedder struct {
	config     *config.EmbedderConfig
	httpClient *http.Client
	dimension  int
}

// NewOllamaEmbedder creates a new Ollama embedder instance
func NewOllamaEmbedder(cfg *config.EmbedderConfig) (*OllamaEmbedder, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("base_url is required for Ollama embedder")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required for Ollama embedder")
	}

	httpClient := &http.Client{
		Timeout: 300 * time.Second, // 5 minutes timeout
	}

	embedder := &OllamaEmbedder{
		config:     cfg,
		httpClient: httpClient,
		dimension:  0, // Will be determined dynamically
	}

	// Try to get dimension by making a test embedding
	if cfg.Dimension != nil {
		embedder.dimension = *cfg.Dimension
	} else {
		// Try to determine dimension dynamically
		testEmbedding, err := embedder.Embed(context.Background(), "test")
		if err != nil {
			return nil, fmt.Errorf("failed to determine embedding dimension: %v", err)
		}
		embedder.dimension = len(testEmbedding)
	}

	return embedder, nil
}

// ollamaEmbedRequest represents the request payload for Ollama embeddings
type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// ollamaEmbedResponse represents the response from Ollama embeddings API
type ollamaEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

// Embed generates embeddings for a single text
func (o *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	results, err := o.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return results[0], nil
}

// EmbedBatch generates embeddings for multiple texts
func (o *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	embeddings := make([][]float64, len(texts))

	// Process in batches to avoid overwhelming the API
	batchSize := 10
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		batchEmbeddings, err := o.embedBatchInternal(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to embed batch %d-%d: %v", i, end, err)
		}

		copy(embeddings[i:end], batchEmbeddings)
	}

	return embeddings, nil
}

// embedBatchInternal handles the actual API calls for a batch
func (o *OllamaEmbedder) embedBatchInternal(ctx context.Context, texts []string) ([][]float64, error) {
	embeddings := make([][]float64, len(texts))

	// For now, process sequentially. Could be optimized with goroutines
	for i, text := range texts {
		reqBody := ollamaEmbedRequest{
			Model:  o.config.Model,
			Prompt: text,
		}

		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %v", err)
		}

		url := fmt.Sprintf("%s/api/embeddings", strings.TrimSuffix(o.config.BaseURL, "/"))
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %v", err)
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := o.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to send request: %v", err)
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close() // Always close the body

		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %v", err)
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}

		var ollamaResp ollamaEmbedResponse
		if err := json.Unmarshal(body, &ollamaResp); err != nil {
			return nil, fmt.Errorf("failed to parse response: %v, body: %s", err, string(body))
		}

		if len(ollamaResp.Embedding) == 0 {
			return nil, fmt.Errorf("received empty embedding for text: %s", text[:min(50, len(text))])
		}

		embeddings[i] = ollamaResp.Embedding
	}

	return embeddings, nil
}

// GetDimension returns the dimension of the embeddings
func (o *OllamaEmbedder) GetDimension() int {
	return o.dimension
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// OpenAIEmbedder implements the Embedder interface for OpenAI
type OpenAIEmbedder struct {
	config     *config.EmbedderConfig
	httpClient *http.Client
	dimension  int
}

// NewOpenAIEmbedder creates a new OpenAI embedder instance
func NewOpenAIEmbedder(cfg *config.EmbedderConfig) (*OpenAIEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("api_key is required for OpenAI embedder")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required for OpenAI embedder")
	}

	httpClient := &http.Client{
		Timeout: 300 * time.Second,
	}

	embedder := &OpenAIEmbedder{
		config:     cfg,
		httpClient: httpClient,
		dimension:  0, // Will be determined dynamically
	}

	// Set dimension based on model if known
	switch cfg.Model {
	case "text-embedding-ada-002":
		embedder.dimension = 1536
	case "text-embedding-3-small":
		embedder.dimension = 1536
	case "text-embedding-3-large":
		embedder.dimension = 3072
	default:
		// Try to determine dimension dynamically
		testEmbedding, err := embedder.Embed(context.Background(), "test")
		if err != nil {
			return nil, fmt.Errorf("failed to determine embedding dimension: %v", err)
		}
		embedder.dimension = len(testEmbedding)
	}

	return embedder, nil
}

// openAIEmbedRequest represents the request payload for OpenAI embeddings
type openAIEmbedRequest struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
	User           string   `json:"user,omitempty"`
}

// openAIEmbedResponse represents the response from OpenAI embeddings API
type openAIEmbedResponse struct {
	Object string                `json:"object"`
	Data   []openAIEmbeddingData `json:"data"`
	Usage  openAIUsage           `json:"usage"`
}

// openAIEmbeddingData represents embedding data in OpenAI response
type openAIEmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// openAIUsage represents usage information in OpenAI response
type openAIUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// Embed generates embeddings for a single text
func (o *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	results, err := o.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return results[0], nil
}

// EmbedBatch generates embeddings for multiple texts
func (o *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	reqBody := openAIEmbedRequest{
		Input: texts,
		Model: o.config.Model,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	url := "https://api.openai.com/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.config.APIKey)

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var openAIResp openAIEmbedResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v, body: %s", err, string(body))
	}

	if len(openAIResp.Data) != len(texts) {
		return nil, fmt.Errorf("received %d embeddings but expected %d", len(openAIResp.Data), len(texts))
	}

	embeddings := make([][]float64, len(texts))
	for i, data := range openAIResp.Data {
		embeddings[i] = data.Embedding
	}

	return embeddings, nil
}

// GetDimension returns the dimension of the embeddings
func (o *OpenAIEmbedder) GetDimension() int {
	return o.dimension
}

