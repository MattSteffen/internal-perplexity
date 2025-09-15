package llm

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
	"go-crawler/pkg/interfaces"
)

// OllamaLLM implements the LLM interface for Ollama
type OllamaLLM struct {
	config     *config.LLMConfig
	httpClient *http.Client
}

// NewOllamaLLM creates a new Ollama LLM instance
func NewOllamaLLM(cfg *config.LLMConfig) (*OllamaLLM, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("base_url is required for Ollama LLM")
	}
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("model_name is required for Ollama LLM")
	}

	httpClient := &http.Client{
		Timeout: time.Duration(cfg.Timeout) * time.Second,
	}

	return &OllamaLLM{
		config:     cfg,
		httpClient: httpClient,
	}, nil
}

// ollamaRequest represents the request payload for Ollama API
type ollamaRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt,omitempty"`
	Messages []ollamaMessage        `json:"messages,omitempty"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

// ollamaMessage represents a message in Ollama format
type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ollamaResponse represents the response from Ollama API
type ollamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// ollamaErrorResponse represents an error response from Ollama
type ollamaErrorResponse struct {
	Error string `json:"error"`
}

// Invoke sends a prompt to the LLM and returns the response
func (o *OllamaLLM) Invoke(ctx context.Context, prompt string, options *interfaces.LLMOptions) (string, error) {
	messages := []interfaces.LLMMessage{
		{Role: "user", Content: prompt},
	}
	return o.InvokeWithMessages(ctx, messages, options)
}

// InvokeWithMessages sends messages to the LLM and returns the response
func (o *OllamaLLM) InvokeWithMessages(ctx context.Context, messages []interfaces.LLMMessage, options *interfaces.LLMOptions) (string, error) {
	ollamaMessages := make([]ollamaMessage, len(messages))
	for i, msg := range messages {
		ollamaMessages[i] = ollamaMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	reqBody := ollamaRequest{
		Model:    o.config.ModelName,
		Messages: ollamaMessages,
		Stream:   false,
	}

	// Add system prompt if configured
	if o.config.SystemPrompt != nil && *o.config.SystemPrompt != "" {
		systemMessage := ollamaMessage{
			Role:    "system",
			Content: *o.config.SystemPrompt,
		}
		reqBody.Messages = append([]ollamaMessage{systemMessage}, reqBody.Messages...)
	}

	// Apply options
	if options != nil {
		reqBody.Options = o.buildOptions(options)
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	url := fmt.Sprintf("%s/api/chat", strings.TrimSuffix(o.config.BaseURL, "/"))
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	// Try to parse as JSON response
	var ollamaResp ollamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		// Try to parse as error response
		var errorResp ollamaErrorResponse
		if jsonErr := json.Unmarshal(body, &errorResp); jsonErr == nil && errorResp.Error != "" {
			return "", fmt.Errorf("Ollama API error: %s", errorResp.Error)
		}
		return "", fmt.Errorf("failed to parse response: %v, body: %s", err, string(body))
	}

	if ollamaResp.Done && ollamaResp.Response == "" {
		return "", fmt.Errorf("received empty response from Ollama")
	}

	return ollamaResp.Response, nil
}

// GetContextLength returns the maximum context length for this model
func (o *OllamaLLM) GetContextLength() int {
	return o.config.ContextLength
}

// buildOptions converts LLMOptions to Ollama options format
func (o *OllamaLLM) buildOptions(opts *interfaces.LLMOptions) map[string]interface{} {
	options := make(map[string]interface{})

	if opts.Temperature != nil {
		options["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		options["num_predict"] = *opts.MaxTokens
	}
	if opts.TopP != nil {
		options["top_p"] = *opts.TopP
	}
	if opts.TopK != nil {
		options["top_k"] = *opts.TopK
	}
	if len(opts.StopSequences) > 0 {
		options["stop"] = opts.StopSequences
	}

	return options
}

// calculateInputLength estimates the token length of input (rough approximation)
func (o *OllamaLLM) calculateInputLength(prompt string, messages []interfaces.LLMMessage) int {
	totalLength := len(prompt)

	for _, msg := range messages {
		totalLength += len(msg.Content)
	}

	// Rough approximation: 1 token ≈ 4 characters
	return totalLength / 4
}

// VllmLLM implements the LLM interface for VLLM
type VllmLLM struct {
	config     *config.LLMConfig
	httpClient *http.Client
}

// NewVllmLLM creates a new VLLM LLM instance
func NewVllmLLM(cfg *config.LLMConfig) (*VllmLLM, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("base_url is required for VLLM LLM")
	}
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("model_name is required for VLLM LLM")
	}

	httpClient := &http.Client{
		Timeout: time.Duration(cfg.Timeout) * time.Second,
	}

	return &VllmLLM{
		config:     cfg,
		httpClient: httpClient,
	}, nil
}

// vllmRequest represents the request payload for VLLM API
type vllmRequest struct {
	Model       string        `json:"model"`
	Messages    []vllmMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	TopK        int           `json:"top_k,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
	Stream      bool          `json:"stream"`
}

// vllmMessage represents a message in VLLM format
type vllmMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// vllmResponse represents the response from VLLM API
type vllmResponse struct {
	Choices []vllmChoice `json:"choices"`
}

// vllmChoice represents a choice in VLLM response
type vllmChoice struct {
	Message      vllmMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// Invoke sends a prompt to the VLLM LLM and returns the response
func (v *VllmLLM) Invoke(ctx context.Context, prompt string, options *interfaces.LLMOptions) (string, error) {
	messages := []interfaces.LLMMessage{
		{Role: "user", Content: prompt},
	}
	return v.InvokeWithMessages(ctx, messages, options)
}

// InvokeWithMessages sends messages to the VLLM LLM and returns the response
func (v *VllmLLM) InvokeWithMessages(ctx context.Context, messages []interfaces.LLMMessage, options *interfaces.LLMOptions) (string, error) {
	vllmMessages := make([]vllmMessage, len(messages))
	for i, msg := range messages {
		vllmMessages[i] = vllmMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	reqBody := vllmRequest{
		Model:    v.config.ModelName,
		Messages: vllmMessages,
		Stream:   false,
	}

	// Add system prompt if configured
	if v.config.SystemPrompt != nil && *v.config.SystemPrompt != "" {
		systemMessage := vllmMessage{
			Role:    "system",
			Content: *v.config.SystemPrompt,
		}
		reqBody.Messages = append([]vllmMessage{systemMessage}, reqBody.Messages...)
	}

	// Apply options
	if options != nil {
		if options.MaxTokens != nil {
			reqBody.MaxTokens = *options.MaxTokens
		}
		if options.Temperature != nil {
			reqBody.Temperature = *options.Temperature
		}
		if options.TopP != nil {
			reqBody.TopP = *options.TopP
		}
		if options.TopK != nil {
			reqBody.TopK = *options.TopK
		}
		if len(options.StopSequences) > 0 {
			reqBody.Stop = options.StopSequences
		}
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(v.config.BaseURL, "/"))
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if v.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+v.config.APIKey)
	}

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	var vllmResp vllmResponse
	if err := json.Unmarshal(body, &vllmResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v, body: %s", err, string(body))
	}

	if len(vllmResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in VLLM response")
	}

	return vllmResp.Choices[0].Message.Content, nil
}

// GetContextLength returns the maximum context length for this model
func (v *VllmLLM) GetContextLength() int {
	return v.config.ContextLength
}

