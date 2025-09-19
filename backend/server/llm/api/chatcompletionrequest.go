package api

// ChatCompletionRequest represents the request body for the chat completions API.
type ChatCompletionRequest struct {
	Messages         []ChatCompletionMessage `json:"messages"`
	Model            string                  `json:"model"`
	FrequencyPenalty float64                 `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int          `json:"logit_bias,omitempty"`
	Logprobs         bool                    `json:"logprobs,omitempty"`
	TopLogprobs      int                     `json:"top_logprobs,omitempty"`
	MaxTokens        int                     `json:"max_tokens,omitempty"`
	N                int                     `json:"n,omitempty"`
	PresencePenalty  float64                 `json:"presence_penalty,omitempty"`
	ResponseFormat   *ResponseFormat         `json:"response_format,omitempty"`
	Seed             int                     `json:"seed,omitempty"`
	Stop             any                     `json:"stop,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Temperature      float64                 `json:"temperature,omitempty"`
	TopP             float64                 `json:"top_p,omitempty"`
	Tools            []ToolDefinition        `json:"tools,omitempty"`
	ToolChoice       any                     `json:"tool_choice,omitempty"`
	ParallelTools    bool                    `json:"parallel_tool_calls,omitempty"`
	User             string                  `json:"user,omitempty"`
}
