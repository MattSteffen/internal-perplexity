package api

// ChatCompletionResponse represents the response from the chat completions API.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Choices           []Choice `json:"choices"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	Object            string   `json:"object"`
	Usage             Usage    `json:"usage,omitempty"`
}

// ChatCompletionStreamResponse represents a streamed chunk of a chat completion response.
type ChatCompletionStreamResponse struct {
	ID                string         `json:"id"`
	Choices           []StreamChoice `json:"choices"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
	Object            string         `json:"object"`
}

// Choice represents a single choice in a chat completion response.
type Choice struct {
	FinishReason string      `json:"finish_reason"`
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	Logprobs     *Logprobs   `json:"logprobs,omitempty"`
}

// StreamChoice represents a single choice in a streamed chat completion response.
type StreamChoice struct {
	Delta        ChatCompletionStreamDelta `json:"delta"`
	FinishReason string                    `json:"finish_reason"`
	Index        int                       `json:"index"`
	Logprobs     *Logprobs                 `json:"logprobs,omitempty"`
}

// ChatCompletionStreamMessage represents the delta of a streamed message.
type ChatCompletionStreamDelta struct {
	Content   string                         `json:"content,omitempty"`
	ToolCalls []ChatCompletionStreamToolCall `json:"tool_calls,omitempty"`
	Role      string                         `json:"role,omitempty"`
}

// ChatCompletionStreamToolCall represents a tool call from the assistant.
type ChatCompletionStreamToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Usage represents the token usage for a chat completion request.
type Usage struct {
	CompletionTokens int `json:"completion_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Logprobs represents the log probabilities of tokens.
type Logprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
}

// TokenLogprob represents the log probability of a single token.
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob represents one of the most likely tokens and its log probability.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}
