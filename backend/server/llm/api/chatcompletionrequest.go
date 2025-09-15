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
	Stop             interface{}             `json:"stop,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Temperature      float64                 `json:"temperature,omitempty"`
	TopP             float64                 `json:"top_p,omitempty"`
	Tools            []ToolDefinition        `json:"tools,omitempty"`
	ToolChoice       interface{}             `json:"tool_choice,omitempty"`
	ParallelTools    bool                    `json:"parallel_tool_calls,omitempty"`
	User             string                  `json:"user,omitempty"`
}

// ChatCompletionRequestMessage represents a message in the chat completion request.
type ChatCompletionMessage interface{}

// ChatCompletionRequestSystemMessage represents a system message.
type ChatCompletionSystemMessage struct {
	Content string `json:"content"`
	Role    string `json:"role"`
	Name    string `json:"name,omitempty"`
}

// ChatCompletionRequestUserMessage represents a user message.
type ChatCompletionUserMessage struct {
	Content interface{} `json:"content"` // string or []ChatCompletionRequestMessageContentPart
	Role    string      `json:"role"`
	Name    string      `json:"name,omitempty"`
}

// ChatCompletionRequestAssistantMessage represents an assistant message.
type ChatCompletionAssistantMessage struct {
	Content   string     `json:"content,omitempty"`
	Role      string     `json:"role"`
	Name      string     `json:"name,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionRequestToolMessage represents a tool message.
type ChatCompletionToolMessage struct {
	Content    string `json:"content"`
	Role       string `json:"role"`
	ToolCallID string `json:"tool_call_id"`
}

// ChatCompletionMessageContentPart represents a part of a user message's content.
type ChatCompletionMessageContentPart interface{}

// ChatCompletionMessageContentPartText represents a text content part.
type ChatCompletionMessageContentPartText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ChatCompletionMessageContentPartImage represents an image content part.
type ChatCompletionMessageContentPartImage struct {
	Type     string   `json:"type"`
	ImageURL ImageURL `json:"image_url"`
}

// ImageURL represents the URL and detail of an image.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}
