package tools

import (
	"context"
	"encoding/json"
	"time"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers/shared"
)

// ToolInput represents input data for tool execution
type ToolInput struct {
	Name     string                 `json:"name"`
	Data     map[string]interface{} `json:"data"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Call     api.ToolCall           `json:"call,omitempty"`
	RawInput json.RawMessage        `json:"raw_input,omitempty"`
}

// ToolResult represents the result of tool execution
type ToolResult struct {
	Success   bool                   `json:"success"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
	Stats     ToolStats              `json:"stats,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Call      api.ToolCall           `json:"call,omitempty"`
	RawOutput json.RawMessage        `json:"raw_output,omitempty"`
}

// ToolStats tracks tool execution statistics
type ToolStats struct {
	ExecutionTime time.Duration `json:"execution_time"`
	TokensUsed    int           `json:"tokens_used,omitempty"`
}

// Tool defines the interface that all tools must implement
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Definition() *api.ToolDefinition
	Execute(ctx context.Context, input *ToolInput, llmProvider shared.LLMProvider) (*ToolResult, error)
}
