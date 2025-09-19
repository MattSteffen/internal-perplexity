package shared

import (
	"encoding/json"
	"time"

	"internal-perplexity/server/llm/api"
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

func ToOpenAISchema(name, description string, schema map[string]any) *api.ToolDefinition {
	return &api.ToolDefinition{
		Type: "function",
		Function: api.FunctionDefinition{
			Name:        name,
			Description: description,
			Parameters:  schema,
		},
	}
}
