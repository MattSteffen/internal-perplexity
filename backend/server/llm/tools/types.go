package tools

import (
	"context"
	"time"

	"internal-perplexity/server/llm/providers/shared"
)

// ToolInput represents input data for tool execution
type ToolInput struct {
	Name string                 `json:"name"`
	Data map[string]interface{} `json:"data"`
}

// ToolResult represents the result of tool execution
type ToolResult struct {
	Success bool                   `json:"success"`
	Data    map[string]interface{} `json:"data,omitempty"`
	Error   string                 `json:"error,omitempty"`
	Stats   ToolStats              `json:"stats,omitempty"`
}

// ToolStats tracks tool execution statistics
type ToolStats struct {
	ExecutionTime time.Duration `json:"execution_time"`
	TokensUsed    int           `json:"tokens_used,omitempty"`
}

// ToolSchema defines the JSON schema for tool input/output validation
type ToolSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	Required   []string               `json:"required,omitempty"`
}

// ToolDefinition represents the OpenAI tool definition format
type ToolDefinition struct {
	Type     string              `json:"type"`
	Function *FunctionDefinition `json:"function"`
}

// FunctionDefinition defines the function schema for LLM providers
type FunctionDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// Tool defines the interface that all tools must implement
type Tool interface {
	Name() string
	Description() string
	Schema() *ToolSchema
	Definition() *ToolDefinition
	Execute(ctx context.Context, input *ToolInput, llmProvider shared.LLMProvider) (*ToolResult, error)
}
