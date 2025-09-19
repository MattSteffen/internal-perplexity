package tools

import (
	"context"
	"fmt"
	"time"

	"internal-perplexity/server/llm/api"
	providershared "internal-perplexity/server/llm/providers/shared"
	toolshared "internal-perplexity/server/llm/tools/shared"
)

// Tool defines the interface that all tools must implement
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Definition() *api.ToolDefinition
	Execute(ctx context.Context, input *toolshared.ToolInput, llmProvider providershared.LLMProvider) (*toolshared.ToolResult, error)
}

// Registry manages tool registration and execution
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry
func (r *Registry) Register(tool Tool) {
	r.tools[tool.Name()] = tool
}

// Get retrieves a tool by name
func (r *Registry) Get(name string) (Tool, error) {
	tool, exists := r.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return tool, nil
}

// List returns all registered tools
func (r *Registry) List() map[string]Tool {
	return r.tools
}

// Execute runs a tool by name with the given input
func (r *Registry) Execute(ctx context.Context, input *toolshared.ToolInput, llmProvider providershared.LLMProvider) (*toolshared.ToolResult, error) {
	tool, err := r.Get(input.Name)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	result, err := tool.Execute(ctx, input, llmProvider)
	if err != nil {
		return nil, err
	}

	result.Stats.ExecutionTime = time.Since(start)
	return result, nil
}
