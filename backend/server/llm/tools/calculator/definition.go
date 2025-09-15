package calculator

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/tools"
)

// Calculator provides mathematical calculation capabilities
type Calculator struct{}

// NewCalculator creates a new calculator tool
func NewCalculator() *Calculator {
	return &Calculator{}
}

// Name returns the tool name
func (c *Calculator) Name() string {
	return "calculator"
}

// Description returns the tool description
func (c *Calculator) Description() string {
	return "Performs mathematical calculations with support for basic arithmetic operations including addition, subtraction, multiplication, division, and parentheses for order of operations."
}

// Schema returns the JSON schema for input validation
func (c *Calculator) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]interface{}{
			"expression": map[string]interface{}{
				"type":        "string",
				"description": "Mathematical expression to evaluate (e.g., '15 + 27 * 3', '2.5 * (10 - 3)')",
			},
		},
		"required": []string{"expression"},
	}
}

// Definition returns the OpenAI tool definition
func (c *Calculator) Definition() *api.ToolDefinition {
	return &api.ToolDefinition{
		Type: "function",
		Function: api.FunctionDefinition{
			Name:        "calculator",
			Description: "Evaluate mathematical expressions with support for basic arithmetic operations. Use this tool when you need to perform calculations, solve math problems, or evaluate numerical expressions. The tool supports addition (+), subtraction (-), multiplication (*), division (/), and parentheses for controlling order of operations. It handles decimal numbers and follows standard mathematical precedence rules.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "The mathematical expression to evaluate. Examples: '15 + 27 * 3', '(10 + 5) * (20 - 15)', '2.5 * (10 - 3)'",
					},
				},
				"required": []string{"expression"},
			},
		},
	}
}

// Execute performs mathematical calculation
func (c *Calculator) Execute(ctx context.Context, input *tools.ToolInput) (*tools.ToolResult, error) {
	expression, ok := input.Data["expression"].(string)
	if !ok {
		return &tools.ToolResult{
			Success: false,
			Error:   "expression field is required and must be a string",
		}, nil
	}

	// Validate expression contains only allowed characters
	if !c.isValidExpression(expression) {
		return &tools.ToolResult{
			Success: false,
			Error:   "expression contains invalid characters. Only numbers, +, -, *, /, (, ), and spaces are allowed",
		}, nil
	}

	result, err := c.evaluateExpression(expression)
	if err != nil {
		return &tools.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("calculation failed: %v", err),
		}, nil
	}

	return &tools.ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"expression": expression,
			"result":     result,
		},
	}, nil
}
