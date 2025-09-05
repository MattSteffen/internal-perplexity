package calculator

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"unicode"

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
	return "Performs mathematical calculations with support for basic arithmetic operations"
}

// Schema returns the JSON schema for input validation
func (c *Calculator) Schema() *tools.ToolSchema {
	return &tools.ToolSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"expression": map[string]interface{}{
				"type":        "string",
				"description": "Mathematical expression to evaluate (e.g., '15 + 27 * 3', '2.5 * (10 - 3)')",
			},
		},
		Required: []string{"expression"},
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

// isValidExpression checks if the expression contains only allowed characters
func (c *Calculator) isValidExpression(expr string) bool {
	for _, char := range expr {
		if !(unicode.IsDigit(char) || char == '+' || char == '-' || char == '*' || char == '/' ||
			char == '(' || char == ')' || char == '.' || char == ' ' || char == '\t') {
			return false
		}
	}
	return true
}

// evaluateExpression performs the actual calculation using a simple recursive parser
func (c *Calculator) evaluateExpression(expr string) (float64, error) {
	// Remove spaces and handle the expression
	expr = strings.ReplaceAll(expr, " ", "")
	return c.parseExpression(expr)
}

// parseExpression parses and evaluates mathematical expressions
func (c *Calculator) parseExpression(expr string) (float64, error) {
	// Handle parentheses first
	if strings.Contains(expr, "(") {
		return c.parseWithParentheses(expr)
	}

	// Parse addition and subtraction
	if plusIdx := strings.LastIndex(expr, "+"); plusIdx > 0 {
		left := expr[:plusIdx]
		right := expr[plusIdx+1:]
		lval, err := c.parseExpression(left)
		if err != nil {
			return 0, err
		}
		rval, err := c.parseExpression(right)
		if err != nil {
			return 0, err
		}
		return lval + rval, nil
	}

	if minusIdx := strings.LastIndex(expr, "-"); minusIdx > 0 {
		left := expr[:minusIdx]
		right := expr[minusIdx+1:]
		lval, err := c.parseExpression(left)
		if err != nil {
			return 0, err
		}
		rval, err := c.parseExpression(right)
		if err != nil {
			return 0, err
		}
		return lval - rval, nil
	}

	// Parse multiplication and division
	if multIdx := strings.LastIndex(expr, "*"); multIdx > 0 {
		left := expr[:multIdx]
		right := expr[multIdx+1:]
		lval, err := c.parseExpression(left)
		if err != nil {
			return 0, err
		}
		rval, err := c.parseExpression(right)
		if err != nil {
			return 0, err
		}
		return lval * rval, nil
	}

	if divIdx := strings.LastIndex(expr, "/"); divIdx > 0 {
		left := expr[:divIdx]
		right := expr[divIdx+1:]
		lval, err := c.parseExpression(left)
		if err != nil {
			return 0, err
		}
		rval, err := c.parseExpression(right)
		if err != nil {
			return 0, err
		}
		if rval == 0 {
			return 0, fmt.Errorf("division by zero")
		}
		return lval / rval, nil
	}

	// Parse number
	return strconv.ParseFloat(expr, 64)
}

// parseWithParentheses handles expressions with parentheses
func (c *Calculator) parseWithParentheses(expr string) (float64, error) {
	// Find the innermost parentheses
	start := strings.LastIndex(expr, "(")
	if start == -1 {
		return c.parseExpression(expr)
	}

	end := strings.Index(expr[start:], ")")
	if end == -1 {
		return 0, fmt.Errorf("unmatched parentheses")
	}
	end += start

	// Evaluate the expression inside parentheses
	innerExpr := expr[start+1 : end]
	innerResult, err := c.parseExpression(innerExpr)
	if err != nil {
		return 0, err
	}

	// Replace the parentheses with the result
	newExpr := expr[:start] + fmt.Sprintf("%g", innerResult) + expr[end+1:]
	return c.parseExpression(newExpr)
}
