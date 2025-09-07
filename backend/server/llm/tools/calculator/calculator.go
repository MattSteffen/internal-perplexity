package calculator

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// isValidExpression checks if the expression contains only allowed characters
func (c *Calculator) isValidExpression(expr string) bool {
	for _, char := range expr {
		if !unicode.IsDigit(char) && char != '+' && char != '-' && char != '*' && char != '/' &&
			char != '(' && char != ')' && char != '.' && char != ' ' && char != '\t' {
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
