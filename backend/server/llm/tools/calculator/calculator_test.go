package calculator

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCalculator_isValidExpression(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		expr     string
		expected bool
	}{
		{"valid addition", "1 + 2", true},
		{"valid subtraction", "10 - 5", true},
		{"valid multiplication", "3 * 4", true},
		{"valid division", "8 / 2", true},
		{"valid parentheses", "(1 + 2) * 3", true},
		{"valid decimal", "2.5 + 1.5", true},
		{"valid spaces", " 1 + 2 ", true},
		{"invalid letters", "1 + a", false},
		{"invalid symbols", "1 @ 2", false},
		{"empty string", "", true},
		{"only spaces", "   ", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.isValidExpression(tt.expr)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCalculator_evaluateExpression(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name        string
		expr        string
		expected    float64
		expectError bool
	}{
		{"simple addition", "1+2", 3, false},
		{"simple subtraction", "5-3", 2, false},
		{"simple multiplication", "4*6", 24, false},
		{"simple division", "8/2", 4, false},
		{"decimal addition", "2.5+1.5", 4, false},
		{"parentheses", "(1+2)*3", 9, false},
		{"nested parentheses", "((2+3)*2)", 10, false},
		{"division by zero", "5/0", 0, true},
		{"unmatched parentheses", "(1+2", 0, true},
		{"invalid expression", "abc", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := calc.evaluateExpression(tt.expr)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestCalculator_parseExpression(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name        string
		expr        string
		expected    float64
		expectError bool
	}{
		{"number", "42", 42, false},
		{"decimal", "3.14", 3.14, false},
		{"addition", "1+2", 3, false},
		{"subtraction", "5-3", 2, false},
		{"multiplication", "4*6", 24, false},
		{"division", "8/2", 4, false},
		{"order of operations", "2+3*4", 14, false},
		{"with spaces", "2+3", 5, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := calc.parseExpression(tt.expr)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestCalculator_parseWithParentheses(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name        string
		expr        string
		expected    float64
		expectError bool
	}{
		{"simple parentheses", "(1+2)", 3, false},
		{"parentheses with multiplication", "(2+3)*4", 20, false},
		{"nested parentheses", "((2+3)*2)", 10, false},
		{"unmatched opening", "(1+2", 0, true},
		{"unmatched closing", "1+2)", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := calc.parseWithParentheses(tt.expr)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}
