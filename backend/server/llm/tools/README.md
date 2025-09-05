# Tools Package

Deterministic tools with predictable input/output and minimal LLM calls.

## Design Principles

### Tool Characteristics
- **Predictable I/O**: Strict input/output schemas
- **Deterministic**: Same input → same output
- **Efficient**: ≤1 LLM call per execution
- **Workflow**: Strict execution patterns
- **Stateless**: No internal state persistence

### Tool Categories
- **Data Processing**: Format conversion, validation, transformation
- **API Integration**: External service calls with error handling
- **Computation**: Mathematical operations, calculations
- **Content Analysis**: Parsing, extraction, metadata generation

## Directory Structure

### [tool-name]/
```
tool_name.go     # Main tool implementation
schema.go        # JSON schema definitions (optional)
types.go         # Input/output types (optional)
```

## Tool Interface

```go
type Tool interface {
    Name() string
    Description() string
    Schema() *ToolSchema
    Execute(ctx context.Context, input *ToolInput) (*ToolResult, error)
}

type ToolSchema struct {
    Input  json.RawMessage `json:"input"`
    Output json.RawMessage `json:"output"`
}

type ToolResult struct {
    Success bool                   `json:"success"`
    Data    interface{}           `json:"data"`
    Error   string                `json:"error,omitempty"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}
```

## Input Validation

Each tool implements comprehensive input validation with specific error handling for invalid inputs.

### Validation Interface
```go
type ToolValidator interface {
    ValidateInput(input *ToolInput) error
    GetValidationSchema() *ValidationSchema
}

type ValidationSchema struct {
    RequiredFields []string               `json:"required_fields"`
    FieldTypes     map[string]string      `json:"field_types"`
    FieldConstraints map[string]Constraint `json:"field_constraints"`
}

type Constraint struct {
    MinLength *int     `json:"min_length,omitempty"`
    MaxLength *int     `json:"max_length,omitempty"`
    MinValue  *float64 `json:"min_value,omitempty"`
    MaxValue  *float64 `json:"max_value,omitempty"`
    Pattern   *string  `json:"pattern,omitempty"`
    Enum      []string `json:"enum,omitempty"`
}
```

### Tool Input Validation
**Common Validation Rules:**
- **Required Fields**: All specified fields must be present
- **Type Validation**: Fields must match expected types
- **Constraint Validation**: Values must meet specified constraints
- **Schema Compliance**: Input must conform to JSON schema

**Error Types:**
```go
type ToolValidationError struct {
    Code    string      `json:"code"`
    Field   string      `json:"field"`
    Message string      `json:"message"`
    Value   interface{} `json:"value,omitempty"`
    Expected string     `json:"expected,omitempty"`
}
```

### Document Summarizer Validation
```go
func (t *DocumentSummarizerTool) ValidateInput(input *ToolInput) error {
    var errors []ToolValidationError

    // Validate content field
    content, exists := input.Data["content"]
    if !exists {
        errors = append(errors, ToolValidationError{
            Code:    "MISSING_REQUIRED_FIELD",
            Field:   "content",
            Message: "content field is required",
        })
    } else {
        contentStr, ok := content.(string)
        if !ok {
            errors = append(errors, ToolValidationError{
                Code:    "INVALID_FIELD_TYPE",
                Field:   "content",
                Message: "content must be a string",
                Value:   content,
                Expected: "string",
            })
        } else if len(strings.TrimSpace(contentStr)) == 0 {
            errors = append(errors, ToolValidationError{
                Code:    "EMPTY_FIELD",
                Field:   "content",
                Message: "content cannot be empty",
            })
        } else if len(contentStr) > 100000 {
            errors = append(errors, ToolValidationError{
                Code:    "CONTENT_TOO_LONG",
                Field:   "content",
                Message: "content exceeds maximum length",
                Value:   len(contentStr),
                Expected: "≤ 100,000 characters",
            })
        }
    }

    // Validate max_length field
    if maxLength, exists := input.Data["max_length"]; exists {
        if maxLenFloat, ok := maxLength.(float64); ok {
            maxLen := int(maxLenFloat)
            if maxLen <= 0 {
                errors = append(errors, ToolValidationError{
                    Code:    "INVALID_FIELD_VALUE",
                    Field:   "max_length",
                    Message: "max_length must be positive",
                    Value:   maxLen,
                    Expected: "> 0",
                })
            } else if maxLen > 5000 {
                errors = append(errors, ToolValidationError{
                    Code:    "VALUE_TOO_HIGH",
                    Field:   "max_length",
                    Message: "max_length exceeds maximum allowed",
                    Value:   maxLen,
                    Expected: "≤ 5,000",
                })
            }
        } else {
            errors = append(errors, ToolValidationError{
                Code:    "INVALID_FIELD_TYPE",
                Field:   "max_length",
                Message: "max_length must be a number",
                Value:   maxLength,
                Expected: "number",
            })
        }
    }

    if len(errors) > 0 {
        return &ToolValidationErrors{Errors: errors}
    }
    return nil
}
```

### Validation Error Codes
- `MISSING_REQUIRED_FIELD`: Required field is missing
- `INVALID_FIELD_TYPE`: Field has wrong data type
- `INVALID_FIELD_VALUE`: Field value violates constraints
- `EMPTY_FIELD`: Field is empty when it shouldn't be
- `CONTENT_TOO_LONG`: Content exceeds length limits
- `VALUE_TOO_HIGH`: Numeric value exceeds maximum
- `VALUE_TOO_LOW`: Numeric value below minimum
- `INVALID_ENUM_VALUE`: Value not in allowed enum

### Schema-Based Validation
```go
func (t *DocumentSummarizerTool) GetValidationSchema() *ValidationSchema {
    return &ValidationSchema{
        RequiredFields: []string{"content"},
        FieldTypes: map[string]string{
            "content":    "string",
            "max_length": "integer",
        },
        FieldConstraints: map[string]Constraint{
            "content": {
                MinLength: &[]int{1}[0],
                MaxLength: &[]int{100000}[0],
            },
            "max_length": {
                MinValue: &[]float64{1}[0],
                MaxValue: &[]float64{5000}[0],
            },
        },
    }
}
```

### Tool Execution with Validation
```go
func (t *DocumentSummarizerTool) Execute(ctx context.Context, input *ToolInput) (*ToolResult, error) {
    // Validate input first
    if err := t.ValidateInput(input); err != nil {
        return &ToolResult{
            Success: false,
            Error:   err.Error(),
            Metadata: map[string]interface{}{
                "validation_errors": err,
            },
        }, err
    }

    // Proceed with validated input
    content := input.Data["content"].(string)
    maxLength := 200 // default
    if ml, ok := input.Data["max_length"]; ok {
        maxLength = int(ml.(float64))
    }

    // Execute summarization logic...
}
```

## Example Tool Implementation

### Document Summarizer Tool
```go
type SummarizerTool struct{}

func (t *SummarizerTool) Name() string {
    return "document_summarizer"
}

func (t *SummarizerTool) Schema() *ToolSchema {
    return &ToolSchema{
        Input: json.RawMessage(`{
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "max_length": {"type": "integer", "default": 200}
            },
            "required": ["content"]
        }`),
        Output: json.RawMessage(`{
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "word_count": {"type": "integer"},
                "compression_ratio": {"type": "number"}
            }
        }`),
    }
}

func (t *SummarizerTool) Execute(ctx context.Context, input *ToolInput) (*ToolResult, error) {
    // Implementation with ≤1 LLM call
    // Deterministic processing
    // Error handling
}
```

### Calculator Tool
```go
type CalculatorTool struct{}

func (c *CalculatorTool) Name() string {
    return "calculator"
}

func (c *CalculatorTool) Description() string {
    return "Performs basic mathematical calculations"
}

func (c *CalculatorTool) Schema() *ToolSchema {
    return &ToolSchema{
        Input: json.RawMessage(`{
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "examples": ["2 + 3 * 4", "(10 - 5) / 2", "sqrt(16) + pow(2, 3)"]
                }
            },
            "required": ["expression"]
        }`),
        Output: json.RawMessage(`{
            "type": "object",
            "properties": {
                "result": {"type": "number"},
                "expression": {"type": "string"},
                "precision": {"type": "integer"}
            }
        }`),
    }
}

func (c *CalculatorTool) Execute(ctx context.Context, input *ToolInput) (*ToolResult, error) {
    // Validate input first
    if err := c.ValidateInput(input); err != nil {
        return &ToolResult{
            Success: false,
            Error:   err.Error(),
            Metadata: map[string]interface{}{
                "validation_errors": err,
            },
        }, err
    }

    expression := input.Data["expression"].(string)

    // Evaluate the mathematical expression
    result, err := c.evaluateExpression(expression)
    if err != nil {
        return &ToolResult{
            Success: false,
            Error:   fmt.Sprintf("Calculation error: %v", err),
            Metadata: map[string]interface{}{
                "expression": expression,
            },
        }, err
    }

    return &ToolResult{
        Success: true,
        Data: map[string]interface{}{
            "result":     result,
            "expression": expression,
            "precision":  6, // Default precision
        },
        Metadata: map[string]interface{}{
            "expression": expression,
            "evaluated_at": time.Now(),
        },
    }, nil
}

func (c *CalculatorTool) ValidateInput(input *ToolInput) error {
    var errors []ToolValidationError

    // Validate expression field
    expression, exists := input.Data["expression"]
    if !exists {
        errors = append(errors, ToolValidationError{
            Code:    "MISSING_REQUIRED_FIELD",
            Field:   "expression",
            Message: "expression field is required",
        })
    } else {
        exprStr, ok := expression.(string)
        if !ok {
            errors = append(errors, ToolValidationError{
                Code:    "INVALID_FIELD_TYPE",
                Field:   "expression",
                Message: "expression must be a string",
                Value:   expression,
                Expected: "string",
            })
        } else if strings.TrimSpace(exprStr) == "" {
            errors = append(errors, ToolValidationError{
                Code:    "EMPTY_FIELD",
                Field:   "expression",
                Message: "expression cannot be empty",
            })
        } else if len(exprStr) > 1000 {
            errors = append(errors, ToolValidationError{
                Code:    "EXPRESSION_TOO_LONG",
                Field:   "expression",
                Message: "expression exceeds maximum length",
                Value:   len(exprStr),
                Expected: "≤ 1000 characters",
            })
        } else {
            // Validate expression syntax
            if !c.isValidExpression(exprStr) {
                errors = append(errors, ToolValidationError{
                    Code:    "INVALID_EXPRESSION",
                    Field:   "expression",
                    Message: "expression contains invalid syntax",
                    Value:   exprStr,
                })
            }
        }
    }

    if len(errors) > 0 {
        return &ToolValidationErrors{Errors: errors}
    }
    return nil
}

func (c *CalculatorTool) evaluateExpression(expression string) (float64, error) {
    // Use a safe expression evaluator
    // This is a simplified implementation - in practice you'd use a proper math parser
    expression = strings.ReplaceAll(expression, " ", "")

    // Basic arithmetic operations
    if strings.Contains(expression, "+") {
        parts := strings.Split(expression, "+")
        if len(parts) == 2 {
            a, err1 := strconv.ParseFloat(parts[0], 64)
            b, err2 := strconv.ParseFloat(parts[1], 64)
            if err1 == nil && err2 == nil {
                return a + b, nil
            }
        }
    }

    if strings.Contains(expression, "-") {
        parts := strings.Split(expression, "-")
        if len(parts) == 2 {
            a, err1 := strconv.ParseFloat(parts[0], 64)
            b, err2 := strconv.ParseFloat(parts[1], 64)
            if err1 == nil && err2 == nil {
                return a - b, nil
            }
        }
    }

    if strings.Contains(expression, "*") {
        parts := strings.Split(expression, "*")
        if len(parts) == 2 {
            a, err1 := strconv.ParseFloat(parts[0], 64)
            b, err2 := strconv.ParseFloat(parts[1], 64)
            if err1 == nil && err2 == nil {
                return a * b, nil
            }
        }
    }

    if strings.Contains(expression, "/") {
        parts := strings.Split(expression, "/")
        if len(parts) == 2 {
            a, err1 := strconv.ParseFloat(parts[0], 64)
            b, err2 := strconv.ParseFloat(parts[1], 64)
            if err1 == nil && err2 == nil {
                if b == 0 {
                    return 0, fmt.Errorf("division by zero")
                }
                return a / b, nil
            }
        }
    }

    // For more complex expressions, you'd integrate a proper math parser
    return 0, fmt.Errorf("unsupported expression format")
}

func (c *CalculatorTool) isValidExpression(expression string) bool {
    // Basic validation - check for balanced parentheses and valid characters
    parenCount := 0
    for _, char := range expression {
        switch char {
        case '(':
            parenCount++
        case ')':
            parenCount--
            if parenCount < 0 {
                return false
            }
        case '+', '-', '*', '/', '.', ' ':
            // Valid operators and decimal point
        default:
            if !unicode.IsDigit(char) {
                return false
            }
        }
    }
    return parenCount == 0
}
```

### Calculator Tool Directory Structure
```
calculator/
├── calculator.go     # Main calculator tool implementation
├── calculator_test.go # Unit tests for calculator functionality
└── validation.go     # Expression validation logic
```

### Calculator Tool Usage
```go
// Register calculator tool
registry := NewToolRegistry()
calculator := NewCalculatorTool()
registry.Register(calculator)

// Execute calculation
result, err := registry.Execute(ctx, &ToolInput{
    Name: "calculator",
    Data: map[string]interface{}{
        "expression": "15 + 27 * 3",
    },
})

if result.Success {
    data := result.Data.(map[string]interface{})
    fmt.Printf("Result: %v\n", data["result"])
    // Output: Result: 96
}
```

### Calculator Tool Validation Examples
```go
// Valid expressions
"2 + 3"           // Basic addition
"10 - 5"          // Subtraction
"4 * 6"           // Multiplication
"20 / 4"          // Division
"(2 + 3) * 4"     // Parentheses

// Invalid expressions
""                 // Empty expression
"2 + "            // Incomplete expression
"abc"             // Non-numeric characters
"2 / 0"           // Division by zero
"(2 + 3"          // Unbalanced parentheses
```

### Calculator Tool Error Codes
- `MISSING_REQUIRED_FIELD`: Expression field missing
- `INVALID_FIELD_TYPE`: Expression not a string
- `EMPTY_FIELD`: Expression is empty
- `EXPRESSION_TOO_LONG`: Expression exceeds 1000 characters
- `INVALID_EXPRESSION`: Expression contains invalid syntax
- `DIVISION_BY_ZERO`: Attempted division by zero

## Tool Registry

```go
type Registry struct {
    tools map[string]Tool
}

func (r *Registry) Register(tool Tool) {
    r.tools[tool.Name()] = tool
}

func (r *Registry) Get(name string) (Tool, error) {
    tool, exists := r.tools[name]
    if !exists {
        return nil, fmt.Errorf("tool %s not found", name)
    }
    return tool, nil
}

func (r *Registry) List() []string {
    names := make([]string, 0, len(r.tools))
    for name := range r.tools {
        names = append(names, name)
    }
    return names
}
```

## Usage in Agents

```go
// Agent uses tools deterministically
func (a *ResearchAgent) Execute(ctx context.Context, input *AgentInput) (*AgentResult, error) {
    // Get web search tool
    searchTool, _ := a.toolRegistry.Get("web_search")

    // Execute with predictable input
    searchInput := &ToolInput{
        Name: "web_search",
        Data: map[string]interface{}{
            "query": input.Query,
            "max_results": 10,
        },
    }

    result, err := searchTool.Execute(ctx, searchInput)
    if err != nil {
        return nil, err
    }

    // Process deterministic output
    return a.processResults(result.Data)
}
```

## Best Practices

1. **Schema First**: Define JSON schemas before implementation
2. **Error Handling**: Comprehensive error types and messages
3. **Validation**: Input validation before processing
4. **Testing**: Unit tests for all execution paths
5. **Documentation**: Clear usage examples and limitations
6. **Performance**: Optimize for speed and resource usage
