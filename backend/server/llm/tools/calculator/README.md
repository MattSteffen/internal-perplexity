# Calculator Tool

The Calculator tool provides mathematical calculation capabilities with support for basic arithmetic operations including addition, subtraction, multiplication, division, and parentheses for order of operations.

## Purpose

The Calculator tool evaluates mathematical expressions safely and deterministically. It supports:
- Basic arithmetic operations (+, -, *, /)
- Decimal numbers
- Parentheses for grouping
- Proper operator precedence
- Division by zero protection
- Input validation

## Features

- **Safe Evaluation**: Validates expressions before evaluation
- **Operator Precedence**: Follows standard mathematical order (parentheses, multiplication/division, addition/subtraction)
- **Decimal Support**: Handles floating-point calculations
- **Error Handling**: Comprehensive error messages for invalid inputs
- **Deterministic**: Same input always produces same output

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `expression` | string | Yes | Mathematical expression to evaluate (e.g., "15 + 27 * 3", "2.5 * (10 - 3)") |

### Expression Format
- **Allowed characters**: Numbers (0-9), decimal points (.), operators (+, -, *, /), parentheses ((), )), spaces
- **Operator precedence**: Parentheses > Multiplication/Division > Addition/Subtraction
- **Maximum length**: No specific limit, but expressions are validated for syntax

## Output Schema

```json
{
  "success": true,
  "data": {
    "expression": "15 + 27 * 3",
    "result": 96
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `expression` | string | The original expression that was evaluated |
| `result` | number | The calculated result as a floating-point number |

## Usage Examples

### Basic Arithmetic

```go
registry := tools.NewRegistry()
calculator := calculator.NewCalculator()
registry.Register(calculator)

result, err := registry.Execute(ctx, &tools.ToolInput{
    Name: "calculator",
    Data: map[string]interface{}{
        "expression": "15 + 27 * 3",
    },
})

if result.Success {
    data := result.Data.(map[string]interface{})
    fmt.Printf("Result: %v\n", data["result"]) // Output: 96
}
```

### API Usage with curl

#### Simple Addition
```bash
curl -X POST http://localhost:8080/tools/calculator \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "expression": "15 + 27 * 3"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "expression": "15 + 27 * 3",
    "result": 96
  },
  "stats": {
    "execution_time": "0.001s"
  }
}
```

#### Complex Expression with Parentheses
```bash
curl -X POST http://localhost:8080/tools/calculator \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "expression": "(10 + 5) * (20 - 15)"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "expression": "(10 + 5) * (20 - 15)",
    "result": 75
  },
  "stats": {
    "execution_time": "0.001s"
  }
}
```

### With Parentheses

```go
input := &tools.ToolInput{
    Name: "calculator",
    Data: map[string]interface{}{
        "expression": "(15 + 27) * 3",
    },
}
// Result: 126
```

### Decimal Calculations

```go
input := &tools.ToolInput{
    Name: "calculator",
    Data: map[string]interface{}{
        "expression": "2.5 * (10 - 3)",
    },
}
// Result: 17.5
```

## Example Inputs and Outputs

### Example 1: Simple Addition
**Input:**
```json
{
  "expression": "15 + 27"
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "expression": "15 + 27",
    "result": 42
  }
}
```

### Example 2: Complex Expression with Parentheses
**Input:**
```json
{
  "expression": "(10 + 5) * (20 - 15)"
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "expression": "(10 + 5) * (20 - 15)",
    "result": 75
  }
}
```

### Example 3: Division with Decimals
**Input:**
```json
{
  "expression": "22.5 / 3"
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "expression": "22.5 / 3",
    "result": 7.5
  }
}
```

## Error Handling

### Invalid Characters
**Input:**
```json
{
  "expression": "15 + abc"
}
```

**Output:**
```json
{
  "success": false,
  "error": "expression contains invalid characters. Only numbers, +, -, *, /, (, ), and spaces are allowed"
}
```

### Division by Zero
**Input:**
```json
{
  "expression": "10 / 0"
}
```

**Output:**
```json
{
  "success": false,
  "error": "calculation failed: division by zero"
}
```

### Unmatched Parentheses
**Input:**
```json
{
  "expression": "(10 + 5"
}
```

**Output:**
```json
{
  "success": false,
  "error": "calculation failed: unmatched parentheses"
}
```

### Missing Required Field
**Input:**
```json
{}
```

**Output:**
```json
{
  "success": false,
  "error": "expression field is required and must be a string"
}
```

## Supported Operations

| Operation | Symbol | Example | Result |
|-----------|--------|---------|--------|
| Addition | + | 5 + 3 | 8 |
| Subtraction | - | 10 - 4 | 6 |
| Multiplication | * | 6 * 7 | 42 |
| Division | / | 15 / 3 | 5 |
| Parentheses | () | (2 + 3) * 4 | 20 |

## Limitations

- **No advanced functions**: No trigonometric, logarithmic, or exponential functions
- **No variables**: Cannot use variables like 'x' or 'y'
- **No scientific notation**: Numbers must be in decimal format
- **Single expressions**: Cannot evaluate multiple expressions at once
- **No complex numbers**: Only real numbers are supported

## Implementation Notes

The calculator uses a recursive parser that:
1. Handles parentheses first (innermost to outermost)
2. Processes multiplication and division (left to right)
3. Processes addition and subtraction (left to right)
4. Supports decimal numbers with proper floating-point arithmetic
