# Tools Package

Deterministic tools with predictable input/output and minimal LLM calls.

## Design Principles

### Tool Characteristics

- **Predictable I/O**: Strict input/output schemas
- **Nearly Deterministic**: Same input → same output (sometimes LLMs are called for a single task)
- **Minimal LLM Reliance**: ≤1 LLM call per execution
- **Workflow**: Strict execution patterns
- **Stateless**: No internal state persistence

### Tool Categories

- **Data Processing**: Format conversion, validation, transformation
- **API Integration**: External service calls with error handling
- **Computation**: Mathematical operations, calculations
- **Content Analysis**: Parsing, extraction, metadata generation

## Directory Structure

### [tool-name]/

- tool.go: Main tool implementation
- definition.go: Implementation of the tool interface
- tool_suite_test.go
- tool_test.go # the deterministic parts of the tool calls

## Tool Interface

```go
type Tool interface {
    Name() string
    Description() string
    Schema() *ToolSchema
    ValidateInput(the-input) map[string]error // problems (from gojsonschema validation)
    OpenAISchema() api.ToolDefinition
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
    Stats Stats
}

type Stats struct {}
```

## Input Validation

Each tool implements comprehensive input validation with specific error handling for invalid inputs.

### Tool Input Validation

Done via json schema validation

#### Calculator Tool

```bash
curl -X POST http://localhost:8080/tools/{toolname} \
  -H "Content-Type: application/json" \
  -H "Auth headers" \
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

### Authentication

Some endpoints may require API key authentication:

```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{...}'
```
