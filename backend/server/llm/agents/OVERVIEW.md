# Agents Package

Agent implementations following specialization patterns and orchestration best practices.

## Architecture

### Agent Types
- **Main Agents**: Primary orchestrators with sub-agent and tool access
- **Sub-Agents**: Specialized agents with tool access, no sub-agents

### Design Patterns
- **Specialization**: Capability, domain, and model-based specialization
- **Composition**: Main agents compose sub-agents
- **Stateless**: All agents except main agent are stateless
- **Monitoring**: All agents return execution stats

## Directory Structure

### main-agents/
#### primary/
**Contents:**
- `primary.go` - Main orchestrator agent
- Access to all sub-agents and general tools
- State management for complex workflows
- Task decomposition and distribution
- Result aggregation and synthesis

**Key Features:**
- Sequential pipeline execution
- MapReduce pattern support
- Parallel sub-agent execution
- Caching integration
- Monitoring and metrics collection

### sub-agents/
#### summary/
**Contents:**
- `summary.go` - Content summarization agent
- Takes list of text contents and instructions
- Returns structured summary based on focus areas
- Single LLM call for summarization

**Key Features:**
- Configurable focus areas and instructions
- Structured output with key points
- Length control and formatting options
- Metadata about summarization process

## Agent Interface

```go
type Agent interface {
    Execute(ctx context.Context, input *AgentInput) (*AgentResult, error)
    GetCapabilities() []Capability
    GetStats() AgentStats
}

type AgentResult struct {
    Content     interface{}
    Success     bool
    TokensUsed  TokenUsage
    Duration    time.Duration
    Metadata    map[string]interface{}
}
```

## Input Validation

Each agent implements comprehensive input validation with specific error types for invalid inputs.

### Validation Interface
```go
type InputValidator interface {
    Validate(input *AgentInput) error
    GetValidationErrors() []ValidationError
}

type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
    Code    string `json:"code"`
    Value   interface{} `json:"value,omitempty"`
}
```

### Primary Agent Validation
**Required Fields:**
- `task`: Must be a non-empty string
- Valid task types: `"summarize_documents"`

**Optional Fields:**
- `timeout`: Must be positive duration if provided
- `context`: Free-form map, no validation

**Error Examples:**
```go
// Missing required task
input := &AgentInput{Data: map[string]interface{}{}}
err := primaryAgent.Validate(input)
// Returns: ValidationError{Code: "MISSING_REQUIRED_FIELD", Field: "task"}

// Invalid task type
input := &AgentInput{Data: map[string]interface{}{"task": "invalid_task"}}
err := primaryAgent.Validate(input)
// Returns: ValidationError{Code: "INVALID_TASK_TYPE", Field: "task", Value: "invalid_task"}
```

### Summary Sub-Agent Validation
**Required Fields:**
- `contents`: Must be non-empty array of strings
- `instructions`: Must be non-empty string

**Optional Fields:**
- `focus_areas`: Array of strings (validated for duplicates)
- `max_length`: Positive integer if provided

**Content Validation:**
```go
// Empty contents array
input := &AgentInput{Data: map[string]interface{}{
    "contents": []string{},
    "instructions": "Summarize this",
}}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "EMPTY_CONTENTS", Field: "contents"}

// Invalid content type
input := &AgentInput{Data: map[string]interface{}{
    "contents": "not an array",
    "instructions": "Summarize this",
}}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "INVALID_CONTENT_TYPE", Field: "contents", Value: "string"}
```

### Validation Error Codes
- `MISSING_REQUIRED_FIELD`: Required field is missing or empty
- `INVALID_FIELD_TYPE`: Field has wrong data type
- `INVALID_FIELD_VALUE`: Field value is invalid
- `EMPTY_CONTENTS`: Contents array is empty
- `INVALID_TASK_TYPE`: Task type not supported
- `DUPLICATE_FOCUS_AREAS`: Focus areas contain duplicates
- `CONTENT_TOO_LONG`: Individual content exceeds length limit
- `INVALID_TIMEOUT`: Timeout value is invalid

### Validation Implementation
```go
func (s *SummaryAgent) Validate(input *AgentInput) error {
    var errors []ValidationError

    // Validate contents
    contents, ok := input.Data["contents"]
    if !ok {
        errors = append(errors, ValidationError{
            Field:   "contents",
            Message: "contents field is required",
            Code:    "MISSING_REQUIRED_FIELD",
        })
    } else {
        contentsArray, ok := contents.([]interface{})
        if !ok {
            errors = append(errors, ValidationError{
                Field:   "contents",
                Message: "contents must be an array",
                Code:    "INVALID_FIELD_TYPE",
                Value:   fmt.Sprintf("%T", contents),
            })
        } else if len(contentsArray) == 0 {
            errors = append(errors, ValidationError{
                Field:   "contents",
                Message: "contents array cannot be empty",
                Code:    "EMPTY_CONTENTS",
            })
        }
    }

    // Validate instructions
    instructions, ok := input.Data["instructions"]
    if !ok {
        errors = append(errors, ValidationError{
            Field:   "instructions",
            Message: "instructions field is required",
            Code:    "MISSING_REQUIRED_FIELD",
        })
    } else if instructionsStr, ok := instructions.(string); !ok {
        errors = append(errors, ValidationError{
            Field:   "instructions",
            Message: "instructions must be a string",
            Code:    "INVALID_FIELD_TYPE",
            Value:   fmt.Sprintf("%T", instructions),
        })
    } else if strings.TrimSpace(instructionsStr) == "" {
        errors = append(errors, ValidationError{
            Field:   "instructions",
            Message: "instructions cannot be empty",
            Code:    "INVALID_FIELD_VALUE",
        })
    }

    if len(errors) > 0 {
        return &ValidationErrors{Errors: errors}
    }
    return nil
}
```

## Usage Patterns

### Primary Agent Orchestration
```go
// Main agent orchestrates sub-agent execution
primary := agents.NewPrimary()
summary := agents.NewSummary()

input := &AgentInput{
    Data: map[string]interface{}{
        "task": "summarize_documents",
        "documents": []string{"doc1 content", "doc2 content"},
        "instructions": "Focus on key findings and recommendations",
    },
}

result, _ := primary.Execute(ctx, input)
```

### Summary Agent Usage
```go
// Direct summary agent usage
summary := agents.NewSummary()

input := &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{
            "First document content...",
            "Second document content...",
        },
        "instructions": "Extract key points and main conclusions",
        "focus_areas": []string{"findings", "recommendations"},
    },
}

result, _ := summary.Execute(ctx, input)
```

### Monitoring
```go
stats := agent.GetStats()
log.Printf("Success: %v, Tokens: %d, Duration: %v",
    stats.Success, stats.TokensUsed.Total, stats.Duration)
```

### API Usage with curl

#### Primary Agent - Document Summarization
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "summarize_documents",
      "contents": [
        "First document content...",
        "Second document content..."
      ],
      "instructions": "Focus on key findings",
      "focus_areas": ["findings", "conclusions"]
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "task": "summarize_documents",
    "result": {
      "summary": "Generated summary...",
      "metadata": {
        "content_count": 2,
        "combined_length": 500,
        "focus_areas": ["findings", "conclusions"]
      }
    },
    "orchestrator": "primary_agent"
  }
}
```

#### Primary Agent - General Query
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "general_query",
      "query": "What are the benefits of microservices?"
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "task": "general_query",
    "query": "What are the benefits of microservices?",
    "answer": "Microservices offer improved scalability, fault isolation...",
    "orchestrator": "primary_agent"
  }
}
```

### Authentication
Agent endpoints require API key authentication:
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{...}'
```
