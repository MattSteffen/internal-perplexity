# Agents Package

Intelligent agent framework with tool-like sub-agent calling, comprehensive validation, and modular architecture.

## Architecture

### Agent Types
- **Primary Agent**: Main orchestrator that handles user interactions and coordinates both sub-agents and tools using unified calling patterns
- **Sub-Agents**: Specialized agents (Summary, Analyst, Researcher, Synthesis) with predefined I/O schemas
- **Synthesis Agent**: Aggregation agent that combines outputs from multiple sub-agents

### Design Patterns
- **Unified Tool-Like Calling**: Both sub-agents and tools are called using structured specifications in unified execution patterns
- **Parallel/Sequential Execution**: Support for complex execution patterns [agent1, agent2] for parallel, [agent1], [agent2] for sequential
- **Schema-Based Validation**: Comprehensive input/output validation using predefined schemas
- **Modular Architecture**: Each agent has separate definition.go, agent_name.go, and test files
- **Aggregation Pattern**: Synthesis agent combines and resolves conflicts between sub-agent outputs

## Directory Structure

### Core Files
- `definition.go`: Core types, interfaces, and schemas
- `types.go`: Legacy compatibility (deprecated)
- `system_prompts.go`: System prompt management
- `task_planner.go`: Task decomposition and planning
- `decision_engine.go`: Decision-making for tool/agent selection
- `execution_engine.go`: Task execution orchestration

### main-agents/primary/
**Contents:**
- `definition.go`: Primary agent schema and construction
- `primary.go`: User interaction handling and sub-agent orchestration
- `primary_test.go`: Comprehensive test coverage
- `OVERVIEW.md`: Detailed documentation and examples

**Key Features:**
- Intelligent query analysis using LLM
- Tool-like sub-agent calling with execution patterns
- Parallel and sequential sub-agent coordination
- Input/output validation with detailed error messages
- Comprehensive statistics and monitoring

### sub-agents/summary/
**Contents:**
- `definition.go`: Summary agent schema and validation
- `summary.go`: Content summarization logic
- `summary_test.go`: Unit tests and validation testing
- `OVERVIEW.md`: Documentation and usage examples

**Key Features:**
- Multi-document summarization
- Instruction-based summarization
- Focus area support
- Comprehensive input/output validation
- Structured summary results

### sub-agents/synthesis/
**Contents:**
- `definition.go`: Synthesis agent schema and validation
- `synthesis.go`: Multi-source synthesis and aggregation
- `synthesis_test.go`: Comprehensive test coverage
- `OVERVIEW.md`: Documentation and usage examples

**Key Features:**
- Multi-source information integration
- Conflict resolution and validation
- Structured synthesis output
- Confidence scoring and assessment
- Metadata preservation and tracking

### sub-agents/analyst/ (NOT IMPLEMENTED)
**Contents:**
- `definition.go`: Analyst agent schema (placeholder)
- `analyst.go`: Returns "not implemented" error
- `analyst_test.go`: Test coverage for future implementation
- `OVERVIEW.md`: Detailed future feature documentation

**Status:** Placeholder with comprehensive documentation for future implementation

### sub-agents/researcher/ (NOT IMPLEMENTED)
**Contents:**
- `definition.go`: Researcher agent schema (placeholder)
- `researcher.go`: Returns "not implemented" error
- `researcher_test.go`: Test coverage for future implementation
- `OVERVIEW.md`: Detailed future feature documentation

**Status:** Placeholder with comprehensive documentation for future implementation

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

## Tool-Like Sub-Agent Calling

The Primary Agent uses a structured specification system for calling sub-agents, similar to OpenAI tool calling:

```json
{
  "groups": [
    {
      "calls": [
        {
          "name": "summary",
          "input": {"contents": ["doc1", "doc2"], "instructions": "Summarize key points"},
          "output_key": "summary_result",
          "description": "Summarize the provided documents"
        },
        {
          "name": "analyst",
          "input": {"data": "analysis_data", "analysis_type": "statistical"},
          "output_key": "analysis_result",
          "description": "Analyze the summarized content"
        }
      ],
      "description": "Parallel execution of summary and analysis"
    },
    {
      "calls": [
        {
          "name": "synthesis",
          "input": {"inputs": ["summary_result", "analysis_result"]},
          "output_key": "final_result",
          "description": "Synthesize all results into final response"
        }
      ],
      "description": "Sequential synthesis of parallel results"
    }
  ]
}
```

## Usage Patterns

### Primary Agent with Sub-Agent Orchestration
```go
// Initialize agents
primaryAgent := primary.NewPrimaryAgent(map[string]agents.Agent{
    "summary":   summary.NewSummaryAgent(),
    "synthesis": synthesis.NewSynthesisAgent(),
})

// User query triggers intelligent analysis and sub-agent calling
result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Query: "Summarize these documents and provide insights",
    Data: map[string]interface{}{
        "documents": []string{"doc1 content", "doc2 content"},
    },
}, llmProvider)

if result.Success {
    data := result.Content.(map[string]interface{})
    results := data["results"].(map[string]interface{})
    fmt.Printf("Orchestrated %d sub-agents\n", len(data["sub_agents_used"].([]string)))
}
```

### Direct Sub-Agent Usage
```go
// Direct summary agent usage with validation
summaryAgent := summary.NewSummaryAgent()

input := &agents.AgentInput{
    Data: map[string]interface{}{
        "contents": []interface{}{
            "First document content...",
            "Second document content...",
        },
        "instructions": "Extract key points and main conclusions",
        "focus_areas": []interface{}{"findings", "recommendations"},
    },
}

// Validate input
if err := summaryAgent.ValidateInput(input); err != nil {
    log.Printf("Input validation failed: %v", err)
    return
}

result, err := summaryAgent.Execute(ctx, input, llmProvider)
if result.Success {
    content := result.Content.(map[string]interface{})
    fmt.Printf("Summary: %s\n", content["summary"])
}
```

### Synthesis Agent for Aggregation
```go
synthesisAgent := synthesis.NewSynthesisAgent()

result, err := synthesisAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "inputs": map[string]interface{}{
            "summary_result":  "Document summary...",
            "analysis_result": "Data analysis...",
            "research_result": "Research findings...",
        },
        "instructions": "Create a comprehensive report combining all findings",
        "format": "structured_report",
    },
}, llmProvider)

if result.Success {
    synthesis := result.Content.(map[string]interface{})
    fmt.Printf("Synthesis confidence: %.2f\n", synthesis["confidence"])
}
```

## Execution Patterns

### Parallel Execution
Multiple sub-agents execute simultaneously:
```go
// [agent1, agent2] - Parallel execution
groups: [{
    calls: [
        {name: "summary", input: {...}, output_key: "summary"},
        {name: "analyst", input: {...}, output_key: "analysis"}
    ],
    description: "Parallel processing"
}]
```

### Sequential Execution
Sub-agents execute in sequence:
```go
// [agent1], [agent2] - Sequential execution
groups: [
    {calls: [{name: "summary", input: {...}, output_key: "summary"}], description: "First step"},
    {calls: [{name: "synthesis", input: {inputs: ["summary"]}, output_key: "final"}], description: "Second step"}
]
```

### Map-Reduce Pattern
Parallel processing followed by aggregation:
```go
// Parallel map phase + sequential reduce phase
groups: [
    {calls: [{name: "summary1"}, {name: "summary2"}], description: "Map phase"},
    {calls: [{name: "synthesis", input: {inputs: ["summary1", "summary2"]}}], description: "Reduce phase"}
]
```

## Monitoring and Statistics

### Agent Statistics
```go
stats := agent.GetStats()
fmt.Printf("Executions: %d, Success Rate: %.2f, Avg Duration: %v\n",
    stats.TotalExecutions, stats.SuccessRate, stats.AverageDuration)
```

### Execution Logging
```go
result, err := agent.Execute(ctx, input, llmProvider)
if result.ExecutionLog != nil {
    for _, step := range result.ExecutionLog {
        fmt.Printf("[%s] %s\n", step.Type, step.Description)
    }
}
```

## Error Handling

### Validation Errors
```json
{
  "success": false,
  "error": "contents field is required",
  "duration": "0.001s",
  "metadata": {
    "error_type": "MISSING_REQUIRED_FIELD",
    "field": "contents"
  }
}
```

### Execution Errors
```json
{
  "success": false,
  "error": "sub-agent 'analyst' not found",
  "duration": "0.5s",
  "metadata": {
    "error_type": "SUB_AGENT_NOT_FOUND",
    "requested_agent": "analyst"
  }
}
```

### Not Implemented Errors
```json
{
  "success": false,
  "error": "analyst agent is not yet implemented - please use summary and researcher agents for current analysis needs",
  "metadata": {
    "error_type": "NOT_IMPLEMENTED",
    "recommended_agents": ["summary"],
    "agent_status": "not_implemented"
  }
}
```

## API Integration

### Primary Agent Endpoint
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "query": "Analyze these documents and provide insights",
      "documents": ["Document 1 content...", "Document 2 content..."]
    },
    "model": "gpt-4"
  }'
```

### Response Format
```json
{
  "success": true,
  "content": {
    "results": {
      "group_0": {
        "summary_result": {
          "summary": "Generated summary...",
          "content_count": 2,
          "combined_length": 1500
        },
        "analysis_result": {
          "insights": ["Key insight 1", "Key insight 2"],
          "confidence": 0.87
        }
      }
    },
    "sub_agents_used": ["summary", "analyst"],
    "execution_spec": {...}
  },
  "tokens_used": 650,
  "duration": "4.1s",
  "metadata": {
    "execution_groups": 1,
    "total_sub_agents": 2
  }
}
```

## Testing

### Unit Tests
```bash
# Run all agent tests
go test ./llm/agents/...

# Run specific agent tests
go test ./llm/agents/sub-agents/summary/
go test ./llm/agents/main-agents/primary/

# Run with coverage
go test -cover ./llm/agents/...
```

### Integration Tests
```go
func TestPrimaryAgentIntegration(t *testing.T) {
    // Initialize all agents
    subAgents := map[string]agents.Agent{
        "summary": summary.NewSummaryAgent(),
        "synthesis": synthesis.NewSynthesisAgent(),
    }
    primaryAgent := primary.NewPrimaryAgent(subAgents)

    // Test full orchestration
    result, err := primaryAgent.Execute(ctx, testInput, llmProvider)
    assert.NoError(t, err)
    assert.True(t, result.Success)
}
```

## Future Extensions

### Planned Enhancements
- **Advanced Execution Patterns**: Complex workflow patterns with conditionals
- **Dynamic Agent Loading**: Load sub-agents on demand
- **Result Caching**: Intelligent caching of execution results
- **Real-time Monitoring**: Live execution monitoring and metrics
- **Agent Marketplace**: Pluggable agent ecosystem

### Integration Points
- **Tool Registry**: Integration with broader tool ecosystem
- **External APIs**: Integration with external services
- **Workflow Templates**: Predefined execution templates
- **Performance Analytics**: Advanced performance monitoring
