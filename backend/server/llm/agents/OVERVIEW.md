# Agents Package

Intelligent agent framework with tool-like support calling, comprehensive validation, and modular architecture.

## Architecture

### Agent Types

- **Main**: Main orchestrator that handles user interactions and coordinates both support and tools using unified calling patterns
- **Support**: Specialized agents (Summary, Analyst, Researcher, Synthesis) with predefined I/O schemas

### Design Patterns

- **Unified Tool-Like Calling**: Both support and tools are called using structured specifications just like tool calling
- **Parallel/Sequential Execution**: Support for complex execution patterns [agent1, agent2] for parallel, [agent1], [agent2] for sequential
- **Schema-Based Validation**: input/output validation using predefined schemas
- **Modular Architecture**: Each agent has separate definition.go, agent_name.go, and test files

## Directory Structure

### Core Files

- `definition.go`: Core types, interfaces, and schemas
- `task_planner.go`: Task decomposition and planning
  - The structured tool to be used in calling agents and agent orchestration
- `task_executor.go`: Task execution orchestration from planned orchestration
- `runtime.go`: The run time interface, implementation, and manager

### main/[agent-name]/

**Contents:**

- `definition.go`: Primary agent schema and construction
- `agent_name.go`: User interaction handling and support orchestration
- `agent_name_test.go`: Comprehensive test coverage of deterministic sections
- `OVERVIEW.md`: Detailed documentation and examples

**Key Features:**

- Intelligent query analysis using LLM
- Tool-like support calling with execution patterns
- Parallel and sequential support coordination
- Input/output validation with detailed error messages
- Comprehensive statistics and monitoring

### support/[support-agent]/

**Contents:**

- `definition.go`: Summary agent schema and validation
- `support_agent_name.go`: Content summarization logic
- `support_agent_name_test.go`: Unit tests and validation testing
- `OVERVIEW.md`: Documentation and usage examples

**Key Features:**

- Multi-document summarization
- Instruction-based summarization
- Focus area support
- Comprehensive input/output validation
- Structured summary results

### support/synthesis/

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

## Agent Interface

```go
type Agent interface {
    Name() string
    ValidateInput(input *AgentInput) []ValidationError
    Execute(ctx context.Context, input *AgentInput, rt *Runtime) (*AgentResult, error)
}

type AgentInput struct {
  Query   string
  Data    map[string]any
  Meta    map[string]any
  Session SessionInfo
}

type AgentResult struct {
    Content     any
    Success     bool
    Stats       AgentStats
    Metadata    map[string]any
}

type AgentStats struct {
  StartedAt   time.Time
  FinishedAt  time.Time
  Duration    time.Duration
  TokensIn    int
  TokensOut   int
  CallsMade   int
  Parallelism int
}

type SessionInfo struct {
  SessionID string
  UserID    string
}

type Runtime struct {
    LLM     llm.Provider
    Broker  Broker
    Agents  map[string]Agent
    Tools   map[string]Tool
    Logger  *zerolog.Logger
}
```

## Input Validation

Each agent implements comprehensive input validation with specific error types for invalid inputs.

### Validation

```go
type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
    Code    string `json:"code"`
    Value   any `json:"value,omitempty"`
}
```

## Tool-Like support Calling

The Primary Agent uses a structured specification system for calling support, similar to OpenAI tool calling:

```go
type AgentCall struct {
  Name Kind
  Arguments struct {
    Support []SupportCalls
    Tools []ToolCall // like openai's tool call syntax, name, arguments
  }
}

type SupportCalls struct {
  Calls []SupportCall
  Description string
}

type SupportCall struct {
  ToolCall ToolCall // like openai's tool call syntax, name, arguments
  Description string
}

type Kind string
const (
    AgentKind Kind = "agent"
    ToolKind  Kind = "tool"
)
```

```json
{
  "name": "support",
  "arguments": {
    "support_calls": [
      {
        "calls": [
          {
            "name": "summary",
            "input": {
              "contents": ["doc1", "doc2"],
              "instructions": "Summarize key points"
            },
            "output_key": "summary_result",
            "description": "Summarize the provided documents"
          },
          {
            "name": "analyst",
            "input": {
              "data": "analysis_data",
              "analysis_type": "statistical"
            },
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
            "input": { "inputs": ["summary_result", "analysis_result"] },
            "output_key": "final_result",
            "description": "Synthesize all results into final response"
          }
        ],
        "description": "Sequential synthesis of parallel results"
      }
    ],
    "tool_calls": []
  }
}
```

## Memory Management

```go
type Broker interface {
    // Stores v, returns opaque ref (UUID, key, …)
    Put(ctx context.Context, v any) (DataRef, error)
    Get(ctx context.Context, r DataRef) (any, error)
}

type DataRef string
```

## Usage Patterns

TODO: Show examples

## Execution Patterns

### Parallel Execution

Multiple support execute simultaneously:

### Sequential Execution

support execute in sequence:

### Map-Reduce Pattern

Parallel processing followed by aggregation:

## Monitoring and Statistics

TODO: Figure out what is relevant

## Error Handling

TODO: Figure out how to do this well

## API Integration

TODO: Show examples
