# Primary Agent

The Primary Agent is the main orchestrator agent that coordinates complex workflows, manages sub-agents, and handles high-level task decomposition and execution. It serves as the central intelligence for multi-step processes and agent orchestration.

## Purpose

The Primary Agent acts as the main orchestrator in the agent system, responsible for:
- **Task Decomposition**: Breaking down complex tasks into manageable sub-tasks
- **Agent Orchestration**: Coordinating multiple sub-agents for complex workflows
- **Workflow Management**: Managing execution flow and result aggregation
- **Decision Making**: Determining which agents and tools to use for specific tasks
- **Result Synthesis**: Combining outputs from multiple agents into coherent responses

## Features

- **Multi-Agent Orchestration**: Coordinates sub-agents for complex tasks
- **Task Routing**: Automatically routes tasks to appropriate specialized agents
- **State Management**: Maintains context across multi-step workflows
- **Error Handling**: Robust error handling and recovery mechanisms
- **Statistics Tracking**: Comprehensive execution statistics and monitoring
- **Extensible Architecture**: Easy to add new task types and agent integrations

## Task Types Supported

### Document Summarization
**Task Type**: `"summarize_documents"`
**Description**: Orchestrates document summarization using the summary sub-agent
**Required Fields**:
- `contents`: Array of document strings to summarize
- `instructions`: Summarization instructions (optional)
- `focus_areas`: Areas to focus on (optional)

### General Queries
**Task Type**: `"general_query"`
**Description**: Handles general questions and requests using direct LLM calls
**Required Fields**:
- `query`: The question or request text

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | Yes | Task type identifier (e.g., "summarize_documents", "general_query") |
| `contents` | array | For summarization | Array of document strings |
| `instructions` | string | No | Instructions for summarization |
| `query` | string | For queries | The query text |
| `focus_areas` | array | No | Focus areas for summarization |

### Context Parameters
Additional parameters can be passed via `input.Context`:
- `model`: LLM model to use (default: "gpt-4")
- `api_key`: API key for the LLM provider
- `timeout`: Execution timeout duration

## Output Schema

```json
{
  "success": true,
  "content": {
    "task": "summarize_documents",
    "result": {
      "summary": "Generated summary text...",
      "metadata": {
        "content_count": 2,
        "combined_length": 1500,
        "focus_areas": ["key_points"],
        "instructions": "Focus on main conclusions"
      }
    },
    "orchestrator": "primary_agent"
  },
  "tokens_used": 450,
  "duration": "2.5s",
  "metadata": {
    "sub_agent": "summary",
    "execution_path": "primary -> summary"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | The executed task type |
| `result` | object | Task-specific result data |
| `orchestrator` | string | Identifier for the orchestrating agent |
| `sub_agent` | string | Name of the sub-agent used (if applicable) |
| `execution_path` | string | Path of agent execution (e.g., "primary -> summary") |

## Usage Examples

### Document Summarization

```go
primaryAgent := primary.NewPrimaryAgent(llmClient, summaryAgent)

result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "task": "summarize_documents",
        "contents": []interface{}{
            "First document content about AI...",
            "Second document content about ML...",
        },
        "instructions": "Focus on key technological advances",
        "focus_areas": []interface{}{"advances", "applications"},
    },
})

if result.Success {
    data := result.Content.(map[string]interface{})
    summary := data["result"].(map[string]interface{})
    fmt.Printf("Summary: %s\n", summary["summary"])
}
```

### API Usage with curl

#### Document Summarization Task
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "summarize_documents",
      "contents": [
        "Go is a programming language developed by Google.",
        "It features garbage collection and concurrent execution."
      ],
      "instructions": "Focus on key features and benefits",
      "focus_areas": ["features", "benefits"]
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
      "summary": "Go is a programming language created by Google that features garbage collection and concurrent execution, offering significant benefits for modern software development.",
      "metadata": {
        "content_count": 2,
        "combined_length": 89,
        "focus_areas": ["features", "benefits"],
        "instructions": "Focus on key features and benefits"
      }
    },
    "orchestrator": "primary_agent"
  }
}
```

#### General Query Task
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "general_query",
      "query": "What are the benefits of microservices architecture?"
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
    "query": "What are the benefits of microservices architecture?",
    "answer": "Microservices architecture offers several key benefits including improved scalability, better fault isolation, technology diversity, and easier deployment and maintenance compared to monolithic architectures.",
    "orchestrator": "primary_agent"
  }
}
```

### General Query

```go
result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "task": "general_query",
        "query": "What are the benefits of microservices architecture?",
    },
    Context: map[string]interface{}{
        "model": "gpt-4",
        "api_key": "your-api-key",
    },
})

if result.Success {
    data := result.Content.(map[string]interface{})
    fmt.Printf("Answer: %s\n", data["answer"])
}
```

## Capabilities

The Primary Agent provides the following capabilities:

```go
capabilities := primaryAgent.GetCapabilities()
// Returns:
[
    {
        "name": "document_summarization",
        "description": "Orchestrate document summarization using specialized sub-agents"
    },
    {
        "name": "general_queries",
        "description": "Handle general queries and provide helpful responses"
    },
    {
        "name": "task_orchestration",
        "description": "Coordinate multiple agents and tools for complex tasks"
    }
]
```

## Architecture

### Execution Flow
1. **Input Validation**: Validate task type and required fields
2. **Task Routing**: Route to appropriate handler based on task type
3. **Agent Orchestration**: Coordinate sub-agents as needed
4. **Result Processing**: Process and format results
5. **Statistics Update**: Update execution statistics
6. **Response Packaging**: Return structured response

### Sub-Agent Integration
The Primary Agent integrates with specialized sub-agents:
- **Summary Agent**: For document summarization tasks
- **Future Agents**: Analyst, Researcher, etc.

### State Management
- Maintains execution statistics across runs
- Tracks success rates and performance metrics
- Manages agent lifecycle and resource usage

## Configuration

### Dependencies
- **LLM Provider**: Required for general queries and orchestration
- **Summary Agent**: Required for document summarization tasks
- **Context Providers**: Optional for enhanced functionality

### Performance Settings
- **Default Model**: GPT-4 (configurable via context)
- **Token Limits**: 10,000 tokens for general queries
- **Timeout Handling**: Configurable execution timeouts

## Error Handling

### Validation Errors
**Input:** Missing required task field
```json
{
  "success": false,
  "error": "task field is required",
  "duration": "0.001s"
}
```

### Task Execution Errors
**Input:** Invalid task type
```json
{
  "success": false,
  "error": "unsupported task type: invalid_task",
  "duration": "0.002s"
}
```

### Sub-Agent Errors
**Input:** Summary agent failure
```json
{
  "success": false,
  "error": "summary agent failed: content validation error",
  "duration": "1.5s"
}
```

## Statistics and Monitoring

The Primary Agent tracks comprehensive statistics:

```go
stats := primaryAgent.GetStats()
// Returns:
{
    "total_executions": 150,
    "average_duration": "2.3s",
    "success_rate": 0.96,
    "total_tokens": 45000
}
```

### Metrics Tracked
- **Total Executions**: Number of tasks processed
- **Average Duration**: Mean execution time per task
- **Success Rate**: Percentage of successful executions
- **Token Usage**: Total tokens consumed across all executions

## Best Practices

### Task Design
1. **Clear Task Types**: Use descriptive, specific task identifiers
2. **Complete Input Data**: Provide all required fields for the task type
3. **Appropriate Context**: Include relevant context parameters when needed

### Error Handling
1. **Validate Inputs**: Always validate input data before processing
2. **Graceful Degradation**: Handle sub-agent failures gracefully
3. **Informative Errors**: Provide clear error messages for debugging

### Performance Optimization
1. **Resource Management**: Monitor and manage resource usage
2. **Caching**: Implement result caching for repeated tasks
3. **Async Processing**: Use asynchronous execution for long-running tasks

## Implementation Details

### Task Handler Pattern
```go
func (p *PrimaryAgent) executeDocumentSummary(ctx context.Context, input *agents.AgentInput, start time.Time) (*agents.AgentResult, error) {
    // 1. Prepare sub-agent input
    // 2. Execute sub-agent
    // 3. Process results
    // 4. Update statistics
    // 5. Return formatted response
}
```

### Validation Strategy
- **Early Validation**: Validate inputs before processing
- **Type Checking**: Ensure correct data types for all fields
- **Business Logic**: Validate business rules and constraints

### Result Aggregation
- **Structured Output**: Consistent result format across all tasks
- **Metadata Preservation**: Maintain execution context and metadata
- **Error Propagation**: Properly propagate errors with context

## Future Extensions

### Planned Features
- **Additional Task Types**: More specialized task handlers
- **Advanced Orchestration**: Complex multi-agent workflows
- **Caching Layer**: Result caching and optimization
- **Monitoring Dashboard**: Real-time performance monitoring

### Integration Points
- **Tool Registry**: Integration with tool ecosystem
- **External Services**: API integrations and external data sources
- **Workflow Engine**: Advanced workflow orchestration capabilities
