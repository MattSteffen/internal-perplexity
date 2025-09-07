# Primary Agent

The Primary Agent is the main orchestrator agent that handles user interactions, analyzes queries, and coordinates sub-agent execution using a tool-like calling system. It serves as the central chatbot interface that decomposes complex user requests into sub-agent calls with parallel and sequential execution patterns.

## Purpose

The Primary Agent acts as the main user interface in the agent system, responsible for:
- **User Interaction**: Handling natural language queries from users
- **Query Analysis**: Using LLM to analyze and decompose user requests
- **Hybrid Orchestration**: Coordinating both sub-agents and tools using unified calling patterns
- **Execution Planning**: Determining optimal combinations of agents and tools
- **Result Synthesis**: Combining outputs from multiple agents and tools into coherent responses

## Features

- **Intelligent Query Analysis**: Uses LLM to analyze user queries and determine execution strategy
- **Hybrid Tool Integration**: Seamlessly combines sub-agents and tools in unified execution patterns
- **Tool-Like Calling System**: Supports both sub-agent and tool calls with structured specifications
- **Parallel & Sequential Execution**: Complex execution patterns [agent1, agent2] for parallel, [agent1], [agent2] for sequential
- **Dynamic Task Decomposition**: Automatically decomposes queries into optimal agent/tool combinations
- **Input/Output Validation**: Comprehensive validation for all agent and tool inputs/outputs
- **Comprehensive Statistics**: Tracks execution metrics for both agents and tools

## Architecture

### File Structure
- `definition.go`: Agent schema, types, and construction
- `primary.go`: Core execution logic and sub-agent orchestration
- `primary_test.go`: Unit tests for deterministic functionality
- `OVERVIEW.md`: Documentation and usage examples

### Execution Flow
1. **Query Analysis**: LLM analyzes user query to determine required sub-agents
2. **Execution Planning**: Creates specification for parallel/sequential execution
3. **Sub-Agent Calling**: Executes sub-agents using tool-like calling pattern
4. **Result Aggregation**: Combines outputs from multiple agents
5. **Response Generation**: Returns structured response to user

### Sub-Agent Calling Pattern
The Primary Agent uses a structured specification system:

```json
{
  "groups": [
    {
      "calls": [
        {
          "name": "calculator",
          "type": "tool",
          "input": {"expression": "15 + 27 * 3"},
          "output_key": "calculation_result",
          "description": "Calculate the mathematical expression"
        },
        {
          "name": "summary",
          "type": "subagent",
          "input": {"contents": ["doc1", "doc2"]},
          "output_key": "summary_result",
          "description": "Summarize the documents"
        }
      ],
      "description": "Parallel execution of calculation and summarization"
    },
    {
      "calls": [
        {
          "name": "synthesis",
          "type": "subagent",
          "input": {"inputs": ["calculation_result", "summary_result"]},
          "output_key": "final_result",
          "description": "Synthesize calculation and summary results"
        }
      ],
      "description": "Sequential synthesis of parallel results"
    }
  ]
}
```

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language user query |

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
    "results": {
      "group_0": {
        "summary_result": {...},
        "analysis_result": {...}
      },
      "group_1": {
        "final_result": {...}
      }
    },
    "sub_agents_used": ["summary", "analyst", "synthesis"],
    "execution_spec": {...}
  },
  "tokens_used": 850,
  "duration": "3.2s",
  "metadata": {
    "execution_groups": 2,
    "total_sub_agents": 3,
    "sub_agents_used": ["summary", "analyst", "synthesis"]
  }
}
```

## Usage Examples

### Basic Query Handling

```go
// Initialize with both sub-agents and tools
toolRegistry := tools.NewRegistry()
// Register available tools...
primaryAgent := primary.NewPrimaryAgent(subAgents, toolRegistry)

result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Query: "Calculate 15 + 27 * 3 and summarize these documents",
    Data: map[string]interface{}{
        "documents": []string{"doc1 content", "doc2 content"},
    },
}, llmProvider)

if result.Success {
    data := result.Content.(map[string]interface{})
    results := data["results"].(map[string]interface{})
    fmt.Printf("Orchestrated %d sub-agents and %d tools\n",
        len(data["sub_agents_used"].([]string)),
        len(data["tools_used"].([]string)))
}
```

### API Usage with curl

```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "query": "Summarize these documents and find related research",
      "documents": [
        "Document content about AI...",
        "Document content about ML..."
      ]
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "content": {
    "results": {
      "group_0": {
        "summary_result": {
          "summary": "Generated summary...",
          "metadata": {"content_count": 2}
        },
        "research_result": {
          "sources": ["source1", "source2"],
          "findings": "Research findings..."
        }
      }
    },
    "sub_agents_used": ["summary", "researcher"],
    "execution_spec": {...}
  },
  "tokens_used": 650,
  "duration": "4.1s"
}
```

## Sub-Agent Integration

### Available Sub-Agents
- **Summary**: Document summarization and content analysis
- **Analyst**: Data analysis and statistical processing
- **Researcher**: Web research and information gathering
- **Synthesis**: Result aggregation and synthesis

### Execution Patterns
- **Sequential**: `[agent1], [agent2]` - Execute agents one after another
- **Parallel**: `[agent1, agent2]` - Execute agents simultaneously
- **Map-Reduce**: Parallel processing followed by aggregation

## Input/Output Validation

### Input Validation
```go
err := primaryAgent.ValidateInput(input)
if err != nil {
    // Handle validation error
    log.Printf("Input validation failed: %v", err)
}
```

### Output Validation
```go
err := primaryAgent.ValidateOutput(result)
if err != nil {
    // Handle output validation error
    log.Printf("Output validation failed: %v", err)
}
```

### Validation Error Types
- `MISSING_REQUIRED_FIELD`: Required field is missing
- `INVALID_FIELD_TYPE`: Field has wrong data type
- `INVALID_OUTPUT`: Output doesn't match expected schema

## Capabilities

The Primary Agent provides these capabilities:

```go
capabilities := primaryAgent.GetCapabilities()
// Returns:
[
    {
        "name": "intelligent_orchestration",
        "description": "Intelligent task decomposition and execution planning"
    },
    {
        "name": "multi_agent_coordination",
        "description": "Coordinate multiple agents using tool-like calling patterns"
    },
    {
        "name": "adaptive_execution",
        "description": "Adapt execution patterns based on query requirements"
    },
    {
        "name": "tool_integration",
        "description": "Seamlessly integrate with sub-agents and tools"
    }
]
```

## Statistics and Monitoring

```go
stats := primaryAgent.GetStats()
// Returns:
{
    "total_executions": 250,
    "average_duration": "3.5s",
    "success_rate": 0.94,
    "total_tokens": 125000,
    "tasks_created": 450,
    "sub_agents_used": 380
}
```

### Metrics Tracked
- **Total Executions**: Number of queries processed
- **Average Duration**: Mean execution time per query
- **Success Rate**: Percentage of successful executions
- **Token Usage**: Total tokens consumed
- **Tasks Created**: Number of sub-agent calls created
- **Sub-Agents Used**: Number of sub-agent executions

## Configuration

### Schema Definition
```go
schema := primaryAgent.GetSchema()
// Provides complete I/O specifications
```

### System Prompts
```go
prompt := primaryAgent.GetSystemPrompt()
// Returns context-aware system prompt
```

## Error Handling

### Common Errors
- **Query Analysis Failure**: LLM fails to analyze query
- **Sub-Agent Not Found**: Requested sub-agent doesn't exist
- **Execution Timeout**: Query execution exceeds time limits
- **Validation Errors**: Input/output validation failures

### Error Response Format
```json
{
  "success": false,
  "error": "sub-agent 'nonexistent' not found",
  "duration": "0.5s",
  "metadata": {
    "error_type": "SUB_AGENT_NOT_FOUND",
    "requested_agent": "nonexistent"
  }
}
```

## Best Practices

### Query Design
1. **Clear Intent**: Write queries with clear, specific intent
2. **Context Provision**: Include relevant context and data
3. **Expectation Setting**: Be clear about desired output format

### Performance Optimization
1. **Query Complexity**: Balance query complexity with execution time
2. **Caching**: Cache frequently used analysis results
3. **Resource Monitoring**: Monitor token usage and execution times

### Error Handling
1. **Graceful Degradation**: Handle sub-agent failures gracefully
2. **Informative Errors**: Provide clear error messages
3. **Fallback Strategies**: Implement fallback execution patterns

## Implementation Details

### Core Components
- **Query Analyzer**: LLM-based query analysis and decomposition
- **Execution Planner**: Creates structured execution specifications
- **Sub-Agent Executor**: Manages parallel and sequential execution
- **Result Aggregator**: Combines and synthesizes results

### Validation Strategy
- **Schema-Based**: Validates against predefined I/O schemas
- **Type Checking**: Ensures correct data types for all fields
- **Business Logic**: Validates business rules and constraints

## Future Extensions

### Planned Features
- **Advanced Execution Patterns**: More complex workflow patterns
- **Dynamic Sub-Agent Loading**: Load sub-agents on demand
- **Result Caching**: Intelligent caching of execution results
- **Real-time Monitoring**: Live execution monitoring and metrics

### Integration Points
- **Tool Registry**: Integration with broader tool ecosystem
- **External APIs**: Integration with external services
- **Workflow Templates**: Predefined execution templates