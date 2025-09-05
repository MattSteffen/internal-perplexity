# Agent Server

Orchestration and API access layer for LLM-powered agents following modern agent system design patterns.

## Architecture Overview

This server implements a comprehensive agent orchestration system with:

- **Vertical/Horizontal Task Decomposition**: Sequential pipelines and MapReduce patterns
- **Specialization Patterns**: Capability, domain, and model-based agent specialization
- **Optimizations**: Caching, parallel execution, stateless design, monitoring
- **API Layer**: RESTful endpoints following MCP-inspired patterns

## Directory Structure

```
server/
├── llm/                    # Core LLM orchestration package
│   ├── models/            # Provider integrations & shared types
│   ├── agents/            # Agent implementations
│   ├── tools/             # Deterministic tools
│   └── services/          # Orchestration frameworks
├── api/                   # REST API endpoints
├── agents/                # Legacy agent implementations
├── handlers/              # HTTP request handlers
├── models/                # Data models & DTOs
└── services/              # Business logic services
```

## Core Components

### LLM Package (`llm/`)

#### Models (`llm/models/`)
- **openai/**: OpenAI ChatCompletion integration
- **[provider]/**: Extensible provider pattern
- **shared/**: Unified interfaces and orchestration framework

#### Agents (`llm/agents/`)
- **main-agents/primary/**: Main orchestrator with sub-agent access
- **sub-agents/**: Specialized agents (researcher, analyst, summary)

#### Tools (`llm/tools/`)
- Deterministic tools with JSON schemas
- ≤1 LLM call per execution
- Predictable input/output patterns

#### Services (`llm/services/`)
- **conversations/**: Context and session management
- **agent-manager/**: Agent lifecycle and task orchestration

### API Layer (`api/`)

RESTful endpoints for agent interaction:
- `POST /agents/{name}` - Execute main agents
- `POST /sub-agents/{name}` - Execute sub-agents
- `POST /tools/{name}` - Execute tools
- `GET /tasks/{id}` - Task status and results

## Design Patterns Implemented

### Task Decomposition
```go
// Sequential Pipeline
researcher → analyst → summary

// MapReduce Pattern
input → (researcher, analyst, summarizer) → consensus
```

### Agent Specialization
```go
// Capability-based
researcherAgent := NewResearcher()  // Information gathering
analystAgent := NewAnalyst()        // Data analysis
summaryAgent := NewSummary()        // Report generation

// Model-based
fastAgent := NewAgent(HaikuModel)   // Speed optimized
complexAgent := NewAgent(OpusModel) // Quality optimized
```

### Orchestration Framework
```go
orchestrator := NewOrchestrator()

// Register agents
orchestrator.RegisterAgent("researcher", researcherAgent)

// Execute with caching and monitoring
result, err := orchestrator.ExecuteTask(ctx, &Task{
    Type: "research",
    Input: query,
    Cache: true,
})
```

## Key Features

### 🚀 Performance Optimizations
- **Caching**: Prompt hash-based result caching (TTL configurable)
- **Parallel Execution**: Independent agents run concurrently
- **Stateless Design**: Everything but main agent is stateless
- **Streaming**: Real-time responses for long-running tasks

### 📊 Monitoring & Observability
- **Metrics**: Token usage, latency, success rates
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Agent and tool health monitoring
- **Tracing**: Request tracing across agent calls

### 🔧 Tool System
- **Deterministic**: Same input → same output
- **Schema-driven**: JSON schema validation
- **Registry**: Centralized tool discovery
- **Testing**: Isolated tool testing framework

### 🌐 API Design
- **RESTful**: Standard HTTP patterns
- **Async Support**: Background task execution
- **Webhooks**: Callback support for long tasks
- **Streaming**: Server-sent events for real-time updates

## Getting Started

### Prerequisites
- Go 1.24+
- PostgreSQL (optional, for persistence)
- Redis (optional, for caching)

### Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd backend/server
go mod download
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Run the server:**
```bash
go run main.go
```

### Configuration

Key configuration options:
```yaml
server:
  port: 8080
  host: "0.0.0.0"

llm:
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"
    # Add other providers...

caching:
  redis_url: "redis://localhost:6379"
  ttl_hours: 24

monitoring:
  prometheus_enabled: true
  metrics_port: 9090
```

## Usage Examples

### Execute a Research Agent
```bash
curl -X POST http://localhost:8080/agents/researcher \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "query": "Latest AI agent system architectures",
      "depth": "comprehensive"
    }
  }'
```

### Execute a Tool
```bash
curl -X POST http://localhost:8080/tools/document_summarizer \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "content": "Your document text here...",
      "max_length": 200
    }
  }'
```

### Check Task Status
```bash
curl http://localhost:8080/tasks/task_123
```

## Development

### Project Structure Guidelines

1. **Models**: Define interfaces and types first
2. **Tools**: Implement deterministic tools with schemas
3. **Agents**: Build specialized agents using tools
4. **Services**: Create orchestration frameworks
5. **API**: Expose functionality through REST endpoints

### Testing Strategy

```go
// Unit tests for tools
func TestDocumentSummarizer(t *testing.T) {
    tool := NewDocumentSummarizer()
    result, err := tool.Execute(ctx, input)
    assert.NoError(t, err)
    assert.True(t, result.Success)
}

// Integration tests for agents
func TestResearchAgentFlow(t *testing.T) {
    agent := NewResearcherAgent(toolRegistry, llmClient)
    result, err := agent.Execute(ctx, query)
    assert.NoError(t, err)
    assert.Greater(t, len(result.Content.(*ResearchResult).Sources), 0)
}
```

### Adding New Components

#### New Tool
1. Create tool directory: `llm/tools/new_tool/`
2. Implement `Tool` interface
3. Define JSON schema
4. Add to tool registry

#### New Agent
1. Choose agent type (main/sub)
2. Implement `Agent` interface
3. Register with agent manager
4. Add API endpoint

#### New Provider
1. Create provider directory: `llm/models/new_provider/`
2. Implement `LLMProvider` interface
3. Add configuration support
4. Update shared interfaces if needed

## Monitoring & Metrics

### Metrics Collected
- **Performance**: Response times, throughput, error rates
- **Resource Usage**: Token consumption, memory usage
- **Agent Stats**: Success rates, tool usage patterns
- **System Health**: Agent availability, tool responsiveness

### Dashboards
Access metrics at `/metrics` (Prometheus format) for:
- Grafana dashboards
- Alert manager integration
- Custom monitoring solutions

## Best Practices

### Agent Development
1. **Keep agents focused**: Single responsibility principle
2. **Use tools effectively**: Leverage deterministic tools
3. **Monitor everything**: Built-in metrics from day one
4. **Handle failures gracefully**: Comprehensive error handling

### Tool Development
1. **Define schemas first**: Clear input/output contracts
2. **Test determinism**: Same input always produces same output
3. **Optimize performance**: Minimize LLM calls
4. **Document thoroughly**: Usage examples and limitations

### API Design
1. **Follow REST principles**: Proper HTTP methods and status codes
2. **Support async operations**: Don't block on long-running tasks
3. **Provide clear errors**: Descriptive error messages
4. **Version your APIs**: Proper API versioning strategy

## Contributing

1. Follow the established patterns and directory structure
2. Add comprehensive tests for new components
3. Update documentation for any new features
4. Ensure all lints pass before submitting

## License

[Your License Here]
