# Agent Server MVP - Task Outline

## Overview
Build a minimal viable product (MVP) for the agent orchestration server with OpenAI API integration, focusing on core functionality and testing with local Ollama instance (localhost:11434, gpt-oss:20b model).

## Core MVP Components

### 1. LLM Foundation (`llm/models/`)
- [ ] OpenAI ChatCompletion client with localhost:11434 fallback using sashabaranov/go-openai
- [ ] Unified provider interface for multiple LLM backends
- [ ] Token counting and usage tracking
- [ ] Error handling and retry logic
- [ ] Test connection to localhost:11434 with gpt-oss:20b

### 2. Tool System (`llm/tools/`)
- [ ] Document summarizer tool (deterministic, ≤1 LLM call)
- [ ] Calculator tool for basic mathematical operations
- [ ] Web search tool (mock implementation)
- [ ] JSON schema validation for tool inputs
- [ ] Tool registry with discovery
- [ ] Tool execution framework

### 3. Agent Framework (`llm/agents/`)
- [ ] Summary sub-agent that takes list of text contents and instructions
- [ ] Primary main-agent orchestrator
- [ ] Agent execution pipeline
- [ ] Agent result formatting with stats

### 4. Services Layer (`llm/services/`)
- [ ] Basic agent manager for agent lifecycle
- [ ] Task orchestration framework
- [ ] Basic caching layer

### 5. API Layer (`api/`)
- [ ] POST /agents/{name} endpoint
- [ ] POST /sub-agents/{name} endpoint
- [ ] POST /tools/{name} endpoint
- [ ] GET /tasks/{id} endpoint
- [ ] Basic error handling and validation

### 6. HTTP Handlers (`handlers/`)
- [ ] Agent execution handlers
- [ ] Tool execution handlers
- [ ] Task status handlers
- [ ] Request validation middleware

### 7. Server Setup (`server/`)
- [ ] net/http server configuration
- [ ] Route setup and middleware
- [ ] CORS and security headers
- [ ] Graceful shutdown handling

### 8. Testing & Integration
- [ ] Unit tests for tools and agents
- [ ] Integration tests with localhost:11434 (gpt-oss:20b)
- [ ] API endpoint tests
- [ ] End-to-end agent execution tests
- [ ] Performance benchmarking

## Development Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up Go project structure
- [ ] Implement OpenAI client with Ollama fallback
- [ ] Create basic tool framework
- [ ] Build simple document summarizer tool
- [ ] Build calculator tool with expression evaluation
- [ ] Write unit tests for tools

### Phase 2: Agent Core (Week 2)
- [ ] Implement summary sub-agent
- [ ] Create primary main-agent
- [ ] Build agent manager service
- [ ] Test agent execution pipeline

### Phase 3: API Integration (Week 3)
- [ ] Create HTTP handlers
- [ ] Set up REST endpoints
- [ ] Add request validation
- [ ] Implement error responses
- [ ] Test API endpoints with curl/Postman

### Phase 4: Testing & Polish (Week 4)
- [ ] Write comprehensive integration tests
- [ ] Test with localhost:11434 (gpt-oss:20b)
- [ ] Performance optimization
- [ ] Documentation updates
- [ ] MVP demonstration

## Testing Strategy

### Local Testing (Ollama)
```bash
# Start Ollama with gpt-oss:20b
ollama serve
ollama pull gpt-oss:20b

# Run tests against localhost:11434
go test ./... -v
```

### API Testing
```bash
# Test agent execution
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -d '{"input": {"task": "summarize documents"}}'

# Test tool execution
curl -X POST http://localhost:8080/tools/document_summarizer \
  -H "Content-Type: application/json" \
  -d '{"input": {"content": "Test document"}}'

# Test calculator tool
curl -X POST http://localhost:8080/tools/calculator \
  -H "Content-Type: application/json" \
  -d '{"input": {"expression": "15 + 27 * 3"}}'
```

## Success Criteria
- [ ] All tools execute deterministically
- [ ] Calculator tool performs accurate mathematical calculations
- [ ] Agents return results with proper stats
- [ ] API endpoints respond correctly
- [ ] Tests pass with localhost:11434
- [ ] Basic agent orchestration works
- [ ] Error handling is robust
- [ ] Code is well-documented

## Dependencies
- [ ] Go 1.24.3
- [ ] Ollama running locally (gpt-oss:20b)
- [ ] github.com/sashabaranov/go-openai OpenAI client
- [ ] github.com/stretchr/testify for testing

## Risk Mitigation
- [ ] Start with simple tools, expand complexity
- [ ] Test each component individually before integration
- [ ] Use Ollama for local testing to avoid API costs
- [ ] Implement proper error handling early
- [ ] Keep interfaces simple and extensible
