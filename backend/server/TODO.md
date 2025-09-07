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
- [x] **COMPLETED: Tools Overhaul** - Enhanced schema system with 3-part structure:
  - JSON schema for input validation
  - JSON schema for output structure
  - OpenAI function definition for LLM integration
- [x] Document summarizer tool (deterministic, ≤1 LLM call)
- [x] Calculator tool for basic mathematical operations with expression parsing
- [x] Retriever tool for Milvus vector database queries
- [x] Enhanced JSON schema validation for tool inputs
- [x] Tool registry with discovery and definition support
- [x] Tool execution framework with LLM provider integration
- [ ] Web search tool (mock implementation) - Next priority

### 3. Agent Framework (`llm/agents/`)
- [x] **COMPLETED: Agent Framework Overhaul** - New modular architecture:
  - Tool-like sub-agent calling system with parallel/sequential execution
  - Schema-based input/output validation for all agents
  - Primary agent as main orchestrator with intelligent query analysis
  - Synthesis agent for multi-source aggregation
  - Comprehensive validation and error handling
- [x] Summary sub-agent with predefined I/O schemas and validation
- [x] Synthesis sub-agent for aggregating multiple agent outputs
- [x] Primary main-agent orchestrator with tool-like calling patterns
- [x] Agent execution pipeline with execution logging
- [x] Agent result formatting with comprehensive stats
- [x] Analyst and Researcher agents (boilerplate with NOT_IMPLEMENTED errors)
- [x] 4-file structure for each agent: definition.go, agent_name.go, agent_name_test.go, OVERVIEW.md

### 4. Services Layer (`llm/services/`)
- [ ] Basic agent manager for agent lifecycle
- [ ] Task orchestration framework
- [ ] Basic caching layer

### 5. API Layer (`api/`)
- [ ] POST /agents/{name} endpoint
- [ ] POST /sub-agents/{name} endpoint
- [x] POST /tools/{name} endpoint - Enhanced with tool definitions
- [ ] GET /tasks/{id} endpoint
- [x] Basic error handling and validation - Updated for tool definitions

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
- [x] Set up Go project structure
- [ ] Implement OpenAI client with Ollama fallback
- [x] **COMPLETED: Enhanced Tool Framework** with 3-part schema system
- [x] Build document summarizer tool with LLM integration
- [x] Build calculator tool with expression evaluation and parsing
- [x] Write comprehensive unit tests for tools
- [x] Implement tool registry with definition support

### Phase 2: Agent Core (Week 2)
- [x] **COMPLETED: Agent Framework Implementation**
- [x] Implement summary sub-agent with schema validation
- [x] Create synthesis sub-agent for aggregation
- [x] Create primary main-agent with tool-like calling
- [x] Build comprehensive test suites for all agents
- [x] Test agent execution pipeline with validation
- [x] Implement analyst/researcher boilerplate (NOT_IMPLEMENTED)
- [x] Complete 4-file structure for all agents

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
- [x] All tools execute deterministically
- [x] Calculator tool performs accurate mathematical calculations
- [x] **COMPLETED: Tools have 3-part schema system** (input, output, function definition)
- [x] Tool registry provides proper discovery with definitions
- [x] Enhanced API endpoints with tool definitions
- [x] **COMPLETED: Agents return results with comprehensive stats and validation**
- [x] **COMPLETED: Schema-based input/output validation for all agents**
- [x] **COMPLETED: Tool-like sub-agent calling system implemented**
- [x] **COMPLETED: Primary agent orchestration with intelligent query analysis**
- [x] **COMPLETED: Synthesis agent for multi-source aggregation**
- [x] **COMPLETED: Analyst/Researcher boilerplate with NOT_IMPLEMENTED errors**
- [x] **COMPLETED: 4-file structure for all agents (definition.go, agent_name.go, test, OVERVIEW.md)**
- [x] **COMPLETED: Comprehensive test suites for all agent components**
- [ ] API endpoints respond correctly (Next priority)
- [ ] Tests pass with localhost:11434 (Next priority)
- [ ] Error handling is robust across all agents
- [x] Code is well-documented with comprehensive OVERVIEW.md files

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




---

