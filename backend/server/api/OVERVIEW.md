# API Package

REST API endpoints for agent orchestration and tool execution following MCP patterns.

## Endpoint Design

### Design Principles
- **RESTful**: Standard HTTP methods and status codes using net/http
- **MCP Inspired**: Tool execution similar to Model Context Protocol
- **Stateless**: No server-side session state
- **Async Support**: Background execution for long-running tasks

## Core Endpoints

### Agent Endpoints

#### POST /agents/{agent-name}
Execute a main agent with full orchestration capabilities.

**Request:**
```json
{
  "input": {
    "query": "Research AI agent systems",
    "context": {},
    "options": {}
  },
  "async": false,
}
```

**Response:**
```json
{
  "task_id": "task_123",
  "status": "completed",
  "result": {
    "content": "...",
    "metadata": {
      "tokens_used": 1500,
      "duration_ms": 2500,
      "sub_agents_used": ["researcher", "analyst"]
    }
  }
}
```


### Sub-Agent Endpoints

#### POST /sub-agents/{sub-agent-name}
Execute a specialized sub-agent with tool access.

**Request:**
```json
{
  "input": {
    "task": "Analyze this dataset",
    "data": {...},
    "parameters": {...}
  },
  "tools": ["data_analyzer", "statistical_tools"]
}
```

**Response:**
```json
{
  "success": true,
  "result": {...},
  "stats": {
    "tokens_used": 800,
    "duration_ms": 1200,
    "tools_used": ["data_analyzer"],
    "success_rate": 1.0
  }
}
```

### Tool Endpoints

#### POST /tools/{tool-name}
Execute a deterministic tool with predictable I/O.

**Request:**
```json
{
  "input": {
    "content": "Document to summarize",
    "max_length": 200,
    "format": "executive"
  }
}
```

**Response:**
```json
{
  "success": true,
  "output": {
    "summary": "Executive summary...",
    "word_count": 45,
    "compression_ratio": 0.15
  },
  "execution_time_ms": 150
}
```

#### GET /tools
List available tools with schemas.

**Response:**
```json
{
  "tools": [
    {
      "name": "document_summarizer",
      "description": "Summarize documents with configurable length",
      "schema": {
        "input": {...},
        "output": {...}
      }
    }
  ]
}
```

### Task Management Endpoints

#### GET /tasks/{task-id}
Get task status and results.

**Response:**
```json
{
  "task_id": "task_123",
  "status": "running|completed|failed",
  "progress": 0.75,
  "result": {...},
  "error": null,
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:02:30Z"
}
```

#### POST /tasks/{task-id}/cancel
Cancel a running task.

**Response:**
```json
{
  "success": true,
  "message": "Task cancelled successfully"
}
```

## Implementation Structure

### Handler Structure
```
api/
├── handlers/
│   ├── agents.go       # Agent execution handlers
│   ├── subagents.go    # Sub-agent handlers
│   ├── tools.go        # Tool execution handlers
│   ├── tasks.go        # Task management handlers
│   └── middleware.go   # Common middleware
├── routes/
│   └── router.go       # Route definitions
├── models/
│   ├── requests.go     # Request DTOs
│   ├── responses.go    # Response DTOs
│   └── errors.go       # Error types
└── server/
    └── server.go       # HTTP server setup
```

### Handler Examples

#### Agent Handler
```go
func (h *AgentHandler) ExecuteAgent(w http.ResponseWriter, r *http.Request) {
    agentName := strings.TrimPrefix(r.URL.Path, "/agents/")
    agentName = strings.Split(agentName, "/")[0]

    var req ExecuteAgentRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        h.writeJSONError(w, http.StatusBadRequest, "Invalid request", err.Error())
        return
    }

    // Get agent from manager
    agent, err := h.agentManager.GetAgent(r.Context(), agentName)
    if err != nil {
        h.writeJSONError(w, http.StatusNotFound, "Agent not found", err.Error())
        return
    }

    // Execute synchronously
    result, err := agent.Execute(r.Context(), req.Input)
    if err != nil {
        h.writeJSONError(w, http.StatusInternalServerError, "Agent execution failed", err.Error())
        return
    }

    response := AgentResponse{
        Result: result.Content,
        Stats:  result,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

#### Tool Handler
```go
func (h *ToolHandler) ExecuteTool(w http.ResponseWriter, r *http.Request) {
    toolName := strings.TrimPrefix(r.URL.Path, "/tools/")
    toolName = strings.Split(toolName, "/")[0]

    var req ExecuteToolRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        h.writeJSONError(w, http.StatusBadRequest, "Invalid request", err.Error())
        return
    }

    // Get tool from registry
    tool, err := h.toolRegistry.Get(toolName)
    if err != nil {
        h.writeJSONError(w, http.StatusNotFound, "Tool not found", err.Error())
        return
    }

    // Execute tool
    input := &ToolInput{
        Name: toolName,
        Data: req.Input,
    }

    result, err := tool.Execute(r.Context(), input)
    if err != nil {
        h.writeJSONError(w, http.StatusInternalServerError, "Tool execution failed", err.Error())
        return
    }

    response := ToolResponse{
        Success: result.Success,
        Output:  result.Data,
        Error:   result.Error,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

## Middleware

### Request Logging
```go
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        // Create a response writer wrapper to capture status code
        wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
        next.ServeHTTP(wrapped, r)

        log.Printf("%s %s %d %v",
            r.Method,
            r.URL.Path,
            wrapped.statusCode,
            time.Since(start))
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

## Error Handling

### Standardized Error Responses
```go
type ErrorResponse struct {
    Error   string                 `json:"error"`
    Code    string                 `json:"code,omitempty"`
    Details map[string]interface{} `json:"details,omitempty"`
}

type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
    Value   interface{} `json:"value,omitempty"`
}
```

### Error Codes
- `INVALID_REQUEST`: Malformed request
- `AGENT_NOT_FOUND`: Agent doesn't exist
- `TOOL_EXECUTION_FAILED`: Tool execution error
- `INTERNAL_ERROR`: Server error

## Testing

### Unit Tests
```go
func TestExecuteTool(t *testing.T) {
    // Mock tool registry
    registry := &mockToolRegistry{}

    // Test handler
    handler := &ToolHandler{registry: registry}

    // Test cases
    tests := []struct {
        name     string
        toolName string
        input    interface{}
        wantCode int
    }{
        {"valid tool", "summarizer", validInput, 200},
        {"invalid tool", "nonexistent", validInput, 404},
        {"invalid input", "summarizer", invalidInput, 400},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

### Integration Tests
```go
func TestAgentExecutionFlow(t *testing.T) {
    // Setup test server
    server := setupTestServer()

    // Test full execution flow
    req := ExecuteAgentRequest{
        Input: map[string]interface{}{
            "task": "summarize_documents",
            "documents": []string{"content1", "content2"},
        },
    }
    resp, err := server.client.Post("/agents/primary", req)

    assert.NoError(t, err)
    assert.Equal(t, 200, resp.StatusCode)
    // Validate response structure
}
```
