# API Layer - MVP Tasks

## Overview
Implement REST API endpoints for agent orchestration following MCP-inspired patterns. Focus on core endpoints needed for MVP functionality.

## Core Endpoints

### 1. Agent Endpoints
- [ ] POST `/agents/{agent-name}` - Execute main agent
- [ ] Add request validation for agent execution
- [ ] Implement response formatting with stats
- [ ] Add error handling for agent failures

```go
// Handler structure
type AgentHandler struct {
    agentManager *AgentManager
    validator    *RequestValidator
}

func (h *AgentHandler) ExecuteAgent(w http.ResponseWriter, r *http.Request) {
    agentName := strings.TrimPrefix(r.URL.Path, "/api/v1/agents/")
    agentName = strings.Split(agentName, "/")[0]

    var req ExecuteAgentRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        h.writeJSONError(w, http.StatusBadRequest, "Invalid request", err.Error())
        return
    }

    // Validate request
    if err := h.validator.ValidateAgentRequest(agentName, &req); err != nil {
        h.writeJSONError(w, http.StatusBadRequest, "Validation failed", err.Error())
        return
    }

    // Execute agent
    task := &Task{
        AgentName: agentName,
        Input:     req.Input,
        Context:   req.Context,
    }

    result, err := h.agentManager.ExecuteTask(r.Context(), task)
    if err != nil {
        h.writeJSONError(w, http.StatusInternalServerError, "Agent execution failed", err.Error())
        return
    }

    response := AgentResponse{
        Success: true,
        Result:  result.Result,
        Stats:   result.Stats,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

### 2. Sub-Agent Endpoints
- [ ] POST `/sub-agents/{sub-agent-name}` - Execute sub-agent
- [ ] Add sub-agent specific validation
- [ ] Implement sub-agent response formatting
- [ ] Add tool execution context

### 3. Tool Endpoints
- [ ] POST `/tools/{tool-name}` - Execute tool directly
- [ ] GET `/tools` - List available tools
- [ ] Add tool schema validation
- [ ] Implement tool result formatting

```go
func (h *ToolHandler) ExecuteTool(c *gin.Context) {
    toolName := c.Param("tool-name")

    var req ExecuteToolRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, ErrorResponse{Error: err.Error()})
        return
    }

    // Get tool from registry
    tool, err := h.toolRegistry.Get(toolName)
    if err != nil {
        c.JSON(404, ErrorResponse{Error: "Tool not found"})
        return
    }

    // Execute tool
    input := &ToolInput{
        Name: toolName,
        Data: req.Input,
    }

    result, err := tool.Execute(c.Request.Context(), input)
    if err != nil {
        c.JSON(500, ErrorResponse{Error: "Tool execution failed: " + err.Error()})
        return
    }

    c.JSON(200, ToolResponse{
        Success: result.Success,
        Output:  result.Data,
        Error:   result.Error,
    })
}
```

## Task Management

### 4. Task Endpoints
- [ ] POST `/tasks` - Create async task
- [ ] GET `/tasks/{task-id}` - Get task status
- [ ] POST `/tasks/{task-id}/cancel` - Cancel task
- [ ] Implement task status polling

```go
func (h *TaskHandler) CreateTask(c *gin.Context) {
    var req CreateTaskRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, ErrorResponse{Error: err.Error()})
        return
    }

    // Create task
    task := &Task{
        AgentName: req.AgentName,
        Input:     req.Input,
        Async:     true,
    }

    taskID, err := h.taskOrchestrator.ExecuteAsync(c.Request.Context(), task)
    if err != nil {
        c.JSON(500, ErrorResponse{Error: "Task creation failed: " + err.Error()})
        return
    }

    c.JSON(202, TaskCreatedResponse{
        TaskID: taskID,
        Status: "pending",
    })
}

func (h *TaskHandler) GetTaskStatus(c *gin.Context) {
    taskID := c.Param("task-id")

    task, err := h.taskStore.Get(taskID)
    if err != nil {
        c.JSON(404, ErrorResponse{Error: "Task not found"})
        return
    }

    c.JSON(200, TaskStatusResponse{
        TaskID:  task.ID,
        Status:  string(task.Status),
        Result:  task.Result,
        Created: task.CreatedAt,
        Updated: task.UpdatedAt,
    })
}
```

## Request/Response Types

### 5. API Types
- [ ] Create request DTOs for all endpoints
- [ ] Implement response DTOs with proper JSON tags
- [ ] Add validation tags for request validation
- [ ] Create error response structures

```go
type ExecuteAgentRequest struct {
    Input    map[string]interface{} `json:"input" binding:"required"`
    Context  map[string]interface{} `json:"context,omitempty"`
    Async    bool                   `json:"async,omitempty"`
    Timeout  time.Duration          `json:"timeout,omitempty"`
}

type AgentResponse struct {
    Success bool        `json:"success"`
    Result  interface{} `json:"result,omitempty"`
    Stats   *AgentStats `json:"stats,omitempty"`
    Error   string      `json:"error,omitempty"`
}

type ErrorResponse struct {
    Error   string                 `json:"error"`
    Code    string                 `json:"code,omitempty"`
    Details map[string]interface{} `json:"details,omitempty"`
}
```

## Middleware

### 6. API Middleware
- [ ] Add request logging middleware
- [ ] Implement CORS middleware
- [ ] Create authentication middleware (basic)
- [ ] Add rate limiting middleware

```go
func LoggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        c.Next()

        log.Printf("%s %s %d %v",
            c.Request.Method,
            c.Request.URL.Path,
            c.Writer.Status(),
            time.Since(start))
    }
}

func CORSMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("Access-Control-Allow-Origin", "*")
        c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }

        c.Next()
    }
}
```

## Testing Tasks

### 7. API Tests
- [ ] Test agent execution endpoints
- [ ] Test tool execution endpoints
- [ ] Test task management endpoints
- [ ] Test error handling and validation

```go
func TestExecuteAgentEndpoint(t *testing.T) {
    // Setup test server
    router := setupTestRouter()

    // Test request
    req := ExecuteAgentRequest{
        Input: map[string]interface{}{
            "query": "test query",
        },
    }

    w := httptest.NewRecorder()
    reqBody, _ := json.Marshal(req)
    httpReq, _ := http.NewRequest("POST", "/agents/researcher", bytes.NewBuffer(reqBody))
    httpReq.Header.Set("Content-Type", "application/json")

    router.ServeHTTP(w, httpReq)

    assert.Equal(t, 200, w.Code)

    var resp AgentResponse
    json.Unmarshal(w.Body.Bytes(), &resp)
    assert.True(t, resp.Success)
}
```

## Implementation Priority

### Phase 1: Core Endpoints
1. [ ] Implement agent execution endpoints
2. [ ] Add tool execution endpoints
3. [ ] Create basic request/response types
4. [ ] Test endpoint functionality

### Phase 2: Task Management
1. [ ] Add async task creation
2. [ ] Implement task status endpoints
3. [ ] Add task cancellation
4. [ ] Test task lifecycle

### Phase 3: Advanced Features
1. [ ] Add comprehensive middleware
2. [ ] Implement authentication
3. [ ] Add rate limiting
4. [ ] Comprehensive API testing

## Configuration

### 8. API Configuration
- [ ] Add API server configuration
- [ ] Configure endpoint timeouts
- [ ] Add CORS settings
- [ ] Configure rate limits

```yaml
api:
  port: 8080
  host: "0.0.0.0"
  cors:
    enabled: true
    origins: ["*"]
  timeouts:
    agent_execution: 300s
    tool_execution: 60s
```

## Success Criteria
- [ ] All core endpoints respond correctly
- [ ] Request validation works properly
- [ ] Error responses are consistent
- [ ] API integrates with agent system
- [ ] Endpoints work with localhost:11434 testing

## Files to Create
- `api/handlers/agents.go`
- `api/handlers/tools.go`
- `api/handlers/tasks.go`
- `api/types/requests.go`
- `api/types/responses.go`
- `api/middleware/logging.go`
- `api/middleware/cors.go`
- `api/handlers/agents_test.go`
