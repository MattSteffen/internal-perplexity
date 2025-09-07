package api

import "time"

// ExecuteAgentRequest represents a request to execute an agent
type ExecuteAgentRequest struct {
	Input   map[string]any `json:"input" binding:"required"`
	Context map[string]any `json:"context,omitempty"`
	Model   string         `json:"model,omitempty"`
	Async   bool           `json:"async,omitempty"`
	Timeout time.Duration  `json:"timeout,omitempty"`
}

// ExecuteSubAgentRequest represents a request to execute a sub-agent
type ExecuteSubAgentRequest struct {
	Input      map[string]any `json:"input" binding:"required"`
	Context    map[string]any `json:"context,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
	Model      string         `json:"model,omitempty"`
	Async      bool           `json:"async,omitempty"`
	Timeout    time.Duration  `json:"timeout,omitempty"`
}

// ExecuteToolRequest represents a request to execute a tool
type ExecuteToolRequest struct {
	Input      map[string]any `json:"input" binding:"required"`
	Context    map[string]any `json:"context,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
	Model      string         `json:"model,omitempty"`
	Async      bool           `json:"async,omitempty"`
	Timeout    time.Duration  `json:"timeout,omitempty"`
}

// AgentResponse represents the response from agent execution
type AgentResponse struct {
	Success      bool   `json:"success"`
	Result       any    `json:"result,omitempty"`
	Stats        any    `json:"stats,omitempty"`
	Error        string `json:"error,omitempty"`
	ExecutionLog any    `json:"execution_log,omitempty"` // Log of execution steps
	TokensUsed   int    `json:"tokens_used,omitempty"`   // Token usage count
	Duration     string `json:"duration,omitempty"`      // Execution duration
}

// SubAgentResponse represents the response from sub-agent execution
type SubAgentResponse struct {
	Success    bool   `json:"success"`
	Result     any    `json:"result,omitempty"`
	Stats      any    `json:"stats,omitempty"`
	Error      string `json:"error,omitempty"`
	TokensUsed int    `json:"tokens_used,omitempty"` // Token usage count
	Duration   string `json:"duration,omitempty"`    // Execution duration
}

// ToolResponse represents the response from tool execution
type ToolResponse struct {
	Success bool   `json:"success"`
	Output  any    `json:"output,omitempty"`
	Error   string `json:"error,omitempty"`
	Stats   any    `json:"stats,omitempty"`
}

// TaskStatusResponse represents a task status response
type TaskStatusResponse struct {
	TaskID   string     `json:"task_id"`
	Status   string     `json:"status"`
	Result   any        `json:"result,omitempty"`
	Error    string     `json:"error,omitempty"`
	Created  time.Time  `json:"created"`
	Updated  time.Time  `json:"updated"`
	Executed *time.Time `json:"executed,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error   string         `json:"error"`
	Code    string         `json:"code,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}
