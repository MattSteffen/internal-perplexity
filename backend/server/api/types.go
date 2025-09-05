package api

import "time"

// ExecuteAgentRequest represents a request to execute an agent
type ExecuteAgentRequest struct {
	Input   map[string]interface{} `json:"input" binding:"required"`
	Context map[string]interface{} `json:"context,omitempty"`
	Async   bool                   `json:"async,omitempty"`
	Timeout time.Duration          `json:"timeout,omitempty"`
}

// ExecuteSubAgentRequest represents a request to execute a sub-agent
type ExecuteSubAgentRequest struct {
	Input   map[string]interface{} `json:"input" binding:"required"`
	Context map[string]interface{} `json:"context,omitempty"`
	Async   bool                   `json:"async,omitempty"`
	Timeout time.Duration          `json:"timeout,omitempty"`
}

// ExecuteToolRequest represents a request to execute a tool
type ExecuteToolRequest struct {
	Input   map[string]interface{} `json:"input" binding:"required"`
	Timeout time.Duration          `json:"timeout,omitempty"`
}

// AgentResponse represents the response from agent execution
type AgentResponse struct {
	Success bool        `json:"success"`
	Result  interface{} `json:"result,omitempty"`
	Stats   interface{} `json:"stats,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// SubAgentResponse represents the response from sub-agent execution
type SubAgentResponse struct {
	Success bool        `json:"success"`
	Result  interface{} `json:"result,omitempty"`
	Stats   interface{} `json:"stats,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// ToolResponse represents the response from tool execution
type ToolResponse struct {
	Success bool        `json:"success"`
	Output  interface{} `json:"output,omitempty"`
	Error   string      `json:"error,omitempty"`
	Stats   interface{} `json:"stats,omitempty"`
}

// TaskStatusResponse represents a task status response
type TaskStatusResponse struct {
	TaskID   string      `json:"task_id"`
	Status   string      `json:"status"`
	Result   interface{} `json:"result,omitempty"`
	Error    string      `json:"error,omitempty"`
	Created  time.Time   `json:"created"`
	Updated  time.Time   `json:"updated"`
	Executed *time.Time  `json:"executed,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error   string                 `json:"error"`
	Code    string                 `json:"code,omitempty"`
	Details map[string]interface{} `json:"details,omitempty"`
}
