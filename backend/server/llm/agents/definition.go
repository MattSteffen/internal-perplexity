package agents

import (
	"context"
	"fmt"
	"time"

	"internal-perplexity/server/llm/providers/shared"
)

// ExecutionPattern defines how tasks should be executed
type ExecutionPattern string

const (
	PatternSequential ExecutionPattern = "sequential" // Execute tasks one after another
	PatternParallel   ExecutionPattern = "parallel"   // Execute tasks in parallel
	PatternMapReduce  ExecutionPattern = "map_reduce" // Map tasks then reduce results
	PatternDirect     ExecutionPattern = "direct"     // Direct response without tasks
)

// TaskType defines the type of task to execute
type TaskType string

const (
	TaskTypeTool     TaskType = "tool"     // Execute a tool
	TaskTypeSubAgent TaskType = "subagent" // Execute a sub-agent
	TaskTypeResponse TaskType = "response" // Direct response to user
)

// Task represents a dynamically created task
type Task struct {
	ID          string                 `json:"id"`
	Type        TaskType               `json:"type"`
	Name        string                 `json:"name"` // Tool name, sub-agent name, or "response"
	Input       map[string]interface{} `json:"input"`
	Description string                 `json:"description"` // Human-readable description
	Priority    int                    `json:"priority"`    // Execution priority (lower = higher priority)
	DependsOn   []string               `json:"depends_on"`  // Task IDs this task depends on
}

// ExecutionPlan represents the complete plan for task execution
type ExecutionPlan struct {
	Tasks           []Task           `json:"tasks"`
	Pattern         ExecutionPattern `json:"pattern"`
	FinalResponse   string           `json:"final_response,omitempty"` // For direct responses
	Reasoning       string           `json:"reasoning"`                // Why this plan was chosen
	EstimatedTokens int              `json:"estimated_tokens"`
}

// SystemPrompt manages agent system prompts
type SystemPrompt struct {
	BasePrompt   string            `json:"base_prompt"`
	Capabilities []string          `json:"capabilities"`
	Tools        []string          `json:"tools"`
	SubAgents    []string          `json:"sub_agents"`
	Examples     []string          `json:"examples"`
	Constraints  map[string]string `json:"constraints"`
}

// AgentInput represents input data for agent execution
type AgentInput struct {
	Query      string         `json:"query"`          // User's natural language query
	Data       map[string]any `json:"data,omitempty"` // Additional structured data
	Context    map[string]any `json:"context,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
}

// AgentResult represents the result of agent execution
type AgentResult struct {
	Content      any             `json:"content"`
	Success      bool            `json:"success"`
	TokensUsed   any             `json:"tokens_used,omitempty"`
	Duration     time.Duration   `json:"duration"`
	Metadata     map[string]any  `json:"metadata,omitempty"`
	ExecutionLog []ExecutionStep `json:"execution_log,omitempty"` // Log of what was executed
}

// ExecutionStep represents a step in the execution process
type ExecutionStep struct {
	Type        string                 `json:"type"` // "task_creation", "task_execution", "tool_call", etc.
	Description string                 `json:"description"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data,omitempty"`
	Duration    time.Duration          `json:"duration,omitempty"`
}

// AgentStats tracks agent execution statistics
type AgentStats struct {
	TotalExecutions int           `json:"total_executions"`
	AverageDuration time.Duration `json:"average_duration"`
	SuccessRate     float64       `json:"success_rate"`
	TotalTokens     int           `json:"total_tokens"`
	TasksCreated    int           `json:"tasks_created"`
	ToolsCalled     int           `json:"tools_called"`
	SubAgentsUsed   int           `json:"sub_agents_used"`
}

// Capability represents an agent capability
type Capability struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Agent defines the interface that all agents must implement
type Agent interface {
	Execute(ctx context.Context, input *AgentInput, llmProvider shared.LLMProvider) (*AgentResult, error)
	GetCapabilities() []Capability
	GetStats() AgentStats
	GetSystemPrompt() *SystemPrompt
}

// IntelligentAgent extends Agent with task creation capabilities
type IntelligentAgent interface {
	Agent
	CreateExecutionPlan(ctx context.Context, input *AgentInput, llmProvider shared.LLMProvider) (*ExecutionPlan, error)
	ExecutePlan(ctx context.Context, plan *ExecutionPlan, llmProvider shared.LLMProvider) (*AgentResult, error)
}

// SubAgentCall represents a call to a sub-agent or tool with input/output specifications
type SubAgentCall struct {
	Name        string                 `json:"name"`        // Sub-agent or tool name
	Type        string                 `json:"type"`        // "subagent" or "tool"
	Input       map[string]interface{} `json:"input"`       // Input data for the sub-agent/tool
	OutputKey   string                 `json:"output_key"`  // Key to store result under
	Description string                 `json:"description"` // Human description
}

// ExecutionGroup represents a group of sub-agent calls (parallel execution)
type ExecutionGroup struct {
	Calls       []SubAgentCall `json:"calls"`       // Sub-agent calls in this group
	Description string         `json:"description"` // Description of this execution group
}

// SubAgentExecution represents the execution specification for sub-agents
type SubAgentExecution struct {
	Groups      []ExecutionGroup `json:"groups"`      // Groups to execute sequentially
	Description string           `json:"description"` // Overall description
}

// InputSchema defines the expected input structure for an agent
type InputSchema struct {
	Required []string               `json:"required"` // Required field names
	Optional []string               `json:"optional"` // Optional field names
	Types    map[string]string      `json:"types"`    // Field type specifications
	Examples map[string]interface{} `json:"examples"` // Example values
}

// OutputSchema defines the expected output structure for an agent
type OutputSchema struct {
	Type        string                 `json:"type"`        // Output type (string, object, array)
	Structure   map[string]interface{} `json:"structure"`   // Expected structure
	Examples    []interface{}          `json:"examples"`    // Example outputs
	Description string                 `json:"description"` // Output description
}

// AgentSchema defines the complete schema for an agent including I/O specifications
type AgentSchema struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Input       InputSchema  `json:"input"`
	Output      OutputSchema `json:"output"`
	Version     string       `json:"version"`
	Author      string       `json:"author"`
}

// ValidationError represents an input/output validation error
type ValidationError struct {
	Field   string      `json:"field"`
	Message string      `json:"message"`
	Code    string      `json:"code"`
	Value   interface{} `json:"value,omitempty"`
}

// NewValidationError creates a new validation error
func NewValidationError(field, message, code string, value interface{}) *ValidationError {
	return &ValidationError{
		Field:   field,
		Message: message,
		Code:    code,
		Value:   value,
	}
}

// Error implements the error interface
func (ve *ValidationError) Error() string {
	return fmt.Sprintf("validation error [%s]: %s (field: %s)", ve.Code, ve.Message, ve.Field)
}
