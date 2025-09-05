package agents

import (
	"context"
	"time"
)

// AgentInput represents input data for agent execution
type AgentInput struct {
	Data       map[string]interface{} `json:"data"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

// AgentResult represents the result of agent execution
type AgentResult struct {
	Content    interface{}            `json:"content"`
	Success    bool                   `json:"success"`
	TokensUsed interface{}            `json:"tokens_used,omitempty"`
	Duration   time.Duration          `json:"duration"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// AgentStats tracks agent execution statistics
type AgentStats struct {
	TotalExecutions int           `json:"total_executions"`
	AverageDuration time.Duration `json:"average_duration"`
	SuccessRate     float64       `json:"success_rate"`
	TotalTokens     int           `json:"total_tokens"`
}

// Capability represents an agent capability
type Capability struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Agent defines the interface that all agents must implement
type Agent interface {
	Execute(ctx context.Context, input *AgentInput) (*AgentResult, error)
	GetCapabilities() []Capability
	GetStats() AgentStats
}
