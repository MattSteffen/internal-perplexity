package agentmanager

import (
	"context"
	"fmt"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// Task represents a task to be executed
type Task struct {
	ID         string                 `json:"id"`
	AgentName  string                 `json:"agent_name"`
	Input      map[string]interface{} `json:"input"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Status     string                 `json:"status"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// TaskResult represents the result of task execution
type TaskResult struct {
	TaskID     string      `json:"task_id"`
	Status     string      `json:"status"`
	Result     interface{} `json:"result,omitempty"`
	Error      string      `json:"error,omitempty"`
	ExecutedAt time.Time   `json:"executed_at"`
}

// AgentManager manages agent lifecycle and task execution
type AgentManager struct {
	agents map[string]agents.Agent
}

// NewAgentManager creates a new agent manager
func NewAgentManager() *AgentManager {
	return &AgentManager{
		agents: make(map[string]agents.Agent),
	}
}

// RegisterAgent registers an agent with the manager
func (am *AgentManager) RegisterAgent(ctx context.Context, name string, agent agents.Agent) error {
	am.agents[name] = agent
	return nil
}

// GetAgent retrieves an agent by name
func (am *AgentManager) GetAgent(ctx context.Context, name string) (agents.Agent, error) {
	agent, exists := am.agents[name]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", name)
	}
	return agent, nil
}

// ListAgents returns all registered agents
func (am *AgentManager) ListAgents(ctx context.Context) (map[string]agents.Agent, error) {
	return am.agents, nil
}

// ExecuteTask executes a task using the specified agent
func (am *AgentManager) ExecuteTask(ctx context.Context, task *Task, llmProvider shared.LLMProvider) (*TaskResult, error) {
	agent, exists := am.agents[task.AgentName]
	if !exists {
		return &TaskResult{
			TaskID: task.ID,
			Status: "failed",
			Error:  fmt.Sprintf("agent not found: %s", task.AgentName),
		}, nil
	}

	input := &agents.AgentInput{
		Data:       task.Input,
		Context:    task.Context,
		Parameters: task.Parameters,
	}

	result, err := agent.Execute(ctx, input, llmProvider)
	if err != nil {
		return &TaskResult{
			TaskID:     task.ID,
			Status:     "failed",
			Error:      err.Error(),
			ExecutedAt: time.Now(),
		}, nil
	}

	return &TaskResult{
		TaskID:     task.ID,
		Status:     "completed",
		Result:     result.Content,
		ExecutedAt: time.Now(),
	}, nil
}
