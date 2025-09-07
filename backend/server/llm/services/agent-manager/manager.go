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
	Query      string                 `json:"query,omitempty"`      // User query for the task
	Input      map[string]interface{} `json:"input"`                // Task-specific input data
	Context    map[string]interface{} `json:"context,omitempty"`    // Additional context
	Parameters map[string]interface{} `json:"parameters,omitempty"` // Task parameters
	Status     string                 `json:"status"`               // pending, running, completed, failed
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
	Priority   int                    `json:"priority"` // Execution priority
}

// TaskResult represents the result of task execution
type TaskResult struct {
	TaskID     string                 `json:"task_id"`
	Status     string                 `json:"status"`
	Result     interface{}            `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
	ExecutedAt time.Time              `json:"executed_at"`
	Duration   time.Duration          `json:"duration"`
	TokensUsed interface{}            `json:"tokens_used,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	AgentStats agents.AgentStats      `json:"agent_stats,omitempty"`
}

// AgentManager manages agent lifecycle and task execution
type AgentManager struct {
	agents      map[string]agents.Agent
	taskHistory map[string]*TaskResult
	stats       ManagerStats
}

// ManagerStats tracks manager-level statistics
type ManagerStats struct {
	TotalTasks      int           `json:"total_tasks"`
	CompletedTasks  int           `json:"completed_tasks"`
	FailedTasks     int           `json:"failed_tasks"`
	AverageDuration time.Duration `json:"average_duration"`
	TotalTokensUsed int           `json:"total_tokens_used"`
	ActiveAgents    int           `json:"active_agents"`
}

// NewAgentManager creates a new agent manager
func NewAgentManager() *AgentManager {
	return &AgentManager{
		agents:      make(map[string]agents.Agent),
		taskHistory: make(map[string]*TaskResult),
		stats: ManagerStats{
			ActiveAgents: 0,
		},
	}
}

// NewTask creates a new task with default values
func NewTask(id, agentName, query string) *Task {
	now := time.Now()
	return &Task{
		ID:        id,
		AgentName: agentName,
		Query:     query,
		Status:    "pending",
		CreatedAt: now,
		UpdatedAt: now,
		Priority:  0,
	}
}

// GetStats returns the manager's statistics
func (am *AgentManager) GetStats() ManagerStats {
	am.stats.ActiveAgents = len(am.agents)
	return am.stats
}

// GetTaskHistory returns the execution history for a specific task
func (am *AgentManager) GetTaskHistory(taskID string) (*TaskResult, error) {
	result, exists := am.taskHistory[taskID]
	if !exists {
		return nil, fmt.Errorf("task history not found for task: %s", taskID)
	}
	return result, nil
}

// ListTaskHistory returns all task execution history
func (am *AgentManager) ListTaskHistory() map[string]*TaskResult {
	return am.taskHistory
}

// RegisterAgent registers an agent with the manager
func (am *AgentManager) RegisterAgent(ctx context.Context, name string, agent agents.Agent) error {
	if agent == nil {
		return fmt.Errorf("cannot register nil agent")
	}
	am.agents[name] = agent
	return nil
}

// UnregisterAgent removes an agent from the manager
func (am *AgentManager) UnregisterAgent(ctx context.Context, name string) error {
	if _, exists := am.agents[name]; !exists {
		return fmt.Errorf("agent not found: %s", name)
	}
	delete(am.agents, name)
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

// ListAgentNames returns the names of all registered agents
func (am *AgentManager) ListAgentNames(ctx context.Context) ([]string, error) {
	names := make([]string, 0, len(am.agents))
	for name := range am.agents {
		names = append(names, name)
	}
	return names, nil
}

// ExecuteTask executes a task using the specified agent
func (am *AgentManager) ExecuteTask(ctx context.Context, task *Task, llmProvider shared.LLMProvider) (*TaskResult, error) {
	start := time.Now()
	am.stats.TotalTasks++

	// Update task status to running
	task.Status = "running"
	task.UpdatedAt = start

	// Initialize result
	taskResult := &TaskResult{
		TaskID:     task.ID,
		Status:     "running",
		ExecutedAt: start,
		Metadata:   make(map[string]interface{}),
	}

	// Check if agent exists
	agent, exists := am.agents[task.AgentName]
	if !exists {
		task.Status = "failed"
		task.UpdatedAt = time.Now()
		taskResult.Status = "failed"
		taskResult.Error = fmt.Sprintf("agent not found: %s", task.AgentName)
		taskResult.Duration = time.Since(start)
		taskResult.Metadata["error_type"] = "AGENT_NOT_FOUND"
		taskResult.Metadata["requested_agent"] = task.AgentName

		am.taskHistory[task.ID] = taskResult
		am.stats.FailedTasks++
		return taskResult, nil
	}

	// Validate LLM provider
	if llmProvider == nil {
		task.Status = "failed"
		task.UpdatedAt = time.Now()
		taskResult.Status = "failed"
		taskResult.Error = "LLM provider is required but not provided"
		taskResult.Duration = time.Since(start)
		taskResult.Metadata["error_type"] = "MISSING_LLM_PROVIDER"

		am.taskHistory[task.ID] = taskResult
		am.stats.FailedTasks++
		return taskResult, nil
	}

	// Prepare agent input - handle both query-based and data-based inputs
	var agentInput *agents.AgentInput
	if task.Query != "" {
		// Query-based input (for intelligent agents)
		agentInput = &agents.AgentInput{
			Query:      task.Query,
			Data:       task.Input,
			Context:    task.Context,
			Parameters: task.Parameters,
		}
	} else {
		// Data-based input (for specialized agents)
		agentInput = &agents.AgentInput{
			Data:       task.Input,
			Context:    task.Context,
			Parameters: task.Parameters,
		}
	}

	// Validate input using agent's validation method if available
	if validator, ok := agent.(interface {
		ValidateInput(*agents.AgentInput) error
	}); ok {
		if err := validator.ValidateInput(agentInput); err != nil {
			task.Status = "failed"
			task.UpdatedAt = time.Now()
			taskResult.Status = "failed"
			taskResult.Error = fmt.Sprintf("input validation failed: %v", err)
			taskResult.Duration = time.Since(start)
			taskResult.Metadata["error_type"] = "VALIDATION_ERROR"
			taskResult.Metadata["validation_error"] = err.Error()

			am.taskHistory[task.ID] = taskResult
			am.stats.FailedTasks++
			return taskResult, nil
		}
	}

	// Execute the agent
	agentResult, err := agent.Execute(ctx, agentInput, llmProvider)
	if err != nil {
		task.Status = "failed"
		task.UpdatedAt = time.Now()
		taskResult.Status = "failed"
		taskResult.Error = fmt.Sprintf("agent execution failed: %v", err)
		taskResult.Duration = time.Since(start)
		taskResult.Metadata["error_type"] = "EXECUTION_ERROR"
		taskResult.Metadata["execution_error"] = err.Error()

		am.taskHistory[task.ID] = taskResult
		am.stats.FailedTasks++
		return taskResult, nil
	}

	// Validate output using agent's validation method if available
	if validator, ok := agent.(interface {
		ValidateOutput(*agents.AgentResult) error
	}); ok {
		if err := validator.ValidateOutput(agentResult); err != nil {
			task.Status = "failed"
			task.UpdatedAt = time.Now()
			taskResult.Status = "failed"
			taskResult.Error = fmt.Sprintf("output validation failed: %v", err)
			taskResult.Duration = time.Since(start)
			taskResult.Metadata["error_type"] = "VALIDATION_ERROR"
			taskResult.Metadata["validation_error"] = err.Error()

			am.taskHistory[task.ID] = taskResult
			am.stats.FailedTasks++
			return taskResult, nil
		}
	}

	// Update task and result on successful completion
	task.Status = "completed"
	task.UpdatedAt = time.Now()
	taskResult.Status = "completed"
	taskResult.Result = agentResult.Content
	taskResult.Duration = agentResult.Duration
	taskResult.TokensUsed = agentResult.TokensUsed
	taskResult.Metadata = agentResult.Metadata

	// Get agent stats if available
	if statsGetter, ok := agent.(interface{ GetStats() agents.AgentStats }); ok {
		taskResult.AgentStats = statsGetter.GetStats()
	}

	// Update manager statistics
	am.stats.CompletedTasks++

	// Update average duration
	totalDuration := am.stats.AverageDuration*time.Duration(am.stats.CompletedTasks-1) + taskResult.Duration
	am.stats.AverageDuration = totalDuration / time.Duration(am.stats.CompletedTasks)

	// Update total tokens used
	if tokens, ok := agentResult.TokensUsed.(int); ok {
		am.stats.TotalTokensUsed += tokens
	}

	// Store in history
	am.taskHistory[task.ID] = taskResult

	return taskResult, nil
}

// ExecuteTaskAsync executes a task asynchronously
func (am *AgentManager) ExecuteTaskAsync(ctx context.Context, task *Task, llmProvider shared.LLMProvider) error {
	go func() {
		_, err := am.ExecuteTask(ctx, task, llmProvider)
		if err != nil {
			// Log error in async execution
			fmt.Printf("Async task execution failed: %v\n", err)
		}
	}()
	return nil
}

// ValidateTask validates a task before execution
func (am *AgentManager) ValidateTask(ctx context.Context, task *Task) error {
	if task == nil {
		return fmt.Errorf("task cannot be nil")
	}

	if task.ID == "" {
		return fmt.Errorf("task ID is required")
	}

	if task.AgentName == "" {
		return fmt.Errorf("agent name is required")
	}

	// Check if agent exists
	_, exists := am.agents[task.AgentName]
	if !exists {
		return fmt.Errorf("agent not found: %s", task.AgentName)
	}

	// Validate that we have either a query or input data
	if task.Query == "" && len(task.Input) == 0 {
		return fmt.Errorf("task must have either a query or input data")
	}

	return nil
}
