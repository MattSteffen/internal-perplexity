package agentmanager

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// MockAgent for testing
type MockAgent struct {
	name        string
	shouldFail  bool
	returnValue interface{}
	execCount   int
}

func (m *MockAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	m.execCount++
	if m.shouldFail {
		return &agents.AgentResult{
			Success: false,
			Content: nil,
			Metadata: map[string]interface{}{
				"error": "mock agent failure",
			},
		}, assert.AnError
	}

	return &agents.AgentResult{
		Success:    true,
		Content:    m.returnValue,
		Duration:   time.Millisecond * 100,
		TokensUsed: 50,
		Metadata: map[string]interface{}{
			"agent_name": m.name,
		},
	}, nil
}

func (m *MockAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{Name: "mock_capability", Description: "Mock capability for testing"},
	}
}

func (m *MockAgent) GetStats() agents.AgentStats {
	return agents.AgentStats{
		TotalExecutions: m.execCount,
		SuccessRate:     1.0,
	}
}

func (m *MockAgent) GetSystemPrompt() *agents.SystemPrompt {
	return &agents.SystemPrompt{
		BasePrompt: "Mock system prompt",
	}
}

// Mock LLM Provider for testing
type MockLLMProvider struct{}

func (m *MockLLMProvider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	return &shared.CompletionResponse{
		Content: "Mock response",
		Usage: shared.TokenUsage{
			TotalTokens: 25,
		},
	}, nil
}

func (m *MockLLMProvider) StreamComplete(ctx context.Context, req *shared.CompletionRequest) (<-chan *shared.StreamChunk, func(), error) {
	ch := make(chan *shared.StreamChunk, 1)
	ch <- &shared.StreamChunk{
		DeltaText: "Mock streaming response",
		Done:      true,
		Usage: &shared.TokenUsage{
			TotalTokens: 25,
		},
	}
	close(ch)
	return ch, func() {}, nil
}

func (m *MockLLMProvider) CountTokens(messages []shared.Message, model string) (int, error) {
	return 10, nil
}

func (m *MockLLMProvider) GetModelCapabilities(model string) shared.ModelCapabilities {
	return shared.ModelCapabilities{
		Streaming:         true,
		Tools:             false,
		ParallelToolCalls: false,
		JSONMode:          false,
		SystemMessage:     true,
		Vision:            false,
		MaxContextTokens:  128000,
	}
}

func (m *MockLLMProvider) GetSupportedModels() []shared.ModelInfo {
	return []shared.ModelInfo{}
}

func (m *MockLLMProvider) SupportsModel(model string) bool {
	return true
}

func (m *MockLLMProvider) Name() string {
	return "mock"
}

func TestNewAgentManager(t *testing.T) {
	manager := NewAgentManager()
	assert.NotNil(t, manager)
	assert.NotNil(t, manager.agents)
	assert.NotNil(t, manager.taskHistory)
	assert.Equal(t, 0, manager.stats.ActiveAgents)
}

func TestNewTask(t *testing.T) {
	task := NewTask("task-1", "summary", "test query")
	assert.NotNil(t, task)
	assert.Equal(t, "task-1", task.ID)
	assert.Equal(t, "summary", task.AgentName)
	assert.Equal(t, "test query", task.Query)
	assert.Equal(t, "pending", task.Status)
	assert.Equal(t, 0, task.Priority)
}

func TestAgentManager_RegisterAndGetAgent(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "test-agent"}

	// Register agent
	err := manager.RegisterAgent(context.Background(), "test-agent", mockAgent)
	assert.NoError(t, err)

	// Get agent
	retrieved, err := manager.GetAgent(context.Background(), "test-agent")
	assert.NoError(t, err)
	assert.Equal(t, mockAgent, retrieved)

	// Get non-existent agent
	_, err = manager.GetAgent(context.Background(), "non-existent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "agent not found")
}

func TestAgentManager_UnregisterAgent(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "test-agent"}

	// Register agent
	err := manager.RegisterAgent(context.Background(), "test-agent", mockAgent)
	assert.NoError(t, err)

	// Unregister agent
	err = manager.UnregisterAgent(context.Background(), "test-agent")
	assert.NoError(t, err)

	// Try to get unregistered agent
	_, err = manager.GetAgent(context.Background(), "test-agent")
	assert.Error(t, err)

	// Try to unregister non-existent agent
	err = manager.UnregisterAgent(context.Background(), "non-existent")
	assert.Error(t, err)
}

func TestAgentManager_ListAgents(t *testing.T) {
	manager := NewAgentManager()
	mockAgent1 := &MockAgent{name: "agent1"}
	mockAgent2 := &MockAgent{name: "agent2"}

	// Register agents
	assert.NoError(t, manager.RegisterAgent(context.Background(), "agent1", mockAgent1))
	assert.NoError(t, manager.RegisterAgent(context.Background(), "agent2", mockAgent2))

	// List agents
	agents, err := manager.ListAgents(context.Background())
	assert.NoError(t, err)
	assert.Len(t, agents, 2)
	assert.Contains(t, agents, "agent1")
	assert.Contains(t, agents, "agent2")

	// List agent names
	names, err := manager.ListAgentNames(context.Background())
	assert.NoError(t, err)
	assert.Len(t, names, 2)
	assert.Contains(t, names, "agent1")
	assert.Contains(t, names, "agent2")
}

func TestAgentManager_ExecuteTask_Success(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{
		name:        "summary",
		returnValue: map[string]interface{}{"summary": "test summary"},
	}
	llmProvider := &MockLLMProvider{}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Create task
	task := NewTask("task-1", "summary", "summarize this content")
	task.Input = map[string]interface{}{
		"contents": []interface{}{"test content"},
	}

	// Execute task
	result, err := manager.ExecuteTask(context.Background(), task, llmProvider)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "task-1", result.TaskID)
	assert.Equal(t, "completed", result.Status)
	assert.Equal(t, "completed", task.Status)
	assert.NotNil(t, result.Result)

	// Check manager stats
	stats := manager.GetStats()
	assert.Equal(t, 1, stats.TotalTasks)
	assert.Equal(t, 1, stats.CompletedTasks)
	assert.Equal(t, 0, stats.FailedTasks)

	// Check task history
	history, err := manager.GetTaskHistory("task-1")
	assert.NoError(t, err)
	assert.Equal(t, result, history)
}

func TestAgentManager_ExecuteTask_AgentNotFound(t *testing.T) {
	manager := NewAgentManager()
	llmProvider := &MockLLMProvider{}

	// Create task with non-existent agent
	task := NewTask("task-1", "non-existent", "test query")

	// Execute task
	result, err := manager.ExecuteTask(context.Background(), task, llmProvider)
	assert.NoError(t, err) // Manager returns result with error, not an error
	assert.NotNil(t, result)
	assert.Equal(t, "task-1", result.TaskID)
	assert.Equal(t, "failed", result.Status)
	assert.Equal(t, "failed", task.Status)
	assert.Contains(t, result.Error, "agent not found")
	assert.Contains(t, result.Metadata["error_type"], "AGENT_NOT_FOUND")
}

func TestAgentManager_ExecuteTask_MissingLLMProvider(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "summary"}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Create task
	task := NewTask("task-1", "summary", "test query")

	// Execute task without LLM provider
	result, err := manager.ExecuteTask(context.Background(), task, nil)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "task-1", result.TaskID)
	assert.Equal(t, "failed", result.Status)
	assert.Equal(t, "failed", task.Status)
	assert.Contains(t, result.Error, "LLM provider is required")
	assert.Contains(t, result.Metadata["error_type"], "MISSING_LLM_PROVIDER")
}

func TestAgentManager_ExecuteTask_AgentFailure(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{
		name:       "summary",
		shouldFail: true,
	}
	llmProvider := &MockLLMProvider{}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Create task
	task := NewTask("task-1", "summary", "test query")

	// Execute task
	result, err := manager.ExecuteTask(context.Background(), task, llmProvider)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "task-1", result.TaskID)
	assert.Equal(t, "failed", result.Status)
	assert.Equal(t, "failed", task.Status)
	assert.Contains(t, result.Error, "agent execution failed")
	assert.Contains(t, result.Metadata["error_type"], "EXECUTION_ERROR")

	// Check manager stats
	stats := manager.GetStats()
	assert.Equal(t, 1, stats.TotalTasks)
	assert.Equal(t, 0, stats.CompletedTasks)
	assert.Equal(t, 1, stats.FailedTasks)
}

func TestAgentManager_ValidateTask(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "summary"}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Test valid task
	validTask := NewTask("task-1", "summary", "test query")
	err := manager.ValidateTask(context.Background(), validTask)
	assert.NoError(t, err)

	// Test nil task
	err = manager.ValidateTask(context.Background(), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "task cannot be nil")

	// Test task without ID
	invalidTask := &Task{AgentName: "summary", Query: "test"}
	err = manager.ValidateTask(context.Background(), invalidTask)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "task ID is required")

	// Test task without agent name
	invalidTask = &Task{ID: "task-1", Query: "test"}
	err = manager.ValidateTask(context.Background(), invalidTask)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "agent name is required")

	// Test task with non-existent agent
	invalidTask = &Task{ID: "task-1", AgentName: "non-existent", Query: "test"}
	err = manager.ValidateTask(context.Background(), invalidTask)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "agent not found")

	// Test task without query or input
	invalidTask = &Task{ID: "task-1", AgentName: "summary"}
	err = manager.ValidateTask(context.Background(), invalidTask)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must have either a query or input data")
}

func TestAgentManager_GetStats(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "summary"}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Check initial stats
	stats := manager.GetStats()
	assert.Equal(t, 1, stats.ActiveAgents)
	assert.Equal(t, 0, stats.TotalTasks)
	assert.Equal(t, 0, stats.CompletedTasks)
	assert.Equal(t, 0, stats.FailedTasks)
}

func TestAgentManager_TaskHistory(t *testing.T) {
	manager := NewAgentManager()
	mockAgent := &MockAgent{name: "summary"}
	llmProvider := &MockLLMProvider{}

	// Register agent
	assert.NoError(t, manager.RegisterAgent(context.Background(), "summary", mockAgent))

	// Execute a task
	task := NewTask("task-1", "summary", "test query")
	_, err := manager.ExecuteTask(context.Background(), task, llmProvider)
	assert.NoError(t, err)

	// Get task history
	history, err := manager.GetTaskHistory("task-1")
	assert.NoError(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "task-1", history.TaskID)
	assert.Equal(t, "completed", history.Status)

	// Get non-existent task history
	_, err = manager.GetTaskHistory("non-existent")
	assert.Error(t, err)

	// List all task history
	allHistory := manager.ListTaskHistory()
	assert.Len(t, allHistory, 1)
	assert.Contains(t, allHistory, "task-1")
}
