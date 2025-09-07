package agents

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/providers/shared"
)

func TestExecutionPatternConstants(t *testing.T) {
	assert.Equal(t, "sequential", string(PatternSequential))
	assert.Equal(t, "parallel", string(PatternParallel))
	assert.Equal(t, "map_reduce", string(PatternMapReduce))
	assert.Equal(t, "direct", string(PatternDirect))
}

func TestTaskTypeConstants(t *testing.T) {
	assert.Equal(t, "tool", string(TaskTypeTool))
	assert.Equal(t, "subagent", string(TaskTypeSubAgent))
	assert.Equal(t, "response", string(TaskTypeResponse))
}

func TestAgentInput_QueryField(t *testing.T) {
	input := &AgentInput{
		Query: "test query",
		Data: map[string]any{
			"test": "data",
		},
	}

	assert.Equal(t, "test query", input.Query)
	assert.Equal(t, "data", input.Data["test"])
}

func TestAgentResult_ExecutionLog(t *testing.T) {
	now := time.Now()
	log := []ExecutionStep{
		{
			Type:        "test_step",
			Description: "Test execution step",
			Timestamp:   now,
			Data: map[string]interface{}{
				"test": "data",
			},
		},
	}

	result := &AgentResult{
		Success:      true,
		Content:      map[string]any{"result": "success"},
		Duration:     time.Second,
		ExecutionLog: log,
	}

	assert.True(t, result.Success)
	assert.Equal(t, "success", result.Content.(map[string]any)["result"])
	assert.Len(t, result.ExecutionLog, 1)
	assert.Equal(t, "test_step", result.ExecutionLog[0].Type)
}

func TestTask_Structure(t *testing.T) {
	task := Task{
		ID:          "task_1",
		Type:        TaskTypeTool,
		Name:        "calculator",
		Description: "Calculate something",
		Input: map[string]interface{}{
			"expression": "2 + 3",
		},
		Priority:  1,
		DependsOn: []string{"task_0"},
	}

	assert.Equal(t, "task_1", task.ID)
	assert.Equal(t, TaskTypeTool, task.Type)
	assert.Equal(t, "calculator", task.Name)
	assert.Equal(t, "Calculate something", task.Description)
	assert.Equal(t, 1, task.Priority)
	assert.Contains(t, task.DependsOn, "task_0")
	assert.Equal(t, "2 + 3", task.Input["expression"])
}

func TestExecutionPlan_Structure(t *testing.T) {
	plan := &ExecutionPlan{
		Tasks: []Task{
			{
				ID:          "task_1",
				Type:        TaskTypeTool,
				Name:        "calculator",
				Description: "Simple calculation",
			},
		},
		Pattern:         PatternSequential,
		FinalResponse:   "",
		Reasoning:       "Simple sequential execution",
		EstimatedTokens: 100,
	}

	assert.Len(t, plan.Tasks, 1)
	assert.Equal(t, PatternSequential, plan.Pattern)
	assert.Equal(t, "Simple sequential execution", plan.Reasoning)
	assert.Equal(t, 100, plan.EstimatedTokens)
	assert.Empty(t, plan.FinalResponse)
}

func TestSystemPrompt_Structure(t *testing.T) {
	prompt := &SystemPrompt{
		BasePrompt: "You are a helpful assistant",
		Capabilities: []string{
			"task_decomposition",
			"tool_usage",
		},
		Tools: []string{
			"calculator",
			"search",
		},
		SubAgents: []string{
			"summary",
			"researcher",
		},
		Examples: []string{
			"Example 1: Do something",
		},
		Constraints: map[string]string{
			"max_tokens": "1000",
		},
	}

	assert.Equal(t, "You are a helpful assistant", prompt.BasePrompt)
	assert.Contains(t, prompt.Capabilities, "task_decomposition")
	assert.Contains(t, prompt.Tools, "calculator")
	assert.Contains(t, prompt.SubAgents, "summary")
	assert.Len(t, prompt.Examples, 1)
	assert.Equal(t, "1000", prompt.Constraints["max_tokens"])
}

func TestAgentStats_Structure(t *testing.T) {
	stats := AgentStats{
		TotalExecutions: 10,
		AverageDuration: time.Second * 5,
		SuccessRate:     0.9,
		TotalTokens:     5000,
		TasksCreated:    25,
		ToolsCalled:     15,
		SubAgentsUsed:   5,
	}

	assert.Equal(t, 10, stats.TotalExecutions)
	assert.Equal(t, time.Second*5, stats.AverageDuration)
	assert.Equal(t, 0.9, stats.SuccessRate)
	assert.Equal(t, 5000, stats.TotalTokens)
	assert.Equal(t, 25, stats.TasksCreated)
	assert.Equal(t, 15, stats.ToolsCalled)
	assert.Equal(t, 5, stats.SubAgentsUsed)
}

func TestCapability_Structure(t *testing.T) {
	capability := Capability{
		Name:        "test_capability",
		Description: "A test capability for testing",
	}

	assert.Equal(t, "test_capability", capability.Name)
	assert.Equal(t, "A test capability for testing", capability.Description)
}

// Mock LLM provider for testing
type MockLLMProvider struct{}

func (m *MockLLMProvider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	return &shared.CompletionResponse{
		Content: "Mock response",
		Usage: shared.TokenUsage{
			TotalTokens: 50,
		},
	}, nil
}

func (m *MockLLMProvider) CountTokens(messages []shared.Message) (int, error) {
	return 25, nil
}

func (m *MockLLMProvider) GetSupportedModels() []shared.ModelInfo {
	return []shared.ModelInfo{}
}

func (m *MockLLMProvider) SupportsModel(model string) bool {
	return true
}

// Mock agent for testing
type MockAgent struct {
	name         string
	capabilities []Capability
}

func (m *MockAgent) Execute(ctx context.Context, input *AgentInput, llmProvider shared.LLMProvider) (*AgentResult, error) {
	return &AgentResult{
		Success:  true,
		Content:  map[string]any{"result": "mock execution"},
		Duration: time.Millisecond * 100,
	}, nil
}

func (m *MockAgent) GetCapabilities() []Capability {
	return m.capabilities
}

func (m *MockAgent) GetStats() AgentStats {
	return AgentStats{}
}

func (m *MockAgent) GetSystemPrompt() *SystemPrompt {
	return &SystemPrompt{
		BasePrompt: "Mock system prompt",
	}
}

func TestAgentInterface(t *testing.T) {
	mockAgent := &MockAgent{
		name: "test_agent",
		capabilities: []Capability{
			{Name: "test", Description: "Test capability"},
		},
	}

	// Test basic interface compliance
	assert.NotNil(t, mockAgent)
	assert.Equal(t, "test_agent", mockAgent.name)

	caps := mockAgent.GetCapabilities()
	assert.Len(t, caps, 1)
	assert.Equal(t, "test", caps[0].Name)

	prompt := mockAgent.GetSystemPrompt()
	assert.NotNil(t, prompt)
	assert.Equal(t, "Mock system prompt", prompt.BasePrompt)
}

func TestExecutionStep_Structure(t *testing.T) {
	step := ExecutionStep{
		Type:        "task_execution",
		Description: "Executing a test task",
		Timestamp:   time.Now(),
		Duration:    time.Second,
		Data: map[string]interface{}{
			"task_id": "test_task",
		},
	}

	assert.Equal(t, "task_execution", step.Type)
	assert.Equal(t, "Executing a test task", step.Description)
	assert.NotZero(t, step.Timestamp)
	assert.Equal(t, time.Second, step.Duration)
	assert.Equal(t, "test_task", step.Data["task_id"])
}
