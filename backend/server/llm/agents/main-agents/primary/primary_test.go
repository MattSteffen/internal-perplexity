package primary

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
)

func TestPrimaryAgentSchema(t *testing.T) {
	schema := PrimaryAgentSchema

	assert.Equal(t, "primary", schema.Name)
	assert.Contains(t, schema.Description, "orchestrator")
	assert.Contains(t, schema.Input.Required, "query")
	assert.Equal(t, "string", schema.Input.Types["query"])
	assert.Equal(t, "object", schema.Output.Structure["result"])
}

func TestNewPrimaryAgent(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	assert.NotNil(t, agent)
	assert.NotNil(t, agent.systemPromptManager)
	assert.NotNil(t, agent.taskPlanner)
	assert.NotNil(t, agent.decisionEngine)
	assert.NotNil(t, agent.toolRegistry)
	assert.Equal(t, float64(1.0), agent.stats.SuccessRate)
	assert.Equal(t, 0, agent.stats.TotalExecutions)
}

func TestPrimaryAgent_GetCapabilities(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	caps := agent.GetCapabilities()
	assert.Len(t, caps, 5)
	assert.Contains(t, caps[0].Name, "orchestration")
	assert.Contains(t, caps[2].Name, "tool_execution")
}

func TestPrimaryAgent_ValidateInput(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	// Valid input
	validInput := &agents.AgentInput{
		Query: "test query",
	}
	err := agent.ValidateInput(validInput)
	assert.NoError(t, err)

	// Invalid input - empty query
	invalidInput := &agents.AgentInput{
		Query: "",
	}
	err = agent.ValidateInput(invalidInput)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "MISSING_REQUIRED_FIELD")
}

func TestPrimaryAgent_GetStats(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	stats := agent.GetStats()
	assert.Equal(t, 0, stats.TotalExecutions)
	assert.Equal(t, float64(1.0), stats.SuccessRate)
	assert.Equal(t, time.Duration(0), stats.AverageDuration)
}

func TestPrimaryAgent_GetSystemPrompt(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	prompt := agent.GetSystemPrompt()
	assert.NotNil(t, prompt)
	assert.Contains(t, prompt.BasePrompt, "orchestrator")
}

func TestValidationError(t *testing.T) {
	err := agents.NewValidationError("test_field", "test message", "TEST_CODE", "test_value")

	assert.Equal(t, "test_field", err.Field)
	assert.Equal(t, "test message", err.Message)
	assert.Equal(t, "TEST_CODE", err.Code)
	assert.Equal(t, "test_value", err.Value)
	assert.Contains(t, err.Error(), "validation error")
}

func TestPrimaryAgent_GetSchema(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	schema := agent.GetSchema()
	assert.Equal(t, "primary", schema.Name)
	assert.Equal(t, "1.0.0", schema.Version)
	assert.Contains(t, schema.Author, "Team")
}

// Mock sub-agent for testing
type MockSubAgent struct {
	name        string
	shouldFail  bool
	returnValue interface{}
}

func (m *MockSubAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	if m.shouldFail {
		return &agents.AgentResult{
			Success: false,
			Content: nil,
		}, assert.AnError
	}

	return &agents.AgentResult{
		Success: true,
		Content: m.returnValue,
	}, nil
}

func (m *MockSubAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{}
}

func (m *MockSubAgent) GetStats() agents.AgentStats {
	return agents.AgentStats{}
}

func (m *MockSubAgent) GetSystemPrompt() *agents.SystemPrompt {
	return &agents.SystemPrompt{}
}

func TestPrimaryAgent_GetAvailableSubAgentNames(t *testing.T) {
	subAgents := map[string]agents.Agent{
		"summary":    &MockSubAgent{name: "summary"},
		"analyst":    &MockSubAgent{name: "analyst"},
		"researcher": &MockSubAgent{name: "researcher"},
	}
	toolRegistry := tools.NewRegistry()
	agent := NewPrimaryAgent(subAgents, toolRegistry)

	names := agent.getAvailableSubAgentNames()
	assert.Len(t, names, 3)
	assert.Contains(t, names, "summary")
	assert.Contains(t, names, "analyst")
	assert.Contains(t, names, "researcher")
}

func TestPrimaryAgent_ToolIntegration(t *testing.T) {
	subAgents := map[string]agents.Agent{}
	toolRegistry := tools.NewRegistry()

	// Create a mock tool
	mockTool := &MockTool{name: "calculator", description: "Mathematical calculations"}
	toolRegistry.Register(mockTool)

	agent := NewPrimaryAgent(subAgents, toolRegistry)

	// Test tool availability
	availableTools := agent.GetAvailableTools()
	assert.Len(t, availableTools, 1)
	assert.Contains(t, availableTools, "calculator")

	// Test tool descriptions
	toolDescriptions := agent.GetAvailableToolDescriptions()
	assert.Len(t, toolDescriptions, 1)
	assert.Equal(t, "Mathematical calculations", toolDescriptions["calculator"])
}

// Mock tool for testing
type MockTool struct {
	name        string
	description string
}

func (m *MockTool) Name() string {
	return m.name
}

func (m *MockTool) Description() string {
	return m.description
}

func (m *MockTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
	}
}

func (m *MockTool) Definition() *api.ToolDefinition {
	return &api.ToolDefinition{
		Type: "function",
		Function: api.FunctionDefinition{
			Name:        m.name,
			Description: m.description,
		},
	}
}

func (m *MockTool) Execute(ctx context.Context, input *tools.ToolInput, llmProvider shared.LLMProvider) (*tools.ToolResult, error) {
	return &tools.ToolResult{
		Success: true,
		Data:    map[string]interface{}{"result": "42"},
	}, nil
}
