package analyst

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/agents"
)

func TestAnalystAgentSchema(t *testing.T) {
	schema := AnalystAgentSchema

	assert.Equal(t, "analyst", schema.Name)
	assert.Contains(t, schema.Description, "NOT YET IMPLEMENTED")
	assert.Contains(t, schema.Input.Required, "data")
	assert.Equal(t, "object", schema.Input.Types["data"])
	assert.Equal(t, "object", schema.Output.Structure["analysis"])
}

func TestNewAnalystAgent(t *testing.T) {
	agent := NewAnalystAgent()

	assert.NotNil(t, agent)
	assert.NotNil(t, agent.systemPromptManager)
	assert.Equal(t, float64(1.0), agent.stats.SuccessRate)
	assert.Equal(t, 0, agent.stats.TotalExecutions)
}

func TestAnalystAgent_Execute(t *testing.T) {
	agent := NewAnalystAgent()

	input := &agents.AgentInput{
		Data: map[string]interface{}{
			"data": "test data",
		},
	}

	result, err := agent.Execute(context.Background(), input, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
	assert.False(t, result.Success)
	assert.Contains(t, result.Metadata, "error")
	assert.Contains(t, result.Metadata, "recommended_agents")
}

func TestAnalystAgent_GetCapabilities(t *testing.T) {
	agent := NewAnalystAgent()

	caps := agent.GetCapabilities()
	assert.Len(t, caps, 5)
	assert.Contains(t, caps[0].Name, "statistical")
	assert.Contains(t, caps[0].Description, "NOT YET IMPLEMENTED")
}

func TestAnalystAgent_ValidateInput(t *testing.T) {
	agent := NewAnalystAgent()

	input := &agents.AgentInput{
		Data: map[string]interface{}{
			"data": "test data",
		},
	}

	err := agent.ValidateInput(input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_IMPLEMENTED")
}

func TestAnalystAgent_ValidateOutput(t *testing.T) {
	agent := NewAnalystAgent()

	result := &agents.AgentResult{
		Success: true,
		Content: map[string]interface{}{},
	}

	err := agent.ValidateOutput(result)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_IMPLEMENTED")
}

func TestAnalystAgent_GetStats(t *testing.T) {
	agent := NewAnalystAgent()

	stats := agent.GetStats()
	assert.Equal(t, 0, stats.TotalExecutions)
	assert.Equal(t, float64(1.0), stats.SuccessRate)
}

func TestAnalystAgent_GetSystemPrompt(t *testing.T) {
	agent := NewAnalystAgent()

	prompt := agent.GetSystemPrompt()
	assert.NotNil(t, prompt)
	assert.Contains(t, prompt.BasePrompt, "NOT YET IMPLEMENTED")
}

func TestAnalystAgent_GetSchema(t *testing.T) {
	agent := NewAnalystAgent()

	schema := agent.GetSchema()
	assert.Equal(t, "analyst", schema.Name)
	assert.Equal(t, "0.1.0", schema.Version)
	assert.Contains(t, schema.Author, "Team")
}
