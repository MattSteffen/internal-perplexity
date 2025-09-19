package researcher

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/agents"
)

func TestResearcherAgentSchema(t *testing.T) {
	schema := ResearcherAgentSchema

	assert.Equal(t, "researcher", schema.Name)
	assert.Contains(t, schema.Description, "NOT YET IMPLEMENTED")
	assert.Contains(t, schema.Input.Required, "query")
	assert.Equal(t, "string", schema.Input.Types["query"])
	assert.Equal(t, "array", schema.Output.Structure["sources"])
}

func TestNewResearcherAgent(t *testing.T) {
	agent := NewResearcherAgent()

	assert.NotNil(t, agent)
	assert.NotNil(t, agent.systemPromptManager)
	assert.Equal(t, float64(1.0), agent.stats.SuccessRate)
	assert.Equal(t, 0, agent.stats.TotalExecutions)
}

func TestResearcherAgent_Execute(t *testing.T) {
	agent := NewResearcherAgent()

	input := &agents.AgentInput{
		Data: map[string]interface{}{
			"query": "test research query",
		},
	}

	result, err := agent.Execute(context.Background(), input, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
	assert.False(t, result.Success)
	assert.Contains(t, result.Metadata, "error")
	assert.Contains(t, result.Metadata, "recommended_agents")
}

func TestResearcherAgent_GetCapabilities(t *testing.T) {
	agent := NewResearcherAgent()

	caps := agent.GetCapabilities()
	assert.Len(t, caps, 5)
	assert.Contains(t, caps[0].Name, "web_research")
	assert.Contains(t, caps[0].Description, "NOT YET IMPLEMENTED")
}

func TestResearcherAgent_ValidateInput(t *testing.T) {
	agent := NewResearcherAgent()

	input := &agents.AgentInput{
		Data: map[string]interface{}{
			"query": "test query",
		},
	}

	err := agent.ValidateInput(input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_IMPLEMENTED")
}

func TestResearcherAgent_ValidateOutput(t *testing.T) {
	agent := NewResearcherAgent()

	result := &agents.AgentResult{
		Success: true,
		Content: map[string]interface{}{},
	}

	err := agent.ValidateOutput(result)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NOT_IMPLEMENTED")
}

func TestResearcherAgent_GetStats(t *testing.T) {
	agent := NewResearcherAgent()

	stats := agent.GetStats()
	assert.Equal(t, 0, stats.TotalExecutions)
	assert.Equal(t, float64(1.0), stats.SuccessRate)
}

func TestResearcherAgent_GetSystemPrompt(t *testing.T) {
	agent := NewResearcherAgent()

	prompt := agent.GetSystemPrompt()
	assert.NotNil(t, prompt)
	assert.Contains(t, prompt.BasePrompt, "NOT YET IMPLEMENTED")
}

func TestResearcherAgent_GetSchema(t *testing.T) {
	agent := NewResearcherAgent()

	schema := agent.GetSchema()
	assert.Equal(t, "researcher", schema.Name)
	assert.Equal(t, "0.1.0", schema.Version)
	assert.Contains(t, schema.Author, "Team")
}
