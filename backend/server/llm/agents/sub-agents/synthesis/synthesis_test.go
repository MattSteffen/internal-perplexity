package synthesis

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

func TestSynthesisAgentSchema(t *testing.T) {
	schema := SynthesisAgentSchema

	assert.Equal(t, "synthesis", schema.Name)
	assert.Contains(t, schema.Description, "Synthesis")
	assert.Contains(t, schema.Input.Required, "inputs")
	assert.Equal(t, "object", schema.Input.Types["inputs"])
	assert.Equal(t, "object", schema.Output.Structure["structure"])
}

func TestNewSynthesisAgent(t *testing.T) {
	agent := NewSynthesisAgent()

	assert.NotNil(t, agent)
	assert.NotNil(t, agent.systemPromptManager)
	assert.Equal(t, float64(1.0), agent.stats.SuccessRate)
	assert.Equal(t, 0, agent.stats.TotalExecutions)
}

func TestSynthesisAgent_GetCapabilities(t *testing.T) {
	agent := NewSynthesisAgent()

	caps := agent.GetCapabilities()
	assert.Len(t, caps, 4)
	assert.Contains(t, caps[0].Name, "synthesis")
	assert.Contains(t, caps[1].Name, "conflict")
}

func TestSynthesisAgent_ValidateInput(t *testing.T) {
	agent := NewSynthesisAgent()

	// Valid input
	validInput := &agents.AgentInput{
		Data: map[string]interface{}{
			"inputs": map[string]interface{}{
				"source1": "content1",
				"source2": "content2",
			},
		},
	}
	err := agent.ValidateInput(validInput)
	assert.NoError(t, err)

	// Invalid input - missing inputs
	invalidInput := &agents.AgentInput{
		Data: map[string]interface{}{},
	}
	err = agent.ValidateInput(invalidInput)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "MISSING_REQUIRED_FIELD")

	// Invalid input - empty inputs
	emptyInput := &agents.AgentInput{
		Data: map[string]interface{}{
			"inputs": map[string]interface{}{},
		},
	}
	err = agent.ValidateInput(emptyInput)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "EMPTY_INPUTS")
}

func TestSynthesisAgent_ValidateOutput(t *testing.T) {
	agent := NewSynthesisAgent()

	// Valid output
	validResult := &agents.AgentResult{
		Content: map[string]interface{}{
			"synthesis":    "Test synthesis",
			"structure":    map[string]interface{}{"key": "value"},
			"confidence":   0.8,
			"sources_used": []string{"source1"},
		},
		Success: true,
	}
	err := agent.ValidateOutput(validResult)
	assert.NoError(t, err)

	// Invalid output - missing required field
	invalidResult := &agents.AgentResult{
		Content: map[string]interface{}{
			"synthesis": "Test synthesis",
			// Missing structure, confidence, sources_used
		},
		Success: true,
	}
	err = agent.ValidateOutput(invalidResult)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "MISSING_OUTPUT_FIELD")
}

func TestSynthesisAgent_GetStats(t *testing.T) {
	agent := NewSynthesisAgent()

	stats := agent.GetStats()
	assert.Equal(t, 0, stats.TotalExecutions)
	assert.Equal(t, float64(1.0), stats.SuccessRate)
	assert.Equal(t, time.Duration(0), stats.AverageDuration)
}

func TestSynthesisAgent_GetSystemPrompt(t *testing.T) {
	agent := NewSynthesisAgent()

	prompt := agent.GetSystemPrompt()
	assert.NotNil(t, prompt)
	assert.Contains(t, prompt.BasePrompt, "Synthesis Agent")
}

func TestSynthesisAgent_GetSchema(t *testing.T) {
	agent := NewSynthesisAgent()

	schema := agent.GetSchema()
	assert.Equal(t, "synthesis", schema.Name)
	assert.Equal(t, "1.0.0", schema.Version)
	assert.Contains(t, schema.Author, "Team")
}

func TestExtractSynthesisText(t *testing.T) {
	agent := NewSynthesisAgent()

	// Test with synthesis section
	response := `This is some text.

Synthesis: This is the main synthesis content that should be extracted.

Structured Components:`
	text := agent.extractSynthesisText(response)
	assert.Contains(t, text, "main synthesis content")

	// Test fallback
	response2 := `This is a substantial paragraph with enough content to be considered the main synthesis text.`
	text2 := agent.extractSynthesisText(response2)
	assert.Contains(t, text2, "substantial paragraph")
}

func TestExtractConfidenceScore(t *testing.T) {
	agent := NewSynthesisAgent()

	// Test high confidence
	assert.Equal(t, 0.9, agent.extractConfidenceScore("High confidence in this synthesis"))

	// Test moderate confidence
	assert.Equal(t, 0.7, agent.extractConfidenceScore("Moderate confidence assessment"))

	// Test low confidence
	assert.Equal(t, 0.4, agent.extractConfidenceScore("Low confidence due to uncertainty"))

	// Test default
	assert.Equal(t, 0.8, agent.extractConfidenceScore("Regular synthesis without confidence indicators"))
}

func TestBuildSynthesisPrompt(t *testing.T) {
	agent := NewSynthesisAgent()

	inputs := map[string]interface{}{
		"summary_agent": "Document summary content",
		"analyst_agent": map[string]interface{}{
			"content": "Analysis results",
			"type":    "data_analysis",
		},
	}

	prompt := agent.buildSynthesisPrompt(inputs, "Create comprehensive report", "structured_report")

	assert.Contains(t, prompt, "Source 1")
	assert.Contains(t, prompt, "Source 2")
	assert.Contains(t, prompt, "comprehensive report")
	assert.Contains(t, prompt, "structured_report")
}

// Mock LLM provider for testing
type MockLLMProvider struct {
	responseContent string
	shouldFail      bool
}

func (m *MockLLMProvider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	if m.shouldFail {
		return nil, assert.AnError
	}

	return &shared.CompletionResponse{
		Content: m.responseContent,
		Usage: shared.TokenUsage{
			TotalTokens: 150,
		},
	}, nil
}

func (m *MockLLMProvider) CountTokens(messages []shared.Message) (int, error) {
	return 100, nil
}

func (m *MockLLMProvider) GetSupportedModels() []shared.ModelInfo {
	return []shared.ModelInfo{}
}

func (m *MockLLMProvider) SupportsModel(model string) bool {
	return true
}

func TestSynthesisAgent_Execute(t *testing.T) {
	agent := NewSynthesisAgent()

	input := &agents.AgentInput{
		Data: map[string]interface{}{
			"inputs": map[string]interface{}{
				"source1": "Content from source 1",
				"source2": "Content from source 2",
			},
			"instructions": "Combine these sources",
		},
	}

	// Mock LLM provider
	llmProvider := &MockLLMProvider{
		responseContent: `Synthesis: Combined content from both sources provides comprehensive information.

Structured Components:
- Key points identified
- Analysis completed

Confidence: High confidence in this synthesis`,
	}

	result, err := agent.Execute(context.Background(), input, llmProvider)

	assert.NoError(t, err)
	assert.True(t, result.Success)
	assert.NotNil(t, result.Content)

	content := result.Content.(map[string]interface{})
	assert.Contains(t, content, "synthesis")
	assert.Contains(t, content, "structure")
	assert.Contains(t, content, "confidence")
	assert.Contains(t, content, "sources_used")
}
