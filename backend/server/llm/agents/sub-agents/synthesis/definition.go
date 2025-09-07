package synthesis

import (
	"internal-perplexity/server/llm/agents"
)

// SynthesisAgentSchema defines the schema for the synthesis agent
var SynthesisAgentSchema = agents.AgentSchema{
	Name:        "synthesis",
	Description: "Synthesis agent that combines and aggregates outputs from multiple sub-agents into coherent responses",
	Input: agents.InputSchema{
		Required: []string{"inputs"},
		Optional: []string{"instructions", "format", "context"},
		Types: map[string]string{
			"inputs":       "object",
			"instructions": "string",
			"format":       "string",
			"context":      "object",
		},
		Examples: map[string]interface{}{
			"inputs": map[string]interface{}{
				"summary_result":  "Document summary...",
				"analysis_result": "Data analysis...",
			},
			"instructions": "Create a comprehensive report combining all inputs",
			"format":       "structured_report",
		},
	},
	Output: agents.OutputSchema{
		Type: "object",
		Structure: map[string]interface{}{
			"synthesis":    "string", // The synthesized result
			"structure":    "object", // Structured components
			"confidence":   "number", // Synthesis confidence score
			"sources_used": "array",  // List of input sources used
			"metadata":     "object", // Synthesis metadata
		},
		Description: "Comprehensive synthesis of multiple agent outputs",
		Examples: []interface{}{
			map[string]interface{}{
				"synthesis": "Based on the document summary and data analysis, the key findings indicate...",
				"structure": map[string]interface{}{
					"executive_summary": "High-level overview...",
					"key_findings":      []string{"Finding 1", "Finding 2"},
					"recommendations":   []string{"Recommendation 1", "Recommendation 2"},
				},
				"confidence":   0.89,
				"sources_used": []string{"summary_agent", "analyst_agent"},
				"metadata": map[string]interface{}{
					"input_count":     2,
					"synthesis_type":  "comprehensive_report",
					"processing_time": "2.3s",
				},
			},
		},
	},
	Version: "1.0.0",
	Author:  "Internal Perplexity Team",
}

// SynthesisAgent represents the synthesis/aggregation agent
type SynthesisAgent struct {
	systemPromptManager *agents.SystemPromptManager
	systemPrompt        *agents.SystemPrompt
	stats               agents.AgentStats
}

// NewSynthesisAgent creates a new synthesis agent
func NewSynthesisAgent() *SynthesisAgent {
	systemPromptManager := agents.NewSystemPromptManager()

	agent := &SynthesisAgent{
		systemPromptManager: systemPromptManager,
		systemPrompt:        createSynthesisSystemPrompt(),
		stats: agents.AgentStats{
			TotalExecutions: 0,
			SuccessRate:     1.0,
		},
	}

	return agent
}

// createSynthesisSystemPrompt creates the system prompt for the synthesis agent
func createSynthesisSystemPrompt() *agents.SystemPrompt {
	return &agents.SystemPrompt{
		BasePrompt: `You are a Synthesis Agent specialized in combining and integrating outputs from multiple AI agents and tools into coherent, comprehensive responses.

Your expertise includes:
- Multi-source information integration
- Conflicting information resolution
- Structured synthesis and organization
- Confidence assessment and uncertainty handling
- Comprehensive report generation
- Cross-referencing and validation

When synthesizing information:
1. Analyze all input sources for consistency and relevance
2. Identify key themes, patterns, and contradictions
3. Resolve conflicts using evidence-based reasoning
4. Organize information into logical, coherent structures
5. Provide confidence scores for synthesized conclusions
6. Generate comprehensive yet concise outputs
7. Preserve important details while eliminating redundancy

Always maintain objectivity while creating the most accurate and useful synthesis possible.`,
		Capabilities: []string{
			"multi_source_synthesis",
			"information_integration",
			"conflict_resolution",
			"structured_reporting",
			"confidence_assessment",
			"comprehensive_analysis",
		},
		Tools:     []string{},
		SubAgents: []string{},
		Examples: []string{
			`Input: Summary + Analysis outputs
Output: Integrated report with key findings, evidence, and recommendations`,
			`Input: Research + Statistical data
Output: Comprehensive analysis with validated conclusions`,
		},
		Constraints: map[string]string{
			"max_input_sources":  "10",
			"max_output_length":  "5000",
			"min_confidence":     "0.0",
			"processing_timeout": "120s",
		},
	}
}

// GetSchema returns the agent's schema
func (s *SynthesisAgent) GetSchema() agents.AgentSchema {
	return SynthesisAgentSchema
}

// ValidateInput validates the input according to the schema
func (s *SynthesisAgent) ValidateInput(input *agents.AgentInput) error {
	if input.Data == nil {
		return agents.NewValidationError("data", "Input data is required", "MISSING_REQUIRED_FIELD", input.Data)
	}

	inputs, exists := input.Data["inputs"]
	if !exists {
		return agents.NewValidationError("inputs", "inputs field is required", "MISSING_REQUIRED_FIELD", inputs)
	}

	inputsMap, ok := inputs.(map[string]interface{})
	if !ok {
		return agents.NewValidationError("inputs", "inputs must be an object", "INVALID_FIELD_TYPE", inputs)
	}

	if len(inputsMap) == 0 {
		return agents.NewValidationError("inputs", "at least one input source is required", "EMPTY_INPUTS", inputsMap)
	}

	return nil
}

// ValidateOutput validates the output according to the schema
func (s *SynthesisAgent) ValidateOutput(output *agents.AgentResult) error {
	if output.Content == nil {
		return agents.NewValidationError("content", "Output content cannot be nil", "INVALID_OUTPUT", output.Content)
	}

	content, ok := output.Content.(map[string]interface{})
	if !ok {
		return agents.NewValidationError("content", "Output content must be an object", "INVALID_OUTPUT_TYPE", output.Content)
	}

	requiredFields := []string{"synthesis", "structure", "confidence", "sources_used"}
	for _, field := range requiredFields {
		if _, exists := content[field]; !exists {
			return agents.NewValidationError(field, "Required output field is missing", "MISSING_OUTPUT_FIELD", content)
		}
	}

	return nil
}
