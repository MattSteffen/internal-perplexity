package summary

import (
	"internal-perplexity/server/llm/agents"
)

// SummaryAgentSchema defines the schema for the summary agent
var SummaryAgentSchema = agents.AgentSchema{
	Name:        "summary",
	Description: "Specialized agent for document summarization and content analysis",
	Input: agents.InputSchema{
		Required: []string{"contents"},
		Optional: []string{"instructions", "focus_areas"},
		Types: map[string]string{
			"contents":     "array",
			"instructions": "string",
			"focus_areas":  "array",
		},
		Examples: map[string]interface{}{
			"contents": []interface{}{
				"First document content...",
				"Second document content...",
			},
			"instructions": "Focus on key findings and conclusions",
			"focus_areas":  []interface{}{"findings", "conclusions"},
		},
	},
	Output: agents.OutputSchema{
		Type: "object",
		Structure: map[string]interface{}{
			"summary":         "string",  // The generated summary
			"content_count":   "integer", // Number of content items processed
			"combined_length": "integer", // Total character count
			"focus_areas":     "array",   // Focus areas used
			"instructions":    "string",  // Instructions provided
			"metadata":        "object",  // Processing metadata
		},
		Description: "Structured summary result with metadata",
		Examples: []interface{}{
			map[string]interface{}{
				"summary":         "Generated summary text...",
				"content_count":   2,
				"combined_length": 1500,
				"focus_areas":     []string{"findings", "conclusions"},
				"instructions":    "Focus on key findings",
				"metadata": map[string]interface{}{
					"processing_time": "1.2s",
					"model_used":      "gpt-4",
				},
			},
		},
	},
	Version: "1.0.0",
	Author:  "Internal Perplexity Team",
}

// SummaryAgent represents the summary sub-agent
type SummaryAgent struct {
	systemPromptManager *agents.SystemPromptManager
	systemPrompt        *agents.SystemPrompt
	stats               agents.AgentStats
}

// NewSummaryAgent creates a new summary agent
func NewSummaryAgent() *SummaryAgent {
	systemPromptManager := agents.NewSystemPromptManager()

	agent := &SummaryAgent{
		systemPromptManager: systemPromptManager,
		systemPrompt:        createSummarySystemPrompt(),
		stats: agents.AgentStats{
			TotalExecutions: 0,
			SuccessRate:     1.0,
		},
	}

	return agent
}

// createSummarySystemPrompt creates the system prompt for the summary agent
func createSummarySystemPrompt() *agents.SystemPrompt {
	return &agents.SystemPrompt{
		BasePrompt: `You are a professional document summarizer specialized in creating concise, accurate summaries of various types of content.

Your expertise includes:
- Multi-document analysis and synthesis
- Key point extraction and prioritization
- Maintaining context and coherence
- Adapting summary style based on content type
- Handling different content formats (text, articles, reports, etc.)

Always focus on the most important information while preserving the essential meaning and context.`,
		Capabilities: []string{
			"content_summarization",
			"key_point_extraction",
			"multi_document_synthesis",
			"context_preservation",
		},
		Tools:     []string{},
		SubAgents: []string{},
		Examples: []string{
			`Input: Multiple research papers about AI
Output: Concise summary highlighting key findings, methodologies, and conclusions`,
			`Input: Technical documentation
Output: Clear summary of functionality, usage, and important details`,
		},
		Constraints: map[string]string{
			"max_input_length":   "10000",
			"max_summary_ratio":  "0.2",
			"processing_timeout": "60s",
		},
	}
}

// GetSchema returns the agent's schema
func (s *SummaryAgent) GetSchema() agents.AgentSchema {
	return SummaryAgentSchema
}

// ValidateInput validates the input according to the schema
func (s *SummaryAgent) ValidateInput(input *agents.AgentInput) error {
	if input.Data == nil {
		return agents.NewValidationError("data", "Input data is required", "MISSING_REQUIRED_FIELD", input.Data)
	}

	contents, exists := input.Data["contents"]
	if !exists {
		return agents.NewValidationError("contents", "contents field is required", "MISSING_REQUIRED_FIELD", contents)
	}

	contentsArray, ok := contents.([]interface{})
	if !ok {
		return agents.NewValidationError("contents", "contents must be an array", "INVALID_FIELD_TYPE", contents)
	}

	if len(contentsArray) == 0 {
		return agents.NewValidationError("contents", "contents array cannot be empty", "EMPTY_CONTENTS", contentsArray)
	}

	// Validate instructions if provided
	if instructions, exists := input.Data["instructions"]; exists {
		if _, ok := instructions.(string); !ok {
			return agents.NewValidationError("instructions", "instructions must be a string", "INVALID_FIELD_TYPE", instructions)
		}
	}

	// Validate focus_areas if provided
	if focusAreas, exists := input.Data["focus_areas"]; exists {
		focusArray, ok := focusAreas.([]interface{})
		if !ok {
			return agents.NewValidationError("focus_areas", "focus_areas must be an array", "INVALID_FIELD_TYPE", focusAreas)
		}

		// Check for duplicate focus areas
		seen := make(map[string]bool)
		for _, area := range focusArray {
			if areaStr, ok := area.(string); ok {
				if seen[areaStr] {
					return agents.NewValidationError("focus_areas", "focus_areas contains duplicates", "DUPLICATE_FOCUS_AREAS", focusAreas)
				}
				seen[areaStr] = true
			} else {
				return agents.NewValidationError("focus_areas", "all focus areas must be strings", "INVALID_FIELD_TYPE", focusAreas)
			}
		}
	}

	return nil
}

// ValidateOutput validates the output according to the schema
func (s *SummaryAgent) ValidateOutput(output *agents.AgentResult) error {
	if output.Content == nil {
		return agents.NewValidationError("content", "Output content cannot be nil", "INVALID_OUTPUT", output.Content)
	}

	content, ok := output.Content.(map[string]interface{})
	if !ok {
		return agents.NewValidationError("content", "Output content must be an object", "INVALID_OUTPUT_TYPE", output.Content)
	}

	requiredFields := []string{"summary", "content_count", "combined_length", "metadata"}
	for _, field := range requiredFields {
		if _, exists := content[field]; !exists {
			return agents.NewValidationError(field, "Required output field is missing", "MISSING_OUTPUT_FIELD", content)
		}
	}

	// Validate summary is string
	if summary, exists := content["summary"]; exists {
		if _, ok := summary.(string); !ok {
			return agents.NewValidationError("summary", "summary must be a string", "INVALID_FIELD_TYPE", summary)
		}
	}

	// Validate content_count is number
	if count, exists := content["content_count"]; exists {
		if _, ok := count.(int); !ok {
			return agents.NewValidationError("content_count", "content_count must be an integer", "INVALID_FIELD_TYPE", count)
		}
	}

	return nil
}
