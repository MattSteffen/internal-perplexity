package summary

import (
	"context"
	"fmt"
	"strings"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// SummaryAgent handles document summarization tasks
type SummaryAgent struct {
	llmClient shared.LLMProvider
	stats     agents.AgentStats
}

// NewSummaryAgent creates a new summary agent
func NewSummaryAgent(llmClient shared.LLMProvider) *SummaryAgent {
	return &SummaryAgent{
		llmClient: llmClient,
		stats: agents.AgentStats{
			TotalExecutions: 0,
			SuccessRate:     1.0,
		},
	}
}

// Execute processes a list of content items and generates a summary
func (s *SummaryAgent) Execute(ctx context.Context, input *agents.AgentInput) (*agents.AgentResult, error) {
	start := time.Now()

	// Validate input
	if err := s.validateInput(input); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Content:  nil,
			Duration: time.Since(start),
			Metadata: map[string]interface{}{
				"error": err.Error(),
			},
		}, nil
	}

	// Extract parameters
	contents := input.Data["contents"].([]interface{})
	instructions := ""
	if inst, ok := input.Data["instructions"].(string); ok {
		instructions = inst
	}

	focusAreas := []string{}
	if fa, ok := input.Data["focus_areas"].([]interface{}); ok {
		for _, area := range fa {
			if areaStr, ok := area.(string); ok {
				focusAreas = append(focusAreas, areaStr)
			}
		}
	}

	// Convert contents to strings
	contentStrings := make([]string, len(contents))
	combinedLength := 0
	for i, content := range contents {
		if contentStr, ok := content.(string); ok {
			contentStrings[i] = contentStr
			combinedLength += len(contentStr)
		}
	}

	// Create prompt
	prompt := s.buildPrompt(contentStrings, instructions, focusAreas)

	// Single LLM call for summarization
	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are a professional summarizer. Create comprehensive yet concise summaries that capture the essential information and key insights from the provided content.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	// Extract model and API key from input context
	model := "gpt-4" // default model
	apiKey := ""

	if input.Context != nil {
		if m, ok := input.Context["model"].(string); ok && m != "" {
			model = m
		}
		if key, ok := input.Context["api_key"].(string); ok {
			apiKey = key
		}
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			MaxTokens:   1000,
			Temperature: 0.3,
		},
		Model:  model,
		APIKey: apiKey,
	}

	resp, err := s.llmClient.Complete(ctx, req)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Content:  nil,
			Duration: time.Since(start),
			Metadata: map[string]interface{}{
				"error": fmt.Sprintf("LLM completion failed: %v", err),
			},
		}, nil
	}

	// Update stats
	s.stats.TotalExecutions++
	if resp.Usage.TotalTokens > 0 {
		s.stats.TotalTokens += resp.Usage.TotalTokens
	}
	s.stats.AverageDuration = time.Duration((int64(s.stats.AverageDuration)*int64(s.stats.TotalExecutions-1) + int64(time.Since(start))) / int64(s.stats.TotalExecutions))

	return &agents.AgentResult{
		Content: map[string]interface{}{
			"summary": resp.Content,
			"metadata": map[string]interface{}{
				"content_count":   len(contentStrings),
				"combined_length": combinedLength,
				"focus_areas":     focusAreas,
				"instructions":    instructions,
			},
		},
		Success:    true,
		TokensUsed: resp.Usage.TotalTokens,
		Duration:   time.Since(start),
		Metadata: map[string]interface{}{
			"input_length":  combinedLength,
			"focus_areas":   focusAreas,
			"content_count": len(contentStrings),
		},
	}, nil
}

// validateInput validates the agent input
func (s *SummaryAgent) validateInput(input *agents.AgentInput) error {
	if input.Data == nil {
		return fmt.Errorf("input data is required")
	}

	contents, exists := input.Data["contents"]
	if !exists {
		return fmt.Errorf("contents field is required")
	}

	contentsSlice, ok := contents.([]interface{})
	if !ok {
		return fmt.Errorf("contents must be an array")
	}

	if len(contentsSlice) == 0 {
		return fmt.Errorf("at least one content item is required")
	}

	return nil
}

// buildPrompt constructs the summarization prompt
func (s *SummaryAgent) buildPrompt(contents []string, instructions string, focusAreas []string) string {
	var prompt strings.Builder

	prompt.WriteString("Please summarize the following content:\n\n")

	for i, content := range contents {
		prompt.WriteString(fmt.Sprintf("Content %d:\n%s\n\n", i+1, content))
	}

	if instructions != "" {
		prompt.WriteString(fmt.Sprintf("Instructions: %s\n\n", instructions))
	}

	if len(focusAreas) > 0 {
		prompt.WriteString("Focus areas: ")
		for i, area := range focusAreas {
			if i > 0 {
				prompt.WriteString(", ")
			}
			prompt.WriteString(area)
		}
		prompt.WriteString("\n\n")
	}

	prompt.WriteString("Provide a comprehensive summary that captures the key points and insights.")

	return prompt.String()
}

// GetCapabilities returns the agent's capabilities
func (s *SummaryAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "content_summarization",
			Description: "Summarize multiple documents or text contents into coherent summaries",
		},
		{
			Name:        "focused_summarization",
			Description: "Generate summaries with specific focus areas and instructions",
		},
	}
}

// GetStats returns the agent's statistics
func (s *SummaryAgent) GetStats() agents.AgentStats {
	return s.stats
}
