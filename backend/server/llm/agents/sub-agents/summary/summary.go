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
// The struct and construction methods are defined in definition.go

// Execute processes a list of content items and generates a summary
func (s *SummaryAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	start := time.Now()

	// Validate input
	if err := s.ValidateInput(input); err != nil {
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

	resp, err := llmProvider.Complete(ctx, req)
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
			Name:        "document_summarization",
			Description: "Summarize multiple documents into coherent overviews",
		},
		{
			Name:        "content_analysis",
			Description: "Analyze content and extract key insights and themes",
		},
		{
			Name:        "structured_summaries",
			Description: "Create structured summaries with focus areas and metadata",
		},
		{
			Name:        "instruction_based_summarization",
			Description: "Summarize content according to specific user instructions",
		},
	}
}

// GetStats returns the agent's statistics
func (s *SummaryAgent) GetStats() agents.AgentStats {
	return s.stats
}

// GetSystemPrompt returns the agent's system prompt
func (s *SummaryAgent) GetSystemPrompt() *agents.SystemPrompt {
	return s.systemPrompt
}
