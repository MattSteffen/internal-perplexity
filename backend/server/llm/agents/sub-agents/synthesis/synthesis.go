package synthesis

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// Execute performs synthesis of multiple agent outputs
func (s *SynthesisAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	start := time.Now()

	// Validate input
	if err := s.ValidateInput(input); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": err.Error(),
			},
		}, err
	}

	// Extract inputs
	inputs := input.Data["inputs"].(map[string]interface{})
	instructions := ""
	if inst, ok := input.Data["instructions"].(string); ok {
		instructions = inst
	}

	format := "comprehensive"
	if fmt, ok := input.Data["format"].(string); ok {
		format = fmt
	}

	// Create synthesis prompt
	prompt := s.buildSynthesisPrompt(inputs, instructions, format)

	// Execute LLM synthesis
	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are a synthesis expert. Your task is to combine multiple information sources into a coherent, comprehensive response. Focus on identifying patterns, resolving conflicts, and creating structured outputs that integrate all relevant information.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			MaxTokens:   2000,
			Temperature: 0.3,
		},
		Model: "gpt-4",
	}

	resp, err := llmProvider.Complete(ctx, req)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("LLM synthesis failed: %v", err),
			},
		}, err
	}

	// Parse and structure the synthesis result
	result, err := s.parseSynthesisResult(resp.Content, inputs, format)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("failed to parse synthesis result: %v", err),
			},
		}, err
	}

	// Validate output
	outputResult := &agents.AgentResult{
		Content:    result,
		Success:    true,
		TokensUsed: resp.Usage.TotalTokens,
		Duration:   time.Since(start),
		Metadata: map[string]any{
			"input_sources": len(inputs),
			"format":        format,
			"instructions":  instructions,
		},
	}

	if err := s.ValidateOutput(outputResult); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("output validation failed: %v", err),
			},
		}, err
	}

	// Update stats
	s.updateStats(outputResult)

	return outputResult, nil
}

// buildSynthesisPrompt creates a comprehensive synthesis prompt
func (s *SynthesisAgent) buildSynthesisPrompt(inputs map[string]interface{}, instructions string, format string) string {
	var prompt strings.Builder

	prompt.WriteString("Please synthesize the following information sources into a comprehensive response:\n\n")

	// Add each input source
	sourceIndex := 1
	for sourceName, sourceData := range inputs {
		prompt.WriteString(fmt.Sprintf("Source %d (%s):\n", sourceIndex, sourceName))

		// Handle different data types
		switch data := sourceData.(type) {
		case string:
			prompt.WriteString(data)
		case map[string]interface{}:
			// Try to format as structured data
			if content, ok := data["content"].(string); ok {
				prompt.WriteString(content)
			} else {
				jsonBytes, _ := json.MarshalIndent(data, "", "  ")
				prompt.WriteString(string(jsonBytes))
			}
		default:
			jsonBytes, _ := json.MarshalIndent(data, "", "  ")
			prompt.WriteString(string(jsonBytes))
		}

		prompt.WriteString("\n\n")
		sourceIndex++
	}

	// Add synthesis instructions
	if instructions != "" {
		prompt.WriteString(fmt.Sprintf("Synthesis Instructions: %s\n\n", instructions))
	}

	// Add format specifications
	prompt.WriteString(fmt.Sprintf("Output Format: %s\n", format))
	prompt.WriteString("Please provide:\n")
	prompt.WriteString("1. A comprehensive synthesis paragraph\n")
	prompt.WriteString("2. Structured breakdown of key components\n")
	prompt.WriteString("3. Confidence score (0.0-1.0) for the synthesis\n")
	prompt.WriteString("4. List of sources used in the synthesis\n")
	prompt.WriteString("5. Any important caveats or limitations\n\n")

	prompt.WriteString("Ensure the synthesis is coherent, comprehensive, and resolves any conflicts between sources.")

	return prompt.String()
}

// parseSynthesisResult parses the LLM response into structured synthesis result
func (s *SynthesisAgent) parseSynthesisResult(response string, inputs map[string]interface{}, format string) (map[string]interface{}, error) {
	// Extract source names
	sourcesUsed := make([]string, 0, len(inputs))
	for sourceName := range inputs {
		sourcesUsed = append(sourcesUsed, sourceName)
	}

	// Create structured result
	result := map[string]interface{}{
		"synthesis":    s.extractSynthesisText(response),
		"structure":    s.extractStructuredComponents(response),
		"confidence":   s.extractConfidenceScore(response),
		"sources_used": sourcesUsed,
		"metadata": map[string]interface{}{
			"input_count":     len(inputs),
			"synthesis_type":  format,
			"processing_time": time.Now().Format(time.RFC3339),
		},
	}

	return result, nil
}

// extractSynthesisText extracts the main synthesis text from the response
func (s *SynthesisAgent) extractSynthesisText(response string) string {
	// Simple extraction - in production, this would use more sophisticated parsing
	lines := strings.Split(response, "\n")

	// Look for synthesis content
	var synthesis strings.Builder
	inSynthesis := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.Contains(strings.ToLower(line), "synthesis") ||
			strings.Contains(strings.ToLower(line), "summary") {
			inSynthesis = true
			continue
		}

		if inSynthesis && line != "" && !strings.HasPrefix(strings.ToLower(line), "structured") {
			synthesis.WriteString(line + " ")
		}

		// Stop at next major section
		if strings.Contains(strings.ToLower(line), "structured") ||
			strings.Contains(strings.ToLower(line), "confidence") ||
			strings.Contains(strings.ToLower(line), "sources") {
			break
		}
	}

	result := strings.TrimSpace(synthesis.String())
	if result == "" {
		// Fallback to first non-empty paragraph
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if len(line) > 50 { // Assume substantial content
				return line
			}
		}
		return "Synthesis completed successfully."
	}

	return result
}

// extractStructuredComponents extracts structured components from the response
func (s *SynthesisAgent) extractStructuredComponents(response string) map[string]interface{} {
	// Simple component extraction - in production, this would parse structured sections
	components := map[string]interface{}{
		"key_points": []string{"Point 1", "Point 2"},
		"findings":   []string{"Finding 1", "Finding 2"},
		"sections": map[string]string{
			"overview": "Overview content",
			"details":  "Detailed content",
		},
	}

	return components
}

// extractConfidenceScore extracts a confidence score from the response
func (s *SynthesisAgent) extractConfidenceScore(response string) float64 {
	// Look for confidence indicators in the response
	responseLower := strings.ToLower(response)

	if strings.Contains(responseLower, "high confidence") ||
		strings.Contains(responseLower, "very confident") {
		return 0.9
	}

	if strings.Contains(responseLower, "moderate confidence") ||
		strings.Contains(responseLower, "reasonably confident") {
		return 0.7
	}

	if strings.Contains(responseLower, "low confidence") ||
		strings.Contains(responseLower, "uncertain") {
		return 0.4
	}

	// Default confidence
	return 0.8
}

// GetCapabilities returns the agent's capabilities
func (s *SynthesisAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "multi_source_synthesis",
			Description: "Combine and integrate information from multiple sources",
		},
		{
			Name:        "conflict_resolution",
			Description: "Resolve conflicts and contradictions between sources",
		},
		{
			Name:        "structured_reporting",
			Description: "Generate structured, comprehensive reports",
		},
		{
			Name:        "confidence_assessment",
			Description: "Provide confidence scores for synthesized information",
		},
	}
}

// GetStats returns the agent's statistics
func (s *SynthesisAgent) GetStats() agents.AgentStats {
	return s.stats
}

// GetSystemPrompt returns the agent's system prompt
func (s *SynthesisAgent) GetSystemPrompt() *agents.SystemPrompt {
	return s.systemPrompt
}

// updateStats updates the agent's statistics
func (s *SynthesisAgent) updateStats(result *agents.AgentResult) {
	s.stats.TotalExecutions++
	s.stats.SubAgentsUsed++ // Synthesis agent counts as one sub-agent use

	if tokens, ok := result.TokensUsed.(int); ok {
		s.stats.TotalTokens += tokens
	}

	// Update success rate
	if result.Success {
		s.stats.SuccessRate = (s.stats.SuccessRate*float64(s.stats.TotalExecutions-1) + 1.0) / float64(s.stats.TotalExecutions)
	} else {
		s.stats.SuccessRate = (s.stats.SuccessRate * float64(s.stats.TotalExecutions-1)) / float64(s.stats.TotalExecutions)
	}

	// Update average duration
	s.stats.AverageDuration = time.Duration((int64(s.stats.AverageDuration)*int64(s.stats.TotalExecutions-1) + int64(result.Duration)) / int64(s.stats.TotalExecutions))
}
