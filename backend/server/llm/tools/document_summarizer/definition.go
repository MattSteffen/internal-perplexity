package document_summarizer

import (
	"context"
	"fmt"
	"strconv"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
)

// DocumentSummarizer provides document summarization capabilities
type DocumentSummarizer struct{}

// NewDocumentSummarizer creates a new document summarizer tool
func NewDocumentSummarizer() *DocumentSummarizer {
	return &DocumentSummarizer{}
}

// Name returns the tool name
func (d *DocumentSummarizer) Name() string {
	return "document_summarizer"
}

// Description returns the tool description
func (d *DocumentSummarizer) Description() string {
	return "Summarizes documents and text content to specified length using intelligent language models"
}

// Schema returns the JSON schema for input validation
func (d *DocumentSummarizer) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]interface{}{
			"content": map[string]interface{}{
				"type":        "string",
				"description": "The document content to summarize",
			},
			"max_length": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum length of summary in words",
				"default":     200,
			},
		},
		"required": []string{"content"},
	}
}

// Definition returns the OpenAI tool definition
func (d *DocumentSummarizer) Definition() *api.ToolDefinition {
	return &api.ToolDefinition{
		Type: "function",
		Function: api.FunctionDefinition{
			Name:        "document_summarizer",
			Description: "Summarize documents and text content to a specified length. Use this tool when you need to condense long text into key points and main ideas. The tool uses advanced language models to create coherent, accurate summaries that preserve essential information while reducing length. Returns the summary text along with statistics about the original and summarized content.",
			Strict:      &[]bool{true}[0],
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"content": map[string]interface{}{
						"type":        "string",
						"description": "The text content to summarize. Can be any length of plain text content.",
					},
					"max_length": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum length of the summary in words (default: 200). The actual summary may be slightly shorter to maintain coherence.",
						"default":     200,
						"minimum":     1,
						"maximum":     5000,
					},
				},
				"required": []string{"content"},
			},
		},
	}
}

// Execute performs document summarization using LLM
func (d *DocumentSummarizer) Execute(ctx context.Context, input *tools.ToolInput, llmProvider shared.LLMProvider) (*tools.ToolResult, error) {
	content, ok := input.Data["content"].(string)
	if !ok {
		return &tools.ToolResult{
			Success: false,
			Error:   "content field is required and must be a string",
		}, nil
	}

	maxLength := 200 // default
	if maxLenVal, exists := input.Data["max_length"]; exists {
		if maxLen, ok := maxLenVal.(float64); ok {
			maxLength = int(maxLen)
		} else if maxLenStr, ok := maxLenVal.(string); ok {
			if parsed, err := strconv.Atoi(maxLenStr); err == nil {
				maxLength = parsed
			}
		}
	}

	// Generate summarization prompt and execute
	result, err := d.summarizeContent(ctx, content, maxLength, llmProvider)
	if err != nil {
		return &tools.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("summarization failed: %v", err),
		}, nil
	}

	return result, nil
}
