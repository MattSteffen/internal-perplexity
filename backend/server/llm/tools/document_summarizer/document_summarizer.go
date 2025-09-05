package document_summarizer

import (
	"context"
	"fmt"
	"strconv"

	"internal-perplexity/server/llm/models/shared"
	"internal-perplexity/server/llm/tools"
)

// DocumentSummarizer provides document summarization capabilities
type DocumentSummarizer struct {
	llmClient shared.LLMProvider
}

// NewDocumentSummarizer creates a new document summarizer tool
func NewDocumentSummarizer(llmClient shared.LLMProvider) *DocumentSummarizer {
	return &DocumentSummarizer{
		llmClient: llmClient,
	}
}

// Name returns the tool name
func (d *DocumentSummarizer) Name() string {
	return "document_summarizer"
}

// Description returns the tool description
func (d *DocumentSummarizer) Description() string {
	return "Summarizes documents and text content to specified length"
}

// Schema returns the JSON schema for input validation
func (d *DocumentSummarizer) Schema() *tools.ToolSchema {
	return &tools.ToolSchema{
		Type: "object",
		Properties: map[string]interface{}{
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
		Required: []string{"content"},
	}
}

// Execute performs document summarization using LLM
func (d *DocumentSummarizer) Execute(ctx context.Context, input *tools.ToolInput) (*tools.ToolResult, error) {
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

	// Single LLM call for summarization
	prompt := fmt.Sprintf("Summarize the following text in %d words or less. Focus on the key points and main ideas:\n\n%s", maxLength, content)

	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are a professional document summarizer. Provide concise, accurate summaries that capture the essential information.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	resp, err := d.llmClient.Complete(ctx, messages, shared.CompletionOptions{
		MaxTokens:   maxLength * 5, // Rough estimate: 5 tokens per word
		Temperature: 0.3,           // Lower temperature for more focused summaries
	})
	if err != nil {
		return &tools.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("LLM completion failed: %v", err),
		}, nil
	}

	// Count tokens used
	tokensUsed, _ := d.llmClient.CountTokens(messages)

	return &tools.ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"summary":         resp.Content,
			"original_length": len(content),
			"summary_length":  len(resp.Content),
		},
		Stats: tools.ToolStats{
			TokensUsed: tokensUsed + resp.Usage.TotalTokens,
		},
	}, nil
}
