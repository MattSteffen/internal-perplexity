package document_summarizer

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
)

// summarizeContent performs the actual LLM-based summarization
func (d *DocumentSummarizer) summarizeContent(ctx context.Context, content string, maxLength int, llmProvider shared.LLMProvider) (*tools.ToolResult, error) {
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

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			Model:       "gpt-4",       // default model
			MaxTokens:   maxLength * 5, // Rough estimate: 5 tokens per word
			Temperature: 0.3,           // Lower temperature for more focused summaries
		},
	}

	resp, err := llmProvider.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %v", err)
	}

	// Count tokens used
	tokensUsed, _ := llmProvider.CountTokens(messages, req.Options.Model)

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
