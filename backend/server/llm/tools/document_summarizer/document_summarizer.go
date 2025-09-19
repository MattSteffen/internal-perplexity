package document_summarizer

import (
	"context"
	"fmt"

	providershared "internal-perplexity/server/llm/providers/shared"
	toolshared "internal-perplexity/server/llm/tools/shared"
)

// summarizeContent performs the actual LLM-based summarization
func (d *DocumentSummarizer) summarizeContent(ctx context.Context, content string, maxLength int, llmProvider providershared.LLMProvider) (*toolshared.ToolResult, error) {
	// Single LLM call for summarization
	prompt := fmt.Sprintf("Summarize the following text in %d words or less. Focus on the key points and main ideas:\n\n%s", maxLength, content)

	messages := []providershared.Message{
		{
			Role:    "system",
			Content: "You are a professional document summarizer. Provide concise, accurate summaries that capture the essential information.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	req := &providershared.CompletionRequest{
		Messages: messages,
		Options: providershared.CompletionOptions{
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

	return &toolshared.ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"summary":         resp.Content,
			"original_length": len(content),
			"summary_length":  len(resp.Content),
		},
		Stats: toolshared.ToolStats{
			TokensUsed: tokensUsed + resp.Usage.TotalTokens,
		},
	}, nil
}
