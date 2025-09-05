package primary

import (
	"context"
	"fmt"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/agents/sub-agents/summary"
	"internal-perplexity/server/llm/providers/shared"
)

// PrimaryAgent is the main orchestrator agent
type PrimaryAgent struct {
	llmClient    shared.LLMProvider
	summaryAgent *summary.SummaryAgent
	stats        agents.AgentStats
}

// NewPrimaryAgent creates a new primary agent
func NewPrimaryAgent(llmClient shared.LLMProvider, summaryAgent *summary.SummaryAgent) *PrimaryAgent {
	return &PrimaryAgent{
		llmClient:    llmClient,
		summaryAgent: summaryAgent,
		stats: agents.AgentStats{
			TotalExecutions: 0,
			SuccessRate:     1.0,
		},
	}
}

// Execute orchestrates tasks based on the input
func (p *PrimaryAgent) Execute(ctx context.Context, input *agents.AgentInput) (*agents.AgentResult, error) {
	start := time.Now()

	// Validate input
	if err := p.validateInput(input); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Content:  nil,
			Duration: time.Since(start),
			Metadata: map[string]interface{}{
				"error": err.Error(),
			},
		}, nil
	}

	taskType := input.Data["task"].(string)

	switch taskType {
	case "summarize_documents":
		return p.executeDocumentSummary(ctx, input, start)
	case "general_query":
		return p.executeGeneralQuery(ctx, input, start)
	default:
		return &agents.AgentResult{
			Success:  false,
			Content:  nil,
			Duration: time.Since(start),
			Metadata: map[string]interface{}{
				"error": fmt.Sprintf("unsupported task type: %s", taskType),
			},
		}, nil
	}
}

// executeDocumentSummary handles document summarization tasks
func (p *PrimaryAgent) executeDocumentSummary(ctx context.Context, input *agents.AgentInput, start time.Time) (*agents.AgentResult, error) {
	// Prepare input for summary agent
	summaryInput := &agents.AgentInput{
		Data: input.Data,
		Context: map[string]interface{}{
			"orchestrator": "primary_agent",
			"task_type":    "document_summary",
		},
	}

	// Execute summary agent
	result, err := p.summaryAgent.Execute(ctx, summaryInput)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Content:  nil,
			Duration: time.Since(start),
			Metadata: map[string]interface{}{
				"error": fmt.Sprintf("summary agent failed: %v", err),
			},
		}, nil
	}

	// Update stats
	p.updateStats(result, time.Since(start))

	return &agents.AgentResult{
		Content: map[string]interface{}{
			"task":         "summarize_documents",
			"result":       result.Content,
			"orchestrator": "primary_agent",
		},
		Success:    result.Success,
		TokensUsed: result.TokensUsed,
		Duration:   time.Since(start),
		Metadata: map[string]interface{}{
			"sub_agent":      "summary",
			"execution_path": "primary -> summary",
		},
	}, nil
}

// executeGeneralQuery handles general query tasks
func (p *PrimaryAgent) executeGeneralQuery(ctx context.Context, input *agents.AgentInput, start time.Time) (*agents.AgentResult, error) {
	query := input.Data["query"].(string)

	// Use LLM for general queries
	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are a helpful AI assistant. Provide clear, accurate, and concise responses.", // TODO: Should be a good system prompt for the agent
		},
		{
			Role:    "user",
			Content: query,
		},
	}

	// Extract model and API key from input context
	model := "gpt-4" // TODO: Should be from a list of providers, this is a default for now
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
			MaxTokens:   10000,
			Temperature: 0.7,
		},
		Model:  model,
		APIKey: apiKey,
	}

	resp, err := p.llmClient.Complete(ctx, req)
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
	p.updateStats(&agents.AgentResult{Success: true, TokensUsed: resp.Usage.TotalTokens}, time.Since(start))

	return &agents.AgentResult{
		Content: map[string]interface{}{
			"task":   "general_query",
			"query":  query,
			"answer": resp.Content,
		},
		Success:    true,
		TokensUsed: resp.Usage.TotalTokens,
		Duration:   time.Since(start),
		Metadata: map[string]interface{}{
			"execution_path": "primary -> llm_direct",
		},
	}, nil
}

// validateInput validates the agent input
func (p *PrimaryAgent) validateInput(input *agents.AgentInput) error {
	if input.Data == nil {
		return fmt.Errorf("input data is required")
	}

	task, exists := input.Data["task"]
	if !exists {
		return fmt.Errorf("task field is required")
	}

	if taskStr, ok := task.(string); !ok {
		return fmt.Errorf("task must be a string")
	} else if taskStr == "" {
		return fmt.Errorf("task cannot be empty")
	}

	return nil
}

// updateStats updates the agent's statistics
func (p *PrimaryAgent) updateStats(result *agents.AgentResult, duration time.Duration) {
	p.stats.TotalExecutions++

	if tokens, ok := result.TokensUsed.(int); ok {
		p.stats.TotalTokens += tokens
	}

	// Update success rate
	if result.Success {
		p.stats.SuccessRate = (p.stats.SuccessRate*float64(p.stats.TotalExecutions-1) + 1.0) / float64(p.stats.TotalExecutions)
	} else {
		p.stats.SuccessRate = (p.stats.SuccessRate * float64(p.stats.TotalExecutions-1)) / float64(p.stats.TotalExecutions)
	}

	// Update average duration
	p.stats.AverageDuration = time.Duration((int64(p.stats.AverageDuration)*int64(p.stats.TotalExecutions-1) + int64(duration)) / int64(p.stats.TotalExecutions))
}

// GetCapabilities returns the agent's capabilities
func (p *PrimaryAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "document_summarization",
			Description: "Orchestrate document summarization using specialized sub-agents",
		},
		{
			Name:        "general_queries",
			Description: "Handle general queries and provide helpful responses",
		},
		{
			Name:        "task_orchestration",
			Description: "Coordinate multiple agents and tools for complex tasks",
		},
	}
}

// GetStats returns the agent's statistics
func (p *PrimaryAgent) GetStats() agents.AgentStats {
	return p.stats
}
