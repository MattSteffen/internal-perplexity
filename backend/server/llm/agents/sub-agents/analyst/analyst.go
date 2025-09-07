package analyst

import (
	"context"
	"errors"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// Execute performs data analysis (NOT YET IMPLEMENTED)
func (a *AnalystAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	start := time.Now()

	// Analyst agent is not yet implemented
	err := errors.New("analyst agent is not yet implemented - please use summary and researcher agents for current analysis needs")

	return &agents.AgentResult{
		Success:  false,
		Duration: time.Since(start),
		Metadata: map[string]any{
			"error":              err.Error(),
			"agent_status":       "not_implemented",
			"recommended_agents": []string{"summary", "researcher"},
		},
	}, err
}

// GetCapabilities returns the agent's capabilities
func (a *AnalystAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "statistical_analysis",
			Description: "Advanced statistical analysis and modeling (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "pattern_recognition",
			Description: "Automated pattern recognition and anomaly detection (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "predictive_modeling",
			Description: "Predictive analytics and forecasting (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "data_visualization",
			Description: "Data visualization generation (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "insight_generation",
			Description: "Automated insight extraction and interpretation (NOT YET IMPLEMENTED)",
		},
	}
}

// GetStats returns the agent's statistics
func (a *AnalystAgent) GetStats() agents.AgentStats {
	return a.stats
}

// GetSystemPrompt returns the agent's system prompt
func (a *AnalystAgent) GetSystemPrompt() *agents.SystemPrompt {
	return a.systemPrompt
}
