package researcher

import (
	"context"
	"errors"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// Execute performs web research and information gathering (NOT YET IMPLEMENTED)
func (r *ResearcherAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	start := time.Now()

	// Researcher agent is not yet implemented
	err := errors.New("researcher agent is not yet implemented - please use summary agent for current content analysis needs")

	return &agents.AgentResult{
		Success:  false,
		Duration: time.Since(start),
		Metadata: map[string]any{
			"error":              err.Error(),
			"agent_status":       "not_implemented",
			"recommended_agents": []string{"summary"},
		},
	}, err
}

// GetCapabilities returns the agent's capabilities
func (r *ResearcherAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "web_research",
			Description: "Automated web research and content discovery (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "source_validation",
			Description: "Source credibility assessment and validation (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "information_synthesis",
			Description: "Multi-source information synthesis and integration (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "credibility_assessment",
			Description: "Automated credibility scoring and ranking (NOT YET IMPLEMENTED)",
		},
		{
			Name:        "research_planning",
			Description: "Strategic research planning and execution (NOT YET IMPLEMENTED)",
		},
	}
}

// GetStats returns the agent's statistics
func (r *ResearcherAgent) GetStats() agents.AgentStats {
	return r.stats
}

// GetSystemPrompt returns the agent's system prompt
func (r *ResearcherAgent) GetSystemPrompt() *agents.SystemPrompt {
	return r.systemPrompt
}
