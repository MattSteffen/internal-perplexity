package primary

import (
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/tools"
)

// PrimaryAgentSchema defines the schema for the primary agent
var PrimaryAgentSchema = agents.AgentSchema{
	Name:        "primary",
	Description: "Main orchestrator agent that handles user interactions and coordinates sub-agents",
	Input: agents.InputSchema{
		Required: []string{"query"},
		Optional: []string{"context", "parameters", "data"},
		Types: map[string]string{
			"query":      "string",
			"context":    "object",
			"parameters": "object",
			"data":       "object",
		},
		Examples: map[string]interface{}{
			"query": "Summarize these documents and find related research",
			"context": map[string]interface{}{
				"model":   "gpt-4",
				"api_key": "sk-...",
				"timeout": 300,
			},
		},
	},
	Output: agents.OutputSchema{
		Type: "object",
		Structure: map[string]interface{}{
			"task":            "string", // Task type that was executed
			"result":          "object", // Task-specific result
			"orchestrator":    "string", // Always "primary_agent"
			"execution_path":  "string", // Path of agent execution
			"sub_agents_used": "array",  // List of sub-agents used
		},
		Description: "Structured response containing task results and execution metadata",
		Examples: []interface{}{
			map[string]interface{}{
				"task": "summarize_documents",
				"result": map[string]interface{}{
					"summary": "Generated summary...",
					"metadata": map[string]interface{}{
						"content_count":   3,
						"combined_length": 1500,
					},
				},
				"orchestrator":    "primary_agent",
				"execution_path":  "primary -> summary",
				"sub_agents_used": []string{"summary"},
			},
		},
	},
	Version: "1.0.0",
	Author:  "Internal Perplexity Team",
}

// PrimaryAgent represents the main orchestrator agent
type PrimaryAgent struct {
	systemPromptManager *agents.SystemPromptManager
	taskPlanner         *agents.TaskPlanner
	decisionEngine      *agents.DecisionEngine
	executionEngine     *agents.ExecutionEngine
	systemPrompt        *agents.SystemPrompt
	stats               agents.AgentStats
	subAgents           map[string]agents.Agent
	toolRegistry        *tools.Registry
}

// NewPrimaryAgent creates a new primary agent
func NewPrimaryAgent(subAgents map[string]agents.Agent, toolRegistry *tools.Registry) *PrimaryAgent {
	systemPromptManager := agents.NewSystemPromptManager()
	taskPlanner := agents.NewTaskPlanner(systemPromptManager)
	decisionEngine := agents.NewDecisionEngine(systemPromptManager)
	executionEngine := agents.NewExecutionEngine(toolRegistry)

	agent := &PrimaryAgent{
		systemPromptManager: systemPromptManager,
		taskPlanner:         taskPlanner,
		decisionEngine:      decisionEngine,
		executionEngine:     executionEngine,
		systemPrompt:        systemPromptManager.GetPrompt("primary", agents.ContextGeneral),
		subAgents:           subAgents,
		toolRegistry:        toolRegistry,
		stats: agents.AgentStats{
			TotalExecutions: 0,
			SuccessRate:     1.0,
		},
	}

	return agent
}

// GetSchema returns the agent's schema
func (p *PrimaryAgent) GetSchema() agents.AgentSchema {
	return PrimaryAgentSchema
}

// SetToolRegistry sets the tool registry for the execution engine
func (p *PrimaryAgent) SetToolRegistry(registry *tools.Registry) {
	p.toolRegistry = registry
	// The execution engine already has the tool registry set in the constructor
}

// GetAvailableTools returns the names of available tools
func (p *PrimaryAgent) GetAvailableTools() []string {
	if p.toolRegistry == nil {
		return []string{}
	}

	tools := p.toolRegistry.List()
	names := make([]string, 0, len(tools))
	for name := range tools {
		names = append(names, name)
	}
	return names
}

// GetAvailableToolDescriptions returns descriptions of available tools
func (p *PrimaryAgent) GetAvailableToolDescriptions() map[string]string {
	if p.toolRegistry == nil {
		return map[string]string{}
	}

	tools := p.toolRegistry.List()
	descriptions := make(map[string]string)
	for name, tool := range tools {
		descriptions[name] = tool.Description()
	}
	return descriptions
}

// ValidateInput validates the input according to the schema
func (p *PrimaryAgent) ValidateInput(input *agents.AgentInput) error {
	if input.Query == "" {
		return agents.NewValidationError("query", "Query field is required", "MISSING_REQUIRED_FIELD", input.Query)
	}
	return nil
}

// ValidateOutput validates the output according to the schema
func (p *PrimaryAgent) ValidateOutput(output *agents.AgentResult) error {
	if output.Content == nil {
		return agents.NewValidationError("content", "Output content cannot be nil", "INVALID_OUTPUT", output.Content)
	}
	return nil
}
