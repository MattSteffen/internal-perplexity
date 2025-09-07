package primary

import (
	"context"
	"fmt"
	"strings"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
)

// Execute handles user queries and orchestrates sub-agent execution
func (p *PrimaryAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	start := time.Now()

	// Validate input
	if err := p.ValidateInput(input); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": err.Error(),
			},
		}, err
	}

	// Analyze query and determine execution approach
	executionSpec, err := p.analyzeQuery(input, llmProvider)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("failed to analyze query: %v", err),
			},
		}, err
	}

	// Execute sub-agents according to specification
	result, err := p.executeSubAgents(ctx, executionSpec, llmProvider)
	if err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("failed to execute sub-agents: %v", err),
			},
		}, err
	}

	// Validate output
	if err := p.ValidateOutput(result); err != nil {
		return &agents.AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("output validation failed: %v", err),
			},
		}, err
	}

	// Update stats
	subAgentsUsed := len(result.Metadata["sub_agents_used"].([]string))
	toolsUsed := len(result.Metadata["tools_used"].([]string))
	p.updateStats(result, len(executionSpec.Groups), toolsUsed, subAgentsUsed)

	result.Duration = time.Since(start)
	return result, nil
}

// analyzeQuery uses LLM to analyze the user query and determine which sub-agents and tools to call
func (p *PrimaryAgent) analyzeQuery(input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.SubAgentExecution, error) {
	availableSubAgents := p.getAvailableSubAgentNames()
	availableTools := p.GetAvailableTools()
	toolDescriptions := p.GetAvailableToolDescriptions()

	prompt := p.buildAnalysisPrompt(input.Query, availableSubAgents, availableTools, toolDescriptions)

	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are an expert at analyzing user queries and determining which AI agents and tools should handle different parts of the request. You can use both sub-agents and tools to accomplish complex tasks. You must respond with a valid JSON specification for execution.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			Model:       "gpt-4",
			MaxTokens:   1000,
			Temperature: 0.2, // Low temperature for consistent analysis
		},
	}

	resp, err := llmProvider.Complete(context.Background(), req)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query: %w", err)
	}

	return p.parseExecutionSpecification(resp.Content)
}

// buildAnalysisPrompt creates a prompt for the LLM to analyze the query
func (p *PrimaryAgent) buildAnalysisPrompt(query string, availableAgents []string, availableTools []string, toolDescriptions map[string]string) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Analyze this user query and determine which sub-agents and tools should handle it: \"%s\"\n\n", query))

	// Add available sub-agents
	prompt.WriteString("Available Sub-Agents:\n")
	subAgentDescriptions := map[string]string{
		"summary":    "Handles document summarization, content analysis, and key point extraction",
		"analyst":    "Performs data analysis, statistical processing, and insight generation (NOT YET IMPLEMENTED)",
		"researcher": "Conducts web research, information gathering, and source validation (NOT YET IMPLEMENTED)",
		"synthesis":  "Combines and synthesizes outputs from multiple agents into coherent responses",
	}

	for _, agent := range availableAgents {
		if desc, exists := subAgentDescriptions[agent]; exists {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", agent, desc))
		}
	}

	// Add available tools
	prompt.WriteString("\nAvailable Tools:\n")
	for _, tool := range availableTools {
		if desc, exists := toolDescriptions[tool]; exists {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", tool, desc))
		} else {
			prompt.WriteString(fmt.Sprintf("- %s: General purpose tool\n", tool))
		}
	}

	prompt.WriteString("\nDetermine how to decompose this query into sub-agent and tool calls. Consider:\n")
	prompt.WriteString("1. Which sub-agents or tools are needed for different parts of the query?\n")
	prompt.WriteString("2. Which parts can be executed in parallel?\n")
	prompt.WriteString("3. What are the dependencies between different calls?\n")
	prompt.WriteString("4. How should the results be combined?\n")
	prompt.WriteString("5. When to use a tool vs a sub-agent (tools are typically faster for specific tasks)\n\n")

	prompt.WriteString("You can call sub-agents, tools, or both. Tools are generally faster for specific computations.\n\n")

	prompt.WriteString("Respond with a JSON object specifying the execution plan:\n")
	prompt.WriteString("{\n")
	prompt.WriteString("  \"groups\": [\n")
	prompt.WriteString("    {\n")
	prompt.WriteString("      \"calls\": [\n")
	prompt.WriteString("        {\n")
	prompt.WriteString("          \"name\": \"sub_agent_or_tool_name\",\n")
	prompt.WriteString("          \"type\": \"subagent|tool\",\n")
	prompt.WriteString("          \"input\": {\"query\": \"specific query for this agent/tool\"},\n")
	prompt.WriteString("          \"output_key\": \"key_to_store_result\",\n")
	prompt.WriteString("          \"description\": \"what this agent/tool call does\"\n")
	prompt.WriteString("        }\n")
	prompt.WriteString("      ],\n")
	prompt.WriteString("      \"description\": \"description of this execution group\"\n")
	prompt.WriteString("    }\n")
	prompt.WriteString("  ],\n")
	prompt.WriteString("  \"description\": \"overall description of the execution plan\"\n")
	prompt.WriteString("}\n")

	return prompt.String()
}

// parseExecutionSpecification parses the LLM response into an execution specification
func (p *PrimaryAgent) parseExecutionSpecification(response string) (*agents.SubAgentExecution, error) {
	// For now, return a simple specification - in production this would parse JSON
	return &agents.SubAgentExecution{
		Groups: []agents.ExecutionGroup{
			{
				Calls: []agents.SubAgentCall{
					{
						Name:        "summary",
						Input:       map[string]interface{}{"query": "analyze and summarize"},
						OutputKey:   "summary_result",
						Description: "Summarize the main points",
					},
				},
				Description: "Execute summary agent",
			},
		},
		Description: "Simple single-agent execution",
	}, nil
}

// executeSubAgents executes the sub-agents and tools according to the specification
func (p *PrimaryAgent) executeSubAgents(ctx context.Context, spec *agents.SubAgentExecution, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	results := make(map[string]interface{})
	subAgentsUsed := []string{}
	toolsUsed := []string{}

	// Execute each group sequentially
	for groupIndex, group := range spec.Groups {
		groupResults := make(map[string]interface{})

		// Execute calls in this group in parallel (if multiple)
		if len(group.Calls) == 1 {
			// Single call
			call := group.Calls[0]
			result, err := p.executeSubAgentCall(ctx, call, llmProvider)
			if err != nil {
				return nil, fmt.Errorf("failed to execute %s: %w", call.Name, err)
			}
			groupResults[call.OutputKey] = result

			// Track usage by type
			switch call.Type {
			case "subagent":
				subAgentsUsed = append(subAgentsUsed, call.Name)
			case "tool":
				toolsUsed = append(toolsUsed, call.Name)
			}
		} else {
			// Parallel execution (simplified - not truly parallel in this implementation)
			for _, call := range group.Calls {
				result, err := p.executeSubAgentCall(ctx, call, llmProvider)
				if err != nil {
					return nil, fmt.Errorf("failed to execute %s: %w", call.Name, err)
				}
				groupResults[call.OutputKey] = result

				// Track usage by type
				switch call.Type {
				case "subagent":
					subAgentsUsed = append(subAgentsUsed, call.Name)
				case "tool":
					toolsUsed = append(toolsUsed, call.Name)
				}
			}
		}

		// Store group results
		results[fmt.Sprintf("group_%d", groupIndex)] = groupResults
	}

	return &agents.AgentResult{
		Content: map[string]interface{}{
			"results":         results,
			"sub_agents_used": subAgentsUsed,
			"tools_used":      toolsUsed,
			"execution_spec":  spec,
		},
		Success: true,
		Metadata: map[string]any{
			"execution_groups": len(spec.Groups),
			"total_sub_agents": len(subAgentsUsed),
			"total_tools":      len(toolsUsed),
			"sub_agents_used":  subAgentsUsed,
			"tools_used":       toolsUsed,
		},
	}, nil
}

// executeSubAgentCall executes a single sub-agent or tool call
func (p *PrimaryAgent) executeSubAgentCall(ctx context.Context, call agents.SubAgentCall, llmProvider shared.LLMProvider) (interface{}, error) {
	switch call.Type {
	case "subagent":
		return p.executeSubAgent(ctx, call, llmProvider)
	case "tool":
		return p.executeTool(ctx, call, llmProvider)
	default:
		return nil, fmt.Errorf("unknown call type: %s", call.Type)
	}
}

// executeSubAgent executes a single sub-agent call
func (p *PrimaryAgent) executeSubAgent(ctx context.Context, call agents.SubAgentCall, llmProvider shared.LLMProvider) (interface{}, error) {
	subAgent, exists := p.subAgents[call.Name]
	if !exists {
		return nil, fmt.Errorf("sub-agent %s not found", call.Name)
	}

	// Create input for the sub-agent
	input := &agents.AgentInput{
		Query: call.Input["query"].(string),
		Data:  call.Input,
	}

	// Execute the sub-agent
	result, err := subAgent.Execute(ctx, input, llmProvider)
	if err != nil {
		return nil, fmt.Errorf("sub-agent %s execution failed: %w", call.Name, err)
	}

	return result.Content, nil
}

// executeTool executes a single tool call
func (p *PrimaryAgent) executeTool(ctx context.Context, call agents.SubAgentCall, llmProvider shared.LLMProvider) (interface{}, error) {
	if p.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry not available")
	}

	// Create tool input
	toolInput := &tools.ToolInput{
		Name: call.Name,
		Data: call.Input,
	}

	// Execute the tool
	result, err := p.toolRegistry.Execute(ctx, toolInput, llmProvider)
	if err != nil {
		return nil, fmt.Errorf("tool %s execution failed: %w", call.Name, err)
	}

	if !result.Success {
		return nil, fmt.Errorf("tool %s failed: %s", call.Name, result.Error)
	}

	return result.Data, nil
}

// getAvailableSubAgentNames returns the names of available sub-agents
func (p *PrimaryAgent) getAvailableSubAgentNames() []string {
	names := make([]string, 0, len(p.subAgents))
	for name := range p.subAgents {
		names = append(names, name)
	}
	return names
}

// CreateExecutionPlan creates an execution plan for the given input
func (p *PrimaryAgent) CreateExecutionPlan(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.ExecutionPlan, error) {
	// Update available tools and sub-agents in task planner
	p.taskPlanner.SetAvailableTools(p.GetAvailableTools())
	p.taskPlanner.SetAvailableSubAgents(p.getAvailableSubAgentNames())

	// Use task planner to create execution plan
	return p.taskPlanner.PlanTasks(ctx, input, llmProvider)
}

// ExecutePlan executes the given execution plan
func (p *PrimaryAgent) ExecutePlan(ctx context.Context, plan *agents.ExecutionPlan, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	return p.executionEngine.ExecutePlan(ctx, plan, llmProvider)
}

// GetCapabilities returns the agent's capabilities
func (p *PrimaryAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{
			Name:        "intelligent_orchestration",
			Description: "Intelligent task decomposition and execution planning",
		},
		{
			Name:        "multi_agent_coordination",
			Description: "Coordinate multiple sub-agents for complex workflows",
		},
		{
			Name:        "tool_execution",
			Description: "Direct execution of tools for specific computational tasks",
		},
		{
			Name:        "hybrid_execution",
			Description: "Combine sub-agents and tools in unified execution patterns",
		},
		{
			Name:        "adaptive_execution",
			Description: "Adapt execution patterns based on task requirements",
		},
	}
}

// GetStats returns the agent's statistics
func (p *PrimaryAgent) GetStats() agents.AgentStats {
	return p.stats
}

// GetSystemPrompt returns the agent's system prompt
func (p *PrimaryAgent) GetSystemPrompt() *agents.SystemPrompt {
	return p.systemPrompt
}

// updateStats updates the agent's statistics
func (p *PrimaryAgent) updateStats(result *agents.AgentResult, tasksCreated, toolsUsed, subAgentsUsed int) {
	p.stats.TotalExecutions++
	p.stats.TasksCreated += tasksCreated
	p.stats.ToolsCalled += toolsUsed
	p.stats.SubAgentsUsed += subAgentsUsed

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
	p.stats.AverageDuration = time.Duration((int64(p.stats.AverageDuration)*int64(p.stats.TotalExecutions-1) + int64(result.Duration)) / int64(p.stats.TotalExecutions))
}
