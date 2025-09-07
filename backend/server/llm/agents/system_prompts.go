package agents

import (
	"fmt"
	"strings"
)

// SystemPromptManager manages system prompts for different agent types and contexts
type SystemPromptManager struct {
	prompts map[string]*SystemPrompt
}

// NewSystemPromptManager creates a new system prompt manager
func NewSystemPromptManager() *SystemPromptManager {
	spm := &SystemPromptManager{
		prompts: make(map[string]*SystemPrompt),
	}
	spm.initializeDefaultPrompts()
	return spm
}

// GetPrompt retrieves a system prompt by name and context
func (spm *SystemPromptManager) GetPrompt(agentType string, context ContextType) *SystemPrompt {
	key := fmt.Sprintf("%s_%s", agentType, context)
	if prompt, exists := spm.prompts[key]; exists {
		return prompt
	}

	// Fallback to base agent type prompt
	if prompt, exists := spm.prompts[agentType]; exists {
		return prompt
	}

	// Ultimate fallback to generic prompt
	return spm.prompts["generic"]
}

// RegisterPrompt registers a custom system prompt
func (spm *SystemPromptManager) RegisterPrompt(key string, prompt *SystemPrompt) {
	spm.prompts[key] = prompt
}

// BuildPrompt dynamically builds a system prompt based on context
func (spm *SystemPromptManager) BuildPrompt(agentType string, context ContextType, tools []string, subAgents []string) *SystemPrompt {
	basePrompt := spm.GetPrompt(agentType, context)

	// Create a copy to avoid modifying the original
	prompt := &SystemPrompt{
		BasePrompt:   basePrompt.BasePrompt,
		Capabilities: make([]string, len(basePrompt.Capabilities)),
		Tools:        make([]string, len(tools)),
		SubAgents:    make([]string, len(subAgents)),
		Examples:     make([]string, len(basePrompt.Examples)),
		Constraints:  make(map[string]string),
	}

	copy(prompt.Capabilities, basePrompt.Capabilities)
	copy(prompt.Tools, tools)
	copy(prompt.SubAgents, subAgents)
	copy(prompt.Examples, basePrompt.Examples)

	// Copy constraints
	for k, v := range basePrompt.Constraints {
		prompt.Constraints[k] = v
	}

	return prompt
}

// ContextType defines different contexts for system prompts
type ContextType string

const (
	ContextGeneral    ContextType = "general"
	ContextTechnical  ContextType = "technical"
	ContextCreative   ContextType = "creative"
	ContextAnalytical ContextType = "analytical"
	ContextSummary    ContextType = "summary"
)

// initializeDefaultPrompts sets up default system prompts
func (spm *SystemPromptManager) initializeDefaultPrompts() {
	// Generic agent prompt
	spm.prompts["generic"] = &SystemPrompt{
		BasePrompt: `You are an intelligent AI agent capable of understanding and responding to user requests.
Focus on being helpful, accurate, and clear in your responses.`,
		Capabilities: []string{"general_assistance"},
		Tools:        []string{},
		SubAgents:    []string{},
		Examples:     []string{},
		Constraints:  map[string]string{},
	}

	// Primary agent prompts
	spm.prompts["primary_general"] = &SystemPrompt{
		BasePrompt: `You are an intelligent orchestrator agent that can break down complex tasks into manageable steps.
You have access to various tools and sub-agents to help you complete tasks efficiently.

Your capabilities include:
- Task decomposition and planning
- Multi-step workflow execution
- Tool and sub-agent coordination
- Intelligent decision making

When given a user query, analyze it carefully and create an appropriate execution plan:
1. Understand the user's intent and requirements
2. Identify what tools or sub-agents would be most helpful
3. Determine the best execution pattern (sequential, parallel, map-reduce, or direct)
4. Consider dependencies between tasks and optimize for efficiency
5. Provide clear reasoning for your decisions

Always prioritize user satisfaction and task completion quality.`,
		Capabilities: []string{
			"task_decomposition",
			"workflow_orchestration",
			"multi_agent_coordination",
			"tool_selection",
			"execution_planning",
		},
		Tools: []string{
			"calculator",
			"document_summarizer",
			"retriever",
		},
		SubAgents: []string{
			"summary",
			"researcher",
			"analyst",
		},
		Examples: []string{
			`Query: "Summarize these documents and calculate some statistics"
Plan: sequential - first summarize documents, then calculate statistics`,
			`Query: "Search for information on multiple topics"
Plan: parallel - search each topic simultaneously`,
			`Query: "Analyze data from multiple sources and provide insights"
Plan: map_reduce - analyze each source in parallel, then combine insights`,
		},
		Constraints: map[string]string{
			"max_tasks":           "10",
			"max_parallel_tasks":  "5",
			"timeout_minutes":     "30",
			"max_tokens_estimate": "50000",
		},
	}

	spm.prompts["primary_technical"] = &SystemPrompt{
		BasePrompt: `You are a technical specialist agent focused on software development, system architecture, and technical problem-solving.
You excel at analyzing code, debugging issues, designing solutions, and providing technical guidance.

Your technical expertise includes:
- Code analysis and review
- System architecture design
- Debugging and troubleshooting
- Performance optimization
- Security analysis
- Documentation generation

When working on technical tasks:
1. Carefully analyze the problem or code
2. Consider multiple solution approaches
3. Evaluate trade-offs between options
4. Provide detailed explanations and reasoning
5. Include code examples when helpful
6. Suggest best practices and patterns

Always ensure your solutions are practical, maintainable, and follow industry standards.`,
		Capabilities: []string{
			"code_analysis",
			"system_design",
			"debugging",
			"performance_optimization",
			"security_analysis",
		},
		Tools: []string{
			"calculator",
			"document_summarizer",
		},
		SubAgents: []string{
			"analyst",
		},
		Examples: []string{
			`Query: "Debug this performance issue in my Go application"
Plan: Use code analysis tools and provide optimization suggestions`,
			`Query: "Design a microservices architecture for my system"
Plan: Create detailed architecture diagrams and implementation plans`,
		},
		Constraints: map[string]string{
			"max_complexity":        "high",
			"detail_level":          "comprehensive",
			"include_examples":      "true",
			"follow_best_practices": "true",
		},
	}

	// Summary agent prompts
	spm.prompts["summary_general"] = &SystemPrompt{
		BasePrompt: `You are a professional document summarizer specialized in creating concise, accurate summaries.
Your expertise includes:
- Multi-document analysis and synthesis
- Key point extraction and prioritization
- Maintaining context and coherence
- Adapting summary style based on content type

Focus on:
1. Identifying the most important information
2. Preserving essential meaning and context
3. Using clear, concise language
4. Structuring summaries logically
5. Adapting length and detail to the content type

Always ensure your summaries are objective, accurate, and helpful to the reader.`,
		Capabilities: []string{
			"content_summarization",
			"key_point_extraction",
			"multi_document_synthesis",
			"context_preservation",
		},
		Tools:     []string{},
		SubAgents: []string{},
		Examples: []string{
			`Input: Multiple research papers about AI
Output: Concise summary highlighting key findings, methodologies, and conclusions`,
			`Input: Technical documentation
Output: Clear summary of functionality, usage, and important details`,
		},
		Constraints: map[string]string{
			"max_input_length":   "10000",
			"max_summary_ratio":  "0.2",
			"processing_timeout": "60s",
		},
	}

	// Task creation prompts for different contexts
	spm.prompts["task_creation_general"] = &SystemPrompt{
		BasePrompt: `You are an expert task planner. Your job is to break down user queries into specific, actionable tasks.

When creating tasks:
1. Analyze the query to understand the user's intent
2. Identify the specific actions needed to fulfill the request
3. Break complex tasks into smaller, manageable steps
4. Consider dependencies between tasks
5. Choose appropriate tools or sub-agents for each task
6. Determine the best execution pattern (sequential, parallel, etc.)

Create tasks that are:
- Specific and actionable
- Independent when possible
- Properly prioritized
- Realistic in scope and time requirements

Always provide clear reasoning for your task breakdown and execution strategy.`,
		Capabilities: []string{
			"task_analysis",
			"task_breakdown",
			"dependency_analysis",
			"execution_planning",
		},
		Tools:     []string{},
		SubAgents: []string{},
		Examples: []string{
			`Query: "Research and summarize the latest developments in quantum computing"
Tasks: [research_web, analyze_findings, create_summary]`,
		},
		Constraints: map[string]string{
			"max_tasks_per_query": "8",
			"max_task_complexity": "medium",
		},
	}
}

// GetFullPrompt returns the complete formatted system prompt
func (sp *SystemPrompt) GetFullPrompt() string {
	var builder strings.Builder

	// Add base prompt
	builder.WriteString(sp.BasePrompt)
	builder.WriteString("\n\n")

	// Add capabilities
	if len(sp.Capabilities) > 0 {
		builder.WriteString("Capabilities:\n")
		for _, cap := range sp.Capabilities {
			builder.WriteString(fmt.Sprintf("- %s\n", cap))
		}
		builder.WriteString("\n")
	}

	// Add tools
	if len(sp.Tools) > 0 {
		builder.WriteString("Available Tools:\n")
		for _, tool := range sp.Tools {
			builder.WriteString(fmt.Sprintf("- %s\n", tool))
		}
		builder.WriteString("\n")
	}

	// Add sub-agents
	if len(sp.SubAgents) > 0 {
		builder.WriteString("Available Sub-Agents:\n")
		for _, agent := range sp.SubAgents {
			builder.WriteString(fmt.Sprintf("- %s\n", agent))
		}
		builder.WriteString("\n")
	}

	// Add constraints
	if len(sp.Constraints) > 0 {
		builder.WriteString("Constraints:\n")
		for key, value := range sp.Constraints {
			builder.WriteString(fmt.Sprintf("- %s: %s\n", key, value))
		}
		builder.WriteString("\n")
	}

	// Add examples
	if len(sp.Examples) > 0 {
		builder.WriteString("Examples:\n")
		for _, example := range sp.Examples {
			builder.WriteString(fmt.Sprintf("- %s\n", example))
		}
	}

	return strings.TrimSpace(builder.String())
}

// ValidatePrompt validates that a system prompt has required components
func (sp *SystemPrompt) ValidatePrompt() error {
	if strings.TrimSpace(sp.BasePrompt) == "" {
		return fmt.Errorf("base prompt cannot be empty")
	}

	if len(sp.Capabilities) == 0 {
		return fmt.Errorf("at least one capability must be specified")
	}

	return nil
}
