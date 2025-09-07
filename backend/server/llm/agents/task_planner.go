package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"internal-perplexity/server/llm/providers/shared"
)

// TaskPlanner handles intelligent task creation and decomposition
type TaskPlanner struct {
	systemPromptManager *SystemPromptManager
	availableTools      []string
	availableSubAgents  []string
}

// NewTaskPlanner creates a new task planner
func NewTaskPlanner(systemPromptManager *SystemPromptManager) *TaskPlanner {
	return &TaskPlanner{
		systemPromptManager: systemPromptManager,
		availableTools:      []string{"calculator", "document_summarizer", "retriever"},
		availableSubAgents:  []string{"summary", "researcher", "analyst"},
	}
}

// PlanTasks analyzes a user query and creates an execution plan
func (tp *TaskPlanner) PlanTasks(ctx context.Context, input *AgentInput, llmProvider shared.LLMProvider) (*ExecutionPlan, error) {
	// Quick analysis for simple patterns
	if plan := tp.analyzeSimplePatterns(input); plan != nil {
		return plan, nil
	}

	// Use LLM for complex task analysis
	return tp.analyzeWithLLM(ctx, input, llmProvider)
}

// analyzeSimplePatterns handles common patterns that don't need LLM analysis
func (tp *TaskPlanner) analyzeSimplePatterns(input *AgentInput) *ExecutionPlan {
	query := strings.ToLower(input.Query)

	// Pattern: Document summarization
	if tp.matchesSummarizationPattern(query) {
		return tp.createSummarizationPlan(input)
	}

	// Pattern: Mathematical calculations
	if tp.matchesCalculationPattern(query) {
		return tp.createCalculationPlan(input)
	}

	// Pattern: Information search/retrieval
	if tp.matchesSearchPattern(query) {
		return tp.createSearchPlan(input)
	}

	// Pattern: Simple questions (direct response)
	if tp.matchesSimpleQuestionPattern(query) {
		return tp.createDirectResponsePlan(input)
	}

	return nil // Complex query needs LLM analysis
}

// analyzeWithLLM uses LLM to analyze complex queries and create task plans
func (tp *TaskPlanner) analyzeWithLLM(ctx context.Context, input *AgentInput, llmProvider shared.LLMProvider) (*ExecutionPlan, error) {
	taskPrompt := tp.systemPromptManager.GetPrompt("task_creation", ContextGeneral)

	messages := []shared.Message{
		{
			Role:    "system",
			Content: taskPrompt.GetFullPrompt(),
		},
		{
			Role:    "user",
			Content: tp.buildTaskAnalysisPrompt(input),
		},
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			MaxTokens:   1000,
			Temperature: 0.3, // Lower temperature for more consistent task planning
		},
		Model: "gpt-4",
	}

	resp, err := llmProvider.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query with LLM: %w", err)
	}

	// Parse the LLM response to extract task plan
	return tp.parseTaskPlanFromResponse(resp.Content, input)
}

// buildTaskAnalysisPrompt creates a prompt for the LLM to analyze the query
func (tp *TaskPlanner) buildTaskAnalysisPrompt(input *AgentInput) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Analyze this user query and create a detailed execution plan: \"%s\"\n\n", input.Query))

	prompt.WriteString("Available Tools:\n")
	for _, tool := range tp.availableTools {
		prompt.WriteString(fmt.Sprintf("- %s\n", tool))
	}

	prompt.WriteString("\nAvailable Sub-Agents:\n")
	for _, agent := range tp.availableSubAgents {
		prompt.WriteString(fmt.Sprintf("- %s\n", agent))
	}

	prompt.WriteString("\nExecution Patterns:\n")
	prompt.WriteString("- sequential: Execute tasks one after another\n")
	prompt.WriteString("- parallel: Execute tasks simultaneously\n")
	prompt.WriteString("- map_reduce: Process multiple items in parallel, then combine results\n")
	prompt.WriteString("- direct: Provide direct response without additional tasks\n")

	prompt.WriteString("\nPlease respond with a JSON object containing:\n")
	prompt.WriteString("{\n")
	prompt.WriteString("  \"pattern\": \"execution_pattern\",\n")
	prompt.WriteString("  \"reasoning\": \"why_this_pattern_was_chosen\",\n")
	prompt.WriteString("  \"tasks\": [\n")
	prompt.WriteString("    {\n")
	prompt.WriteString("      \"id\": \"task_id\",\n")
	prompt.WriteString("      \"type\": \"tool|subagent|response\",\n")
	prompt.WriteString("      \"name\": \"tool_or_agent_name\",\n")
	prompt.WriteString("      \"description\": \"human_readable_description\",\n")
	prompt.WriteString("      \"priority\": 1,\n")
	prompt.WriteString("      \"depends_on\": [\"task_id\"],\n")
	prompt.WriteString("      \"input\": {\"key\": \"value\"}\n")
	prompt.WriteString("    }\n")
	prompt.WriteString("  ],\n")
	prompt.WriteString("  \"estimated_tokens\": 1000\n")
	prompt.WriteString("}\n")

	return prompt.String()
}

// parseTaskPlanFromResponse extracts the task plan from LLM response
func (tp *TaskPlanner) parseTaskPlanFromResponse(response string, input *AgentInput) (*ExecutionPlan, error) {
	// Try to extract JSON from the response
	jsonPattern := regexp.MustCompile(`\{.*\}`)
	jsonMatch := jsonPattern.FindString(response)

	if jsonMatch == "" {
		return nil, fmt.Errorf("no JSON found in LLM response")
	}

	var plan ExecutionPlan
	if err := json.Unmarshal([]byte(jsonMatch), &plan); err != nil {
		return nil, fmt.Errorf("failed to parse task plan JSON: %w", err)
	}

	// Validate the plan
	if err := tp.validateExecutionPlan(&plan); err != nil {
		return nil, fmt.Errorf("invalid execution plan: %w", err)
	}

	// Enrich tasks with input data if needed
	tp.enrichTasksWithInputData(&plan, input)

	return &plan, nil
}

// validateExecutionPlan checks if the execution plan is valid
func (tp *TaskPlanner) validateExecutionPlan(plan *ExecutionPlan) error {
	if len(plan.Tasks) == 0 && plan.Pattern != PatternDirect {
		return fmt.Errorf("execution plan must have tasks unless pattern is 'direct'")
	}

	// Validate task dependencies
	taskIDs := make(map[string]bool)
	for _, task := range plan.Tasks {
		if task.ID == "" {
			return fmt.Errorf("task must have an ID")
		}
		if taskIDs[task.ID] {
			return fmt.Errorf("duplicate task ID: %s", task.ID)
		}
		taskIDs[task.ID] = true

		// Validate task type
		if task.Type != TaskTypeTool && task.Type != TaskTypeSubAgent && task.Type != TaskTypeResponse {
			return fmt.Errorf("invalid task type: %s", task.Type)
		}

		// Validate tool/agent availability
		if task.Type == TaskTypeTool && !tp.isToolAvailable(task.Name) {
			return fmt.Errorf("tool not available: %s", task.Name)
		}
		if task.Type == TaskTypeSubAgent && !tp.isSubAgentAvailable(task.Name) {
			return fmt.Errorf("sub-agent not available: %s", task.Name)
		}

		// Validate dependencies
		for _, dep := range task.DependsOn {
			if !taskIDs[dep] {
				return fmt.Errorf("task %s depends on non-existent task %s", task.ID, dep)
			}
		}
	}

	return nil
}

// enrichTasksWithInputData adds input data to tasks that need it
func (tp *TaskPlanner) enrichTasksWithInputData(plan *ExecutionPlan, input *AgentInput) {
	for i := range plan.Tasks {
		task := &plan.Tasks[i]

		// If task input is empty, inherit from main input
		if len(task.Input) == 0 {
			task.Input = input.Data
		}

		// Add query context if not present
		if _, hasQuery := task.Input["query"]; !hasQuery && input.Query != "" {
			task.Input["query"] = input.Query
		}
	}
}

// Pattern matching functions
func (tp *TaskPlanner) matchesSummarizationPattern(query string) bool {
	patterns := []string{
		"summarize", "summary", "summarise", "condense", "abstract",
		"extract key points", "key findings", "main points",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	return false
}

func (tp *TaskPlanner) matchesCalculationPattern(query string) bool {
	patterns := []string{
		"calculate", "compute", "math", "arithmetic", "equation",
		"add", "subtract", "multiply", "divide", "solve",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	return false
}

func (tp *TaskPlanner) matchesSearchPattern(query string) bool {
	patterns := []string{
		"search", "find", "look for", "retrieve", "get information",
		"research", "investigate", "explore",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	return false
}

func (tp *TaskPlanner) matchesSimpleQuestionPattern(query string) bool {
	// Simple questions that can be answered directly
	simplePatterns := []string{
		"what is", "how does", "why does", "when did", "where is",
		"who is", "explain", "define", "describe",
	}

	for _, pattern := range simplePatterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}

	// Short queries (likely simple questions)
	return len(strings.Fields(query)) < 10
}

// Create simple pattern-based plans
func (tp *TaskPlanner) createSummarizationPlan(input *AgentInput) *ExecutionPlan {
	return &ExecutionPlan{
		Tasks: []Task{
			{
				ID:          "summarize_content",
				Type:        TaskTypeSubAgent,
				Name:        "summary",
				Description: "Summarize the provided content",
				Input:       input.Data,
				Priority:    1,
				DependsOn:   []string{},
			},
		},
		Pattern:         PatternSequential,
		Reasoning:       "Query involves summarization - using summary sub-agent",
		EstimatedTokens: 1000,
	}
}

func (tp *TaskPlanner) createCalculationPlan(input *AgentInput) *ExecutionPlan {
	return &ExecutionPlan{
		Tasks: []Task{
			{
				ID:          "perform_calculation",
				Type:        TaskTypeTool,
				Name:        "calculator",
				Description: "Perform mathematical calculation",
				Input:       input.Data,
				Priority:    1,
				DependsOn:   []string{},
			},
		},
		Pattern:         PatternSequential,
		Reasoning:       "Query involves mathematical calculation - using calculator tool",
		EstimatedTokens: 500,
	}
}

func (tp *TaskPlanner) createSearchPlan(input *AgentInput) *ExecutionPlan {
	return &ExecutionPlan{
		Tasks: []Task{
			{
				ID:          "search_information",
				Type:        TaskTypeTool,
				Name:        "retriever",
				Description: "Search for requested information",
				Input:       input.Data,
				Priority:    1,
				DependsOn:   []string{},
			},
		},
		Pattern:         PatternSequential,
		Reasoning:       "Query involves information search - using retriever tool",
		EstimatedTokens: 800,
	}
}

func (tp *TaskPlanner) createDirectResponsePlan(input *AgentInput) *ExecutionPlan {
	return &ExecutionPlan{
		Tasks:           []Task{},
		Pattern:         PatternDirect,
		FinalResponse:   input.Query, // Will be processed by LLM
		Reasoning:       "Simple query that can be answered directly",
		EstimatedTokens: 500,
	}
}

// Utility functions
func (tp *TaskPlanner) isToolAvailable(toolName string) bool {
	for _, tool := range tp.availableTools {
		if tool == toolName {
			return true
		}
	}
	return false
}

func (tp *TaskPlanner) isSubAgentAvailable(agentName string) bool {
	for _, agent := range tp.availableSubAgents {
		if agent == agentName {
			return true
		}
	}
	return false
}

// SetAvailableTools updates the list of available tools
func (tp *TaskPlanner) SetAvailableTools(tools []string) {
	tp.availableTools = make([]string, len(tools))
	copy(tp.availableTools, tools)
}

// SetAvailableSubAgents updates the list of available sub-agents
func (tp *TaskPlanner) SetAvailableSubAgents(agents []string) {
	tp.availableSubAgents = make([]string, len(agents))
	copy(tp.availableSubAgents, agents)
}

// GetAvailableTools returns the list of available tools
func (tp *TaskPlanner) GetAvailableTools() []string {
	tools := make([]string, len(tp.availableTools))
	copy(tools, tp.availableTools)
	return tools
}

// GetAvailableSubAgents returns the list of available sub-agents
func (tp *TaskPlanner) GetAvailableSubAgents() []string {
	agents := make([]string, len(tp.availableSubAgents))
	copy(agents, tp.availableSubAgents)
	return agents
}
