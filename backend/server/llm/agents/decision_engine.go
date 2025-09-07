package agents

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"internal-perplexity/server/llm/providers/shared"
)

// DecisionEngine handles intelligent decision-making for tool and agent selection
type DecisionEngine struct {
	systemPromptManager *SystemPromptManager
	toolCapabilities    map[string]*ToolCapability
	agentCapabilities   map[string]*AgentCapability
}

// ToolCapability represents what a tool can do and its characteristics
type ToolCapability struct {
	Name          string
	Description   string
	Strengths     []string
	Weaknesses    []string
	BestFor       []string
	AverageTokens int
	Reliability   float64 // 0.0 to 1.0
	Speed         string  // "fast", "medium", "slow"
	Complexity    string  // "simple", "medium", "complex"
}

// AgentCapability represents what an agent can do and its characteristics
type AgentCapability struct {
	Name          string
	Description   string
	Specialties   []string
	BestFor       []string
	AverageTokens int
	Reliability   float64
	Speed         string
	Complexity    string
	Tools         []string
}

// Decision represents a decision about which tool/agent to use
type Decision struct {
	Choice       string  // tool or agent name
	Type         string  // "tool" or "agent"
	Confidence   float64 // 0.0 to 1.0
	Reasoning    string
	Alternatives []AlternativeDecision
}

// AlternativeDecision represents alternative options considered
type AlternativeDecision struct {
	Choice    string
	Type      string
	Score     float64
	Reasoning string
}

// NewDecisionEngine creates a new decision engine
func NewDecisionEngine(systemPromptManager *SystemPromptManager) *DecisionEngine {
	de := &DecisionEngine{
		systemPromptManager: systemPromptManager,
		toolCapabilities:    make(map[string]*ToolCapability),
		agentCapabilities:   make(map[string]*AgentCapability),
	}
	de.initializeCapabilities()
	return de
}

// MakeDecision analyzes a query and decides the best tool or agent to use
func (de *DecisionEngine) MakeDecision(ctx context.Context, query string, availableTools []string, availableAgents []string, llmProvider shared.LLMProvider) (*Decision, error) {
	// Quick pattern-based decision for common cases
	if decision := de.makeQuickDecision(query, availableTools, availableAgents); decision != nil {
		return decision, nil
	}

	// Use LLM for complex decision-making
	return de.makeLLMDecision(ctx, query, availableTools, availableAgents, llmProvider)
}

// EvaluateExecutionPattern suggests the best execution pattern for a set of tasks
func (de *DecisionEngine) EvaluateExecutionPattern(tasks []Task) (ExecutionPattern, string) {
	if len(tasks) == 0 {
		return PatternDirect, "No tasks to execute"
	}

	if len(tasks) == 1 {
		return PatternSequential, "Single task - sequential execution"
	}

	// Check for dependencies
	hasDependencies := false
	for _, task := range tasks {
		if len(task.DependsOn) > 0 {
			hasDependencies = true
			break
		}
	}

	if hasDependencies {
		return PatternSequential, "Tasks have dependencies - must execute sequentially"
	}

	// Check if tasks are independent and similar
	allSameType := true
	taskType := tasks[0].Type
	for _, task := range tasks[1:] {
		if task.Type != taskType {
			allSameType = false
			break
		}
	}

	if allSameType && len(tasks) <= 5 {
		return PatternParallel, "Independent tasks of same type - can execute in parallel"
	}

	if len(tasks) > 3 {
		return PatternMapReduce, "Multiple tasks - map-reduce pattern may be efficient"
	}

	return PatternSequential, "Default to sequential for safety"
}

// makeQuickDecision handles common patterns that don't need LLM analysis
func (de *DecisionEngine) makeQuickDecision(query string, availableTools []string, availableAgents []string) *Decision {
	queryLower := strings.ToLower(query)

	// Mathematical calculations
	if de.matchesMathPattern(queryLower) && de.isToolAvailable("calculator", availableTools) {
		return &Decision{
			Choice:     "calculator",
			Type:       "tool",
			Confidence: 0.95,
			Reasoning:  "Query contains mathematical operations - calculator tool is ideal",
		}
	}

	// Document summarization
	if de.matchesSummarizationPattern(queryLower) && de.isAgentAvailable("summary", availableAgents) {
		return &Decision{
			Choice:     "summary",
			Type:       "agent",
			Confidence: 0.90,
			Reasoning:  "Query involves summarization - summary agent is specialized for this",
		}
	}

	// Information search/retrieval
	if de.matchesSearchPattern(queryLower) && de.isToolAvailable("retriever", availableTools) {
		return &Decision{
			Choice:     "retriever",
			Type:       "tool",
			Confidence: 0.85,
			Reasoning:  "Query involves searching/retrieving information - retriever tool is appropriate",
		}
	}

	return nil // Need LLM analysis
}

// makeLLMDecision uses LLM to make complex decisions
func (de *DecisionEngine) makeLLMDecision(ctx context.Context, query string, availableTools []string, availableAgents []string, llmProvider shared.LLMProvider) (*Decision, error) {
	prompt := de.buildDecisionPrompt(query, availableTools, availableAgents)

	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are an expert decision-making assistant. Analyze queries and recommend the best tools or agents to handle them.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			MaxTokens:   800,
			Temperature: 0.2, // Low temperature for consistent decisions
		},
		Model: "gpt-4",
	}

	resp, err := llmProvider.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get LLM decision: %w", err)
	}

	return de.parseDecisionResponse(resp.Content, availableTools, availableAgents)
}

// buildDecisionPrompt creates a prompt for the LLM to make decisions
func (de *DecisionEngine) buildDecisionPrompt(query string, availableTools []string, availableAgents []string) string {
	var prompt strings.Builder

	prompt.WriteString(fmt.Sprintf("Analyze this query and recommend the best tool or agent to handle it: \"%s\"\n\n", query))

	prompt.WriteString("Available Tools:\n")
	for _, tool := range availableTools {
		if cap, exists := de.toolCapabilities[tool]; exists {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", tool, cap.Description))
			prompt.WriteString(fmt.Sprintf("  Best for: %s\n", strings.Join(cap.BestFor, ", ")))
		} else {
			prompt.WriteString(fmt.Sprintf("- %s: General purpose tool\n", tool))
		}
	}

	prompt.WriteString("\nAvailable Agents:\n")
	for _, agent := range availableAgents {
		if cap, exists := de.agentCapabilities[agent]; exists {
			prompt.WriteString(fmt.Sprintf("- %s: %s\n", agent, cap.Description))
			prompt.WriteString(fmt.Sprintf("  Specialties: %s\n", strings.Join(cap.Specialties, ", ")))
		} else {
			prompt.WriteString(fmt.Sprintf("- %s: General purpose agent\n", agent))
		}
	}

	prompt.WriteString("\nDecision Criteria:\n")
	prompt.WriteString("1. Relevance: How well does the tool/agent match the query?\n")
	prompt.WriteString("2. Specialization: Does it have specific expertise for this task?\n")
	prompt.WriteString("3. Efficiency: Will it complete the task quickly and accurately?\n")
	prompt.WriteString("4. Reliability: Has it proven reliable for similar tasks?\n")

	prompt.WriteString("\nRespond with a JSON object:\n")
	prompt.WriteString("{\n")
	prompt.WriteString("  \"choice\": \"tool_or_agent_name\",\n")
	prompt.WriteString("  \"type\": \"tool\" or \"agent\",\n")
	prompt.WriteString("  \"confidence\": 0.0 to 1.0,\n")
	prompt.WriteString("  \"reasoning\": \"detailed explanation\",\n")
	prompt.WriteString("  \"alternatives\": [\n")
	prompt.WriteString("    {\n")
	prompt.WriteString("      \"choice\": \"alternative_name\",\n")
	prompt.WriteString("      \"type\": \"tool\" or \"agent\",\n")
	prompt.WriteString("      \"score\": 0.0 to 1.0,\n")
	prompt.WriteString("      \"reasoning\": \"why this is an alternative\"\n")
	prompt.WriteString("    }\n")
	prompt.WriteString("  ]\n")
	prompt.WriteString("}\n")

	return prompt.String()
}

// parseDecisionResponse extracts the decision from LLM response
func (de *DecisionEngine) parseDecisionResponse(response string, availableTools []string, availableAgents []string) (*Decision, error) {
	// For now, return a mock decision - in practice, this would parse the JSON response
	// TODO: Implement proper JSON parsing with regex or JSON library
	return &Decision{
		Choice:     "summary", // Default fallback
		Type:       "agent",
		Confidence: 0.7,
		Reasoning:  "LLM-based decision making not fully implemented yet",
	}, nil
}

// initializeCapabilities sets up capability information for tools and agents
func (de *DecisionEngine) initializeCapabilities() {
	// Tool capabilities
	de.toolCapabilities["calculator"] = &ToolCapability{
		Name:          "calculator",
		Description:   "Performs mathematical calculations and computations",
		Strengths:     []string{"precision", "speed", "reliability"},
		Weaknesses:    []string{"limited to math", "no context understanding"},
		BestFor:       []string{"arithmetic", "equations", "computations", "mathematical problems"},
		AverageTokens: 100,
		Reliability:   0.98,
		Speed:         "fast",
		Complexity:    "simple",
	}

	de.toolCapabilities["document_summarizer"] = &ToolCapability{
		Name:          "document_summarizer",
		Description:   "Summarizes documents and text content",
		Strengths:     []string{"comprehensive", "context-aware", "flexible length"},
		Weaknesses:    []string{"requires good input", "LLM-dependent quality"},
		BestFor:       []string{"summaries", "key points", "content analysis", "long documents"},
		AverageTokens: 800,
		Reliability:   0.85,
		Speed:         "medium",
		Complexity:    "medium",
	}

	de.toolCapabilities["retriever"] = &ToolCapability{
		Name:          "retriever",
		Description:   "Searches and retrieves information from databases",
		Strengths:     []string{"fast retrieval", "large datasets", "semantic search"},
		Weaknesses:    []string{"depends on data quality", "may return irrelevant results"},
		BestFor:       []string{"information retrieval", "database queries", "knowledge search"},
		AverageTokens: 300,
		Reliability:   0.90,
		Speed:         "fast",
		Complexity:    "medium",
	}

	// Agent capabilities
	de.agentCapabilities["summary"] = &AgentCapability{
		Name:          "summary",
		Description:   "Specialized in document summarization and content analysis",
		Specialties:   []string{"content summarization", "key point extraction", "document analysis"},
		BestFor:       []string{"summarizing documents", "extracting insights", "content analysis"},
		AverageTokens: 1000,
		Reliability:   0.88,
		Speed:         "medium",
		Complexity:    "medium",
		Tools:         []string{},
	}

	de.agentCapabilities["researcher"] = &AgentCapability{
		Name:          "researcher",
		Description:   "Handles research tasks and information gathering",
		Specialties:   []string{"web research", "information synthesis", "comprehensive analysis"},
		BestFor:       []string{"research projects", "information gathering", "comprehensive studies"},
		AverageTokens: 2000,
		Reliability:   0.82,
		Speed:         "slow",
		Complexity:    "high",
		Tools:         []string{"retriever"},
	}

	de.agentCapabilities["analyst"] = &AgentCapability{
		Name:          "analyst",
		Description:   "Performs data analysis and insights generation",
		Specialties:   []string{"data analysis", "pattern recognition", "insights generation"},
		BestFor:       []string{"data analysis", "trend identification", "insight generation"},
		AverageTokens: 1500,
		Reliability:   0.85,
		Speed:         "medium",
		Complexity:    "high",
		Tools:         []string{"calculator", "retriever"},
	}
}

// Pattern matching functions
func (de *DecisionEngine) matchesMathPattern(query string) bool {
	patterns := []string{
		"calculate", "compute", "math", "arithmetic", "equation",
		"add", "subtract", "multiply", "divide", "solve",
		"what is", "how much", "equals",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}

	// Check for mathematical operators
	operators := []string{"+", "-", "*", "/", "=", "×", "÷"}
	for _, op := range operators {
		if strings.Contains(query, op) {
			return true
		}
	}

	return false
}

func (de *DecisionEngine) matchesSummarizationPattern(query string) bool {
	patterns := []string{
		"summarize", "summary", "summarise", "condense", "abstract",
		"extract key points", "key findings", "main points",
		"tl;dr", "brief", "overview",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	return false
}

func (de *DecisionEngine) matchesSearchPattern(query string) bool {
	patterns := []string{
		"search", "find", "look for", "retrieve", "get information",
		"research", "investigate", "explore", "discover",
	}

	for _, pattern := range patterns {
		if strings.Contains(query, pattern) {
			return true
		}
	}
	return false
}

// Utility functions
func (de *DecisionEngine) isToolAvailable(toolName string, availableTools []string) bool {
	for _, tool := range availableTools {
		if tool == toolName {
			return true
		}
	}
	return false
}

func (de *DecisionEngine) isAgentAvailable(agentName string, availableAgents []string) bool {
	for _, agent := range availableAgents {
		if agent == agentName {
			return true
		}
	}
	return false
}

// OptimizeTaskOrder optimizes the order of tasks for better execution
func (de *DecisionEngine) OptimizeTaskOrder(tasks []Task) []Task {
	if len(tasks) <= 1 {
		return tasks
	}

	// Create a copy to avoid modifying the original
	optimized := make([]Task, len(tasks))
	copy(optimized, tasks)

	// Sort by priority first (higher priority first)
	sort.Slice(optimized, func(i, j int) bool {
		return optimized[i].Priority > optimized[j].Priority
	})

	return optimized
}

// EstimateExecutionTime estimates the total execution time for a set of tasks
func (de *DecisionEngine) EstimateExecutionTime(tasks []Task, pattern ExecutionPattern) float64 {
	if len(tasks) == 0 {
		return 0
	}

	var totalTime float64

	switch pattern {
	case PatternSequential:
		for _, task := range tasks {
			totalTime += de.estimateTaskTime(task)
		}
	case PatternParallel:
		// Parallel execution time is the maximum of individual task times
		maxTime := 0.0
		for _, task := range tasks {
			taskTime := de.estimateTaskTime(task)
			if taskTime > maxTime {
				maxTime = taskTime
			}
		}
		totalTime = maxTime
	case PatternMapReduce:
		// Estimate map phase (parallel) + reduce phase (sequential)
		mapTime := 0.0
		reduceTime := 0.0

		// Find reduce tasks (those with dependencies)
		mapTasks := []Task{}
		reduceTasks := []Task{}

		for _, task := range tasks {
			if len(task.DependsOn) > 0 {
				reduceTasks = append(reduceTasks, task)
			} else {
				mapTasks = append(mapTasks, task)
			}
		}

		// Map phase time (parallel)
		for _, task := range mapTasks {
			taskTime := de.estimateTaskTime(task)
			if taskTime > mapTime {
				mapTime = taskTime
			}
		}

		// Reduce phase time (sequential)
		for _, task := range reduceTasks {
			reduceTime += de.estimateTaskTime(task)
		}

		totalTime = mapTime + reduceTime
	default:
		// Default to sequential
		for _, task := range tasks {
			totalTime += de.estimateTaskTime(task)
		}
	}

	return totalTime
}

// estimateTaskTime estimates execution time for a single task
func (de *DecisionEngine) estimateTaskTime(task Task) float64 {
	baseTime := 1.0 // Base time in seconds

	// Adjust based on task type
	switch task.Type {
	case TaskTypeTool:
		if cap, exists := de.toolCapabilities[task.Name]; exists {
			switch cap.Speed {
			case "fast":
				baseTime = 0.5
			case "medium":
				baseTime = 2.0
			case "slow":
				baseTime = 5.0
			}
		}
	case TaskTypeSubAgent:
		if cap, exists := de.agentCapabilities[task.Name]; exists {
			switch cap.Speed {
			case "fast":
				baseTime = 1.0
			case "medium":
				baseTime = 3.0
			case "slow":
				baseTime = 8.0
			}
		}
	case TaskTypeResponse:
		baseTime = 2.0 // LLM response time
	}

	// Adjust based on input complexity
	inputSize := len(fmt.Sprintf("%v", task.Input))
	if inputSize > 1000 {
		baseTime *= 2.0 // Double time for large inputs
	}

	return baseTime
}

// CalculateEfficiencyScore calculates an efficiency score for a decision
func (de *DecisionEngine) CalculateEfficiencyScore(decision *Decision, taskComplexity string) float64 {
	score := decision.Confidence

	// Adjust based on choice characteristics
	switch decision.Type {
	case "tool":
		if cap, exists := de.toolCapabilities[decision.Choice]; exists {
			score *= cap.Reliability

			// Adjust for task complexity match
			if de.matchesComplexity(taskComplexity, cap.Complexity) {
				score *= 1.2
			}
		}
	case "agent":
		if cap, exists := de.agentCapabilities[decision.Choice]; exists {
			score *= cap.Reliability

			// Adjust for task complexity match
			if de.matchesComplexity(taskComplexity, cap.Complexity) {
				score *= 1.2
			}
		}
	}

	// Cap the score at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return math.Round(score*100) / 100 // Round to 2 decimal places
}

// matchesComplexity checks if task complexity matches capability
func (de *DecisionEngine) matchesComplexity(taskComplexity, capabilityComplexity string) bool {
	// Simple matching logic
	switch taskComplexity {
	case "simple":
		return capabilityComplexity == "simple"
	case "medium":
		return capabilityComplexity == "simple" || capabilityComplexity == "medium"
	case "complex":
		return capabilityComplexity == "medium" || capabilityComplexity == "complex"
	default:
		return true // Unknown complexity, assume match
	}
}
