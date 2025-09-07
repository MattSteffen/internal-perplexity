// Package agentmanager provides example usage of the AgentManager with the agent framework
package agentmanager

import (
	"context"
	"fmt"
	"log"
	"time"

	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
)

// MockAgent for demonstrating usage (uses the same implementation as in tests)
type ExampleMockAgent struct {
	name        string
	returnValue interface{}
}

func (m *ExampleMockAgent) Execute(ctx context.Context, input *agents.AgentInput, llmProvider shared.LLMProvider) (*agents.AgentResult, error) {
	return &agents.AgentResult{
		Success:    true,
		Content:    m.returnValue,
		Duration:   time.Millisecond * 100,
		TokensUsed: 50,
		Metadata: map[string]interface{}{
			"agent_name": m.name,
		},
	}, nil
}

func (m *ExampleMockAgent) GetCapabilities() []agents.Capability {
	return []agents.Capability{
		{Name: "mock_capability", Description: "Mock capability for demonstration"},
	}
}

func (m *ExampleMockAgent) GetStats() agents.AgentStats {
	return agents.AgentStats{
		TotalExecutions: 1,
		SuccessRate:     1.0,
	}
}

func (m *ExampleMockAgent) GetSystemPrompt() *agents.SystemPrompt {
	return &agents.SystemPrompt{
		BasePrompt: "Mock system prompt",
	}
}

// ExampleMockLLMProvider for demonstrating usage
type ExampleMockLLMProvider struct{}

func (m *ExampleMockLLMProvider) Complete(ctx context.Context, req *shared.CompletionRequest) (*shared.CompletionResponse, error) {
	return &shared.CompletionResponse{
		Content: "Mock response",
		Usage: shared.TokenUsage{
			TotalTokens: 25,
		},
	}, nil
}

func (m *ExampleMockLLMProvider) StreamComplete(ctx context.Context, req *shared.CompletionRequest) (<-chan *shared.StreamChunk, func(), error) {
	ch := make(chan *shared.StreamChunk, 1)
	ch <- &shared.StreamChunk{
		DeltaText: "Mock streaming response",
		Done:      true,
		Usage: &shared.TokenUsage{
			TotalTokens: 25,
		},
	}
	close(ch)
	return ch, func() {}, nil
}

func (m *ExampleMockLLMProvider) CountTokens(messages []shared.Message, model string) (int, error) {
	return 10, nil
}

func (m *ExampleMockLLMProvider) GetModelCapabilities(model string) shared.ModelCapabilities {
	return shared.ModelCapabilities{
		Streaming:         true,
		Tools:             false,
		ParallelToolCalls: false,
		JSONMode:          false,
		SystemMessage:     true,
		Vision:            false,
		MaxContextTokens:  128000,
	}
}

func (m *ExampleMockLLMProvider) GetSupportedModels() []shared.ModelInfo {
	return []shared.ModelInfo{}
}

func (m *ExampleMockLLMProvider) SupportsModel(model string) bool {
	return true
}

func (m *ExampleMockLLMProvider) Name() string {
	return "mock"
}

// ExampleUsage demonstrates how to use the updated AgentManager with the agent framework
func ExampleUsage() {
	// Create a new agent manager
	manager := NewAgentManager()

	// Create mock LLM provider
	llmProvider := &ExampleMockLLMProvider{}

	// Create and register mock agents
	summaryAgent := &ExampleMockAgent{
		name: "summary",
		returnValue: map[string]interface{}{
			"summary":         "This is a generated summary of the provided content.",
			"content_count":   2,
			"combined_length": 500,
		},
	}

	analystAgent := &ExampleMockAgent{
		name: "analyst",
		returnValue: map[string]interface{}{
			"analysis": map[string]interface{}{
				"insights": []string{
					"Key insight 1",
					"Key insight 2",
				},
				"confidence": 0.85,
			},
		},
	}

	// Register agents with the manager
	err := manager.RegisterAgent(context.Background(), "summary", summaryAgent)
	if err != nil {
		log.Fatalf("Failed to register summary agent: %v", err)
	}

	err = manager.RegisterAgent(context.Background(), "analyst", analystAgent)
	if err != nil {
		log.Fatalf("Failed to register analyst agent: %v", err)
	}

	// Example 1: Execute a task with query-based input (for intelligent agents like PrimaryAgent)
	fmt.Println("=== Example 1: Query-based task execution ===")
	task1 := NewTask("task-001", "summary", "Please summarize the following documents")
	task1.Input = map[string]interface{}{
		"contents": []interface{}{
			"This is the first document about artificial intelligence and machine learning.",
			"This is the second document about neural networks and deep learning.",
		},
		"instructions": "Focus on key concepts and technologies",
	}
	task1.Context = map[string]interface{}{
		"model":   "gpt-4",
		"timeout": 60,
	}

	result1, err := manager.ExecuteTask(context.Background(), task1, llmProvider)
	if err != nil {
		log.Printf("Task execution failed: %v", err)
	} else {
		fmt.Printf("Task Status: %s\n", result1.Status)
		fmt.Printf("Execution Duration: %v\n", result1.Duration)
		fmt.Printf("Tokens Used: %v\n", result1.TokensUsed)
		fmt.Printf("Result: %+v\n", result1.Result)
	}

	// Example 2: Execute a task with data-based input (for specialized agents)
	fmt.Println("\n=== Example 2: Data-based task execution ===")
	task2 := NewTask("task-002", "analyst", "") // No query for specialized agents
	task2.Input = map[string]interface{}{
		"data": map[string]interface{}{
			"dataset": []interface{}{
				map[string]interface{}{
					"feature1": 1.5,
					"feature2": "A",
					"target":   0.8,
				},
				map[string]interface{}{
					"feature1": 2.3,
					"feature2": "B",
					"target":   0.6,
				},
			},
		},
		"analysis_type": "statistical",
	}
	task2.Parameters = map[string]interface{}{
		"confidence_level": 0.95,
		"output_format":    "detailed",
	}

	result2, err := manager.ExecuteTask(context.Background(), task2, llmProvider)
	if err != nil {
		log.Printf("Task execution failed: %v", err)
	} else {
		fmt.Printf("Task Status: %s\n", result2.Status)
		fmt.Printf("Execution Duration: %v\n", result2.Duration)
		fmt.Printf("Result: %+v\n", result2.Result)
	}

	// Example 3: Task validation
	fmt.Println("\n=== Example 3: Task validation ===")
	validTask := NewTask("task-003", "summary", "Valid task")
	err = manager.ValidateTask(context.Background(), validTask)
	if err != nil {
		fmt.Printf("Validation failed: %v\n", err)
	} else {
		fmt.Println("Task validation passed")
	}

	// Example 4: Manager statistics
	fmt.Println("\n=== Example 4: Manager statistics ===")
	stats := manager.GetStats()
	fmt.Printf("Total Tasks: %d\n", stats.TotalTasks)
	fmt.Printf("Completed Tasks: %d\n", stats.CompletedTasks)
	fmt.Printf("Failed Tasks: %d\n", stats.FailedTasks)
	fmt.Printf("Average Duration: %v\n", stats.AverageDuration)
	fmt.Printf("Active Agents: %d\n", stats.ActiveAgents)

	// Example 5: Task history
	fmt.Println("\n=== Example 5: Task history ===")
	history, err := manager.GetTaskHistory("task-001")
	if err != nil {
		fmt.Printf("Failed to get task history: %v\n", err)
	} else {
		fmt.Printf("Task History - ID: %s, Status: %s, Duration: %v\n",
			history.TaskID, history.Status, history.Duration)
	}

	// Example 6: List all agents
	fmt.Println("\n=== Example 6: List agents ===")
	agentNames, err := manager.ListAgentNames(context.Background())
	if err != nil {
		fmt.Printf("Failed to list agents: %v\n", err)
	} else {
		fmt.Printf("Registered Agents: %v\n", agentNames)
	}

	// Example 7: Error handling - agent not found
	fmt.Println("\n=== Example 7: Error handling ===")
	task3 := NewTask("task-004", "non-existent-agent", "This will fail")
	result3, err := manager.ExecuteTask(context.Background(), task3, llmProvider)
	if err != nil {
		fmt.Printf("Execution error: %v\n", err)
	} else {
		fmt.Printf("Task failed as expected - Status: %s, Error: %s\n",
			result3.Status, result3.Error)
		fmt.Printf("Error Type: %v\n", result3.Metadata["error_type"])
	}
}

// DemonstrateIOHandling shows how the manager properly handles IO with the agent framework
func DemonstrateIOHandling() {
	fmt.Println("=== IO Handling Demonstration ===")

	manager := NewAgentManager()
	llmProvider := &ExampleMockLLMProvider{}

	// Register a mock agent that demonstrates proper IO handling
	mockAgent := &ExampleMockAgent{
		name: "io-demo",
		returnValue: map[string]interface{}{
			"processed_input":      "demonstrates how input is passed through",
			"validation_performed": true,
			"structured_output": map[string]interface{}{
				"section1": "First section of output",
				"section2": "Second section of output",
				"metadata": map[string]interface{}{
					"confidence":      0.92,
					"processing_time": "150ms",
				},
			},
		},
	}

	if err := manager.RegisterAgent(context.Background(), "io-demo", mockAgent); err != nil {
		log.Printf("Failed to register agent: %v", err)
		return
	}

	// Create a task that demonstrates comprehensive IO handling
	task := NewTask("io-demo-task", "io-demo", "Demonstrate comprehensive IO handling")
	task.Input = map[string]interface{}{
		"data": map[string]interface{}{
			"type": "structured_input",
			"content": []interface{}{
				"Input item 1",
				"Input item 2",
			},
		},
		"configuration": map[string]interface{}{
			"format":   "structured",
			"validate": true,
			"timeout":  30,
		},
	}
	task.Context = map[string]interface{}{
		"user_id":    "demo-user",
		"session_id": "demo-session",
		"model":      "gpt-4",
	}
	task.Parameters = map[string]interface{}{
		"output_format":        "comprehensive",
		"include_metadata":     true,
		"confidence_threshold": 0.8,
	}

	fmt.Println("Input Task Structure:")
	fmt.Printf("  ID: %s\n", task.ID)
	fmt.Printf("  Agent: %s\n", task.AgentName)
	fmt.Printf("  Query: %s\n", task.Query)
	fmt.Printf("  Input Keys: %v\n", getMapKeys(task.Input))
	fmt.Printf("  Context Keys: %v\n", getMapKeys(task.Context))
	fmt.Printf("  Parameters Keys: %v\n", getMapKeys(task.Parameters))

	// Execute the task
	result, err := manager.ExecuteTask(context.Background(), task, llmProvider)
	if err != nil {
		fmt.Printf("Execution failed: %v\n", err)
		return
	}

	fmt.Println("\nOutput Result Structure:")
	fmt.Printf("  Task ID: %s\n", result.TaskID)
	fmt.Printf("  Status: %s\n", result.Status)
	fmt.Printf("  Duration: %v\n", result.Duration)
	fmt.Printf("  Tokens Used: %v\n", result.TokensUsed)
	fmt.Printf("  Metadata Keys: %v\n", getMapKeys(result.Metadata))
	fmt.Printf("  Agent Stats - Total Executions: %d\n", result.AgentStats.TotalExecutions)

	fmt.Println("\nIO Flow Summary:")
	fmt.Println("1. Task input validated by manager")
	fmt.Println("2. Input converted to AgentInput format")
	fmt.Println("3. Agent validates input using its ValidateInput method")
	fmt.Println("4. Agent processes input and returns AgentResult")
	fmt.Println("5. Agent validates output using its ValidateOutput method")
	fmt.Println("6. Manager converts AgentResult to TaskResult")
	fmt.Println("7. Task and result metadata updated")
	fmt.Println("8. Execution history stored")
	fmt.Println("9. Manager statistics updated")

	// Show manager stats after execution
	finalStats := manager.GetStats()
	fmt.Printf("\nFinal Manager Stats: %+v\n", finalStats)
}

// Helper function to get keys from a map
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// Example showing how to use async execution
func ExampleAsyncExecution() {
	fmt.Println("=== Async Execution Example ===")

	manager := NewAgentManager()
	llmProvider := &ExampleMockLLMProvider{}

	mockAgent := &ExampleMockAgent{
		name: "async-demo",
		returnValue: map[string]interface{}{
			"async_result": "This task was executed asynchronously",
		},
	}

	if err := manager.RegisterAgent(context.Background(), "async-demo", mockAgent); err != nil {
		log.Printf("Failed to register agent: %v", err)
		return
	}

	task := NewTask("async-task", "async-demo", "Async execution demo")

	// Execute asynchronously
	err := manager.ExecuteTaskAsync(context.Background(), task, llmProvider)
	if err != nil {
		fmt.Printf("Failed to start async execution: %v\n", err)
		return
	}

	fmt.Println("Async task started successfully")
	fmt.Println("Note: In a real application, you would wait for completion or use channels")

	// Simulate waiting for completion (in real code, use proper synchronization)
	// time.Sleep(time.Second)

	// Check if task completed
	if history, err := manager.GetTaskHistory("async-task"); err == nil {
		fmt.Printf("Async task completed with status: %s\n", history.Status)
	}
}
