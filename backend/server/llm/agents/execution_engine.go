package agents

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
)

// ExecutionEngine handles the execution of task plans
type ExecutionEngine struct {
	toolRegistry *tools.Registry
}

// NewExecutionEngine creates a new execution engine
func NewExecutionEngine(toolRegistry *tools.Registry) *ExecutionEngine {
	return &ExecutionEngine{
		toolRegistry: toolRegistry,
	}
}

// ExecutePlan executes an execution plan and returns the results
func (e *ExecutionEngine) ExecutePlan(ctx context.Context, plan *ExecutionPlan, llmProvider shared.LLMProvider) (*AgentResult, error) {
	start := time.Now()
	var executionLog []ExecutionStep

	// Log plan execution start
	executionLog = append(executionLog, ExecutionStep{
		Type:        "plan_execution_start",
		Description: fmt.Sprintf("Starting execution of plan with pattern: %s", plan.Pattern),
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"pattern":          plan.Pattern,
			"task_count":       len(plan.Tasks),
			"estimated_tokens": plan.EstimatedTokens,
		},
	})

	var result *AgentResult
	var err error

	switch plan.Pattern {
	case PatternSequential:
		result, err = e.executeSequential(ctx, plan, llmProvider, &executionLog)
	case PatternParallel:
		result, err = e.executeParallel(ctx, plan, llmProvider, &executionLog)
	case PatternMapReduce:
		result, err = e.executeMapReduce(ctx, plan, llmProvider, &executionLog)
	case PatternDirect:
		result, err = e.executeDirect(ctx, plan, &executionLog)
	default:
		return &AgentResult{
			Success:  false,
			Duration: time.Since(start),
			Metadata: map[string]any{
				"error": fmt.Sprintf("unsupported execution pattern: %s", plan.Pattern),
			},
			ExecutionLog: executionLog,
		}, fmt.Errorf("unsupported execution pattern: %s", plan.Pattern)
	}

	if err != nil {
		executionLog = append(executionLog, ExecutionStep{
			Type:        "plan_execution_error",
			Description: fmt.Sprintf("Plan execution failed: %v", err),
			Timestamp:   time.Now(),
			Data: map[string]interface{}{
				"error": err.Error(),
			},
		})
		return &AgentResult{
			Success:      false,
			Duration:     time.Since(start),
			Metadata:     map[string]any{"error": err.Error()},
			ExecutionLog: executionLog,
		}, err
	}

	// Update result with execution log
	result.Duration = time.Since(start)
	result.ExecutionLog = executionLog

	return result, nil
}

// executeSequential executes tasks in sequence
func (e *ExecutionEngine) executeSequential(ctx context.Context, plan *ExecutionPlan, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (*AgentResult, error) {
	taskResults := make(map[string]interface{})

	// Sort tasks by priority and dependencies
	sortedTasks := e.sortTasksByPriority(plan.Tasks)

	for _, task := range sortedTasks {
		// Check if dependencies are satisfied
		if !e.areDependenciesSatisfied(task, taskResults) {
			*executionLog = append(*executionLog, ExecutionStep{
				Type:        "task_skipped",
				Description: fmt.Sprintf("Skipping task %s due to unsatisfied dependencies", task.ID),
				Timestamp:   time.Now(),
				Data: map[string]interface{}{
					"task_id":    task.ID,
					"depends_on": task.DependsOn,
				},
			})
			continue
		}

		taskStart := time.Now()
		*executionLog = append(*executionLog, ExecutionStep{
			Type:        "task_execution_start",
			Description: fmt.Sprintf("Executing task: %s", task.Description),
			Timestamp:   taskStart,
			Data: map[string]interface{}{
				"task_id":   task.ID,
				"task_type": task.Type,
				"task_name": task.Name,
			},
		})

		result, err := e.executeTask(ctx, task, llmProvider, executionLog)
		if err != nil {
			return nil, fmt.Errorf("task %s failed: %w", task.ID, err)
		}

		taskResults[task.ID] = result

		*executionLog = append(*executionLog, ExecutionStep{
			Type:        "task_execution_complete",
			Description: fmt.Sprintf("Completed task: %s", task.Description),
			Timestamp:   time.Now(),
			Duration:    time.Since(taskStart),
			Data: map[string]interface{}{
				"task_id":   task.ID,
				"task_type": task.Type,
				"task_name": task.Name,
			},
		})
	}

	return &AgentResult{
		Content: map[string]interface{}{
			"task_results": taskResults,
			"pattern":      PatternSequential,
		},
		Success: true,
		Metadata: map[string]any{
			"execution_pattern": PatternSequential,
			"tasks_executed":    len(taskResults),
		},
	}, nil
}

// executeParallel executes tasks in parallel
func (e *ExecutionEngine) executeParallel(ctx context.Context, plan *ExecutionPlan, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (*AgentResult, error) {
	taskResults := make(map[string]interface{})
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(plan.Tasks))

	// Execute all tasks in parallel
	for _, task := range plan.Tasks {
		wg.Add(1)
		go func(t Task) {
			defer wg.Done()

			taskStart := time.Now()
			mu.Lock()
			*executionLog = append(*executionLog, ExecutionStep{
				Type:        "task_execution_start",
				Description: fmt.Sprintf("Executing task: %s", t.Description),
				Timestamp:   taskStart,
				Data: map[string]interface{}{
					"task_id":   t.ID,
					"task_type": t.Type,
					"task_name": t.Name,
				},
			})
			mu.Unlock()

			result, err := e.executeTask(ctx, t, llmProvider, executionLog)
			if err != nil {
				errChan <- fmt.Errorf("task %s failed: %w", t.ID, err)
				return
			}

			mu.Lock()
			taskResults[t.ID] = result
			*executionLog = append(*executionLog, ExecutionStep{
				Type:        "task_execution_complete",
				Description: fmt.Sprintf("Completed task: %s", t.Description),
				Timestamp:   time.Now(),
				Duration:    time.Since(taskStart),
				Data: map[string]interface{}{
					"task_id":   t.ID,
					"task_type": t.Type,
					"task_name": t.Name,
				},
			})
			mu.Unlock()
		}(task)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	if err := <-errChan; err != nil {
		return nil, err
	}

	return &AgentResult{
		Content: map[string]interface{}{
			"task_results": taskResults,
			"pattern":      PatternParallel,
		},
		Success: true,
		Metadata: map[string]any{
			"execution_pattern": PatternParallel,
			"tasks_executed":    len(taskResults),
		},
	}, nil
}

// executeMapReduce executes tasks using map-reduce pattern
func (e *ExecutionEngine) executeMapReduce(ctx context.Context, plan *ExecutionPlan, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (*AgentResult, error) {
	// Separate map and reduce tasks
	mapTasks := []Task{}
	var reduceTask *Task

	for _, task := range plan.Tasks {
		if task.Type == TaskTypeResponse && len(task.DependsOn) > 0 {
			reduceTask = &task
		} else {
			mapTasks = append(mapTasks, task)
		}
	}

	// Execute map tasks in parallel
	mapResults := make(map[string]interface{})
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(mapTasks))

	for _, task := range mapTasks {
		wg.Add(1)
		go func(t Task) {
			defer wg.Done()

			result, err := e.executeTask(ctx, t, llmProvider, executionLog)
			if err != nil {
				errChan <- fmt.Errorf("map task %s failed: %w", t.ID, err)
				return
			}

			mu.Lock()
			mapResults[t.ID] = result
			mu.Unlock()
		}(task)
	}

	wg.Wait()
	close(errChan)

	// Check for map errors
	if err := <-errChan; err != nil {
		return nil, err
	}

	// Execute reduce task with map results
	if reduceTask != nil {
		reduceTask.Input["map_results"] = mapResults
		reduceResult, err := e.executeTask(ctx, *reduceTask, llmProvider, executionLog)
		if err != nil {
			return nil, fmt.Errorf("reduce task failed: %w", err)
		}

		return &AgentResult{
			Content: map[string]interface{}{
				"map_results":   mapResults,
				"reduce_result": reduceResult,
				"pattern":       PatternMapReduce,
			},
			Success: true,
			Metadata: map[string]any{
				"execution_pattern": PatternMapReduce,
				"map_tasks":         len(mapTasks),
				"reduce_task":       reduceTask.ID,
			},
		}, nil
	}

	return &AgentResult{
		Content: map[string]interface{}{
			"map_results": mapResults,
			"pattern":     PatternMapReduce,
		},
		Success: true,
		Metadata: map[string]any{
			"execution_pattern": PatternMapReduce,
			"map_tasks":         len(mapTasks),
		},
	}, nil
}

// executeDirect returns a direct response without executing tasks
func (e *ExecutionEngine) executeDirect(ctx context.Context, plan *ExecutionPlan, executionLog *[]ExecutionStep) (*AgentResult, error) {
	*executionLog = append(*executionLog, ExecutionStep{
		Type:        "direct_response",
		Description: "Providing direct response without task execution",
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"response": plan.FinalResponse,
		},
	})

	return &AgentResult{
		Content: map[string]interface{}{
			"response": plan.FinalResponse,
			"pattern":  PatternDirect,
		},
		Success: true,
		Metadata: map[string]any{
			"execution_pattern": PatternDirect,
			"reasoning":         plan.Reasoning,
		},
	}, nil
}

// executeTask executes a single task
func (e *ExecutionEngine) executeTask(ctx context.Context, task Task, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (interface{}, error) {
	switch task.Type {
	case TaskTypeTool:
		return e.executeToolTask(ctx, task, llmProvider, executionLog)
	case TaskTypeSubAgent:
		return e.executeSubAgentTask(ctx, task, llmProvider, executionLog)
	case TaskTypeResponse:
		return e.executeResponseTask(ctx, task, llmProvider, executionLog)
	default:
		return nil, fmt.Errorf("unsupported task type: %s", task.Type)
	}
}

// executeToolTask executes a tool task
func (e *ExecutionEngine) executeToolTask(ctx context.Context, task Task, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (interface{}, error) {
	toolInput := &tools.ToolInput{
		Name: task.Name,
		Data: task.Input,
	}

	result, err := e.toolRegistry.Execute(ctx, toolInput, llmProvider)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"tool_name": task.Name,
		"result":    result.Data,
		"success":   result.Success,
	}, nil
}

// executeSubAgentTask executes a sub-agent task
func (e *ExecutionEngine) executeSubAgentTask(ctx context.Context, task Task, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (interface{}, error) {
	// TODO: Implement sub-agent execution when agent manager is integrated
	return nil, fmt.Errorf("sub-agent execution not yet implemented: %s", task.Name)
}

// executeResponseTask executes a response task (direct LLM call)
func (e *ExecutionEngine) executeResponseTask(ctx context.Context, task Task, llmProvider shared.LLMProvider, executionLog *[]ExecutionStep) (interface{}, error) {
	messages := []shared.Message{
		{
			Role:    "system",
			Content: "You are a helpful AI assistant. Provide clear, accurate, and concise responses.",
		},
		{
			Role:    "user",
			Content: task.Description,
		},
	}

	req := &shared.CompletionRequest{
		Messages: messages,
		Options: shared.CompletionOptions{
			MaxTokens:   1000,
			Temperature: 0.7,
		},
		Model: "gpt-4",
	}

	resp, err := llmProvider.Complete(ctx, req)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"response": resp.Content,
		"tokens":   resp.Usage.TotalTokens,
		"model":    req.Model,
	}, nil
}

// sortTasksByPriority sorts tasks by priority (lower number = higher priority)
func (e *ExecutionEngine) sortTasksByPriority(tasks []Task) []Task {
	sorted := make([]Task, len(tasks))
	copy(sorted, tasks)

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Priority < sorted[j].Priority
	})

	return sorted
}

// areDependenciesSatisfied checks if all dependencies of a task are satisfied
func (e *ExecutionEngine) areDependenciesSatisfied(task Task, completedTasks map[string]interface{}) bool {
	for _, dep := range task.DependsOn {
		if _, exists := completedTasks[dep]; !exists {
			return false
		}
	}
	return true
}
