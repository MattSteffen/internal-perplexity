# LLM Services - MVP Tasks

## Overview
Build orchestration services for agent management, conversation handling, and task coordination. Focus on core services needed for MVP functionality.

## Agent Manager Service (`agent-manager/`)

### 1. Agent Lifecycle Management
- [ ] Create `AgentManager` interface and implementation
- [ ] Implement agent registration and discovery
- [ ] Add agent health monitoring
- [ ] Create agent configuration management

```go
type AgentManager interface {
    RegisterAgent(ctx context.Context, config *AgentConfig) error
    GetAgent(ctx context.Context, name string) (Agent, error)
    ListAgents(ctx context.Context) ([]*AgentInfo, error)
    ExecuteTask(ctx context.Context, task *Task) (*TaskResult, error)
}

type AgentManagerImpl struct {
    agents    map[string]Agent
    registry  *ToolRegistry
    llmClient LLMProvider
}

func (am *AgentManagerImpl) ExecuteTask(ctx context.Context, task *Task) (*TaskResult, error) {
    agent, exists := am.agents[task.AgentName]
    if !exists {
        return nil, fmt.Errorf("agent %s not found", task.AgentName)
    }

    input := &AgentInput{
        Data:       task.Input,
        Context:    task.Context,
        Parameters: task.Parameters,
    }

    result, err := agent.Execute(ctx, input)
    if err != nil {
        return nil, err
    }

    return &TaskResult{
        TaskID:     task.ID,
        Status:     StatusCompleted,
        Result:     result.Content,
        Stats:      result,
        ExecutedAt: time.Now(),
    }, nil
}
```

### 2. Task Orchestration
- [ ] Implement task queue and execution
- [ ] Add task status tracking
- [ ] Create task result caching
- [ ] Add task timeout handling

```go
type TaskOrchestrator struct {
    agentManager *AgentManagerImpl
    taskStore    TaskStore
}

func (to *TaskOrchestrator) ExecuteAsync(ctx context.Context, task *Task) (string, error) {
    taskID := generateTaskID()
    task.ID = taskID
    task.Status = StatusPending

    // Store task
    err := to.taskStore.Save(task)
    if err != nil {
        return "", err
    }

    // Execute asynchronously
    go func() {
        result, err := to.agentManager.ExecuteTask(context.Background(), task)
        if err != nil {
            task.Status = StatusFailed
            task.Error = err.Error()
        } else {
            task.Status = StatusCompleted
            task.Result = result
        }
        to.taskStore.Update(task)
    }()

    return taskID, nil
}
```

## Conversation Service (`conversations/`)

### 3. Conversation Management
- [ ] Create `ConversationService` interface
- [ ] Implement conversation creation and retrieval
- [ ] Add message history management
- [ ] Create conversation context handling

```go
type ConversationService interface {
    CreateConversation(ctx context.Context, config *ConversationConfig) (*Conversation, error)
    GetConversation(ctx context.Context, id string) (*Conversation, error)
    AddMessage(ctx context.Context, convID string, message *Message) error
    GetMessages(ctx context.Context, convID string, limit int) ([]*Message, error)
}

type ConversationServiceImpl struct {
    store ConversationStore
}

func (cs *ConversationServiceImpl) AddMessage(ctx context.Context, convID string, message *Message) error {
    conv, err := cs.store.Get(convID)
    if err != nil {
        return err
    }

    conv.Messages = append(conv.Messages, message)
    conv.UpdatedAt = time.Now()

    // Implement context window management
    if len(conv.Messages) > conv.Config.MaxMessages {
        conv.Messages = conv.Messages[len(conv.Messages)-conv.Config.MaxMessages:]
    }

    return cs.store.Save(conv)
}
```

## Testing Tasks

### 4. Unit Tests
- [ ] Test agent manager registration and discovery
- [ ] Test task execution and result handling
- [ ] Test conversation message management
- [ ] Test service error handling

### 5. Integration Tests
- [ ] Test agent manager with real agents
- [ ] Test task orchestration end-to-end
- [ ] Test conversation persistence
- [ ] Test concurrent task execution

```go
func TestAgentManagerIntegration(t *testing.T) {
    llmClient := NewMockLLMClient("http://localhost:11434/v1", "gpt-oss:20b")
    toolRegistry := NewToolRegistry()
    toolRegistry.Register(NewDocumentSummarizer(llmClient))

    agentManager := NewAgentManager(llmClient, toolRegistry)
    researcher := NewResearcherAgent(llmClient, toolRegistry)
    agentManager.RegisterAgent(context.Background(), &AgentConfig{
        Name:   "researcher",
        Agent:  researcher,
        Config: map[string]interface{}{},
    })

    task := &Task{
        AgentName: "researcher",
        Input: map[string]interface{}{
            "query": "test query",
        },
    }

    result, err := agentManager.ExecuteTask(context.Background(), task)
    assert.NoError(t, err)
    assert.Equal(t, StatusCompleted, result.Status)
}
```

## Service Infrastructure

### 6. Storage Abstractions
- [ ] Create `TaskStore` interface for task persistence
- [ ] Implement `ConversationStore` for conversation storage
- [ ] Add in-memory implementations for MVP
- [ ] Create storage configuration

### 7. Monitoring and Metrics
- [ ] Add service health checks
- [ ] Implement basic metrics collection
- [ ] Create service status endpoints
- [ ] Add performance monitoring

## Implementation Priority

### Phase 1: Core Services
1. [ ] Implement AgentManager interface
2. [ ] Create basic ConversationService
3. [ ] Add service configuration
4. [ ] Test service interactions

### Phase 2: Task Orchestration
1. [ ] Build task execution framework
2. [ ] Add async task processing
3. [ ] Implement task status tracking
4. [ ] Test concurrent execution

### Phase 3: Advanced Features
1. [ ] Add conversation context management
2. [ ] Implement service metrics
3. [ ] Add health monitoring
4. [ ] Comprehensive testing

## Configuration

### 8. Service Configuration
- [ ] Add service enable/disable settings
- [ ] Configure service-specific parameters
- [ ] Add service timeouts and limits
- [ ] Configure storage backends

```yaml
services:
  agent_manager:
    enabled: true
    max_concurrent_tasks: 10
    task_timeout: 300s
  conversations:
    enabled: true
    max_messages: 100
    context_window: 50
```

## Success Criteria
- [ ] Agent manager can register and execute agents
- [ ] Task orchestration works asynchronously
- [ ] Conversation service manages message history
- [ ] Services integrate properly with agents and tools
- [ ] Error handling is comprehensive
- [ ] Services work with localhost:11434 testing

## Files to Create
- `llm/services/agent-manager/manager.go`
- `llm/services/agent-manager/types.go`
- `llm/services/conversations/service.go`
- `llm/services/conversations/types.go`
- `llm/services/agent-manager/manager_test.go`
- `llm/services/conversations/service_test.go`
