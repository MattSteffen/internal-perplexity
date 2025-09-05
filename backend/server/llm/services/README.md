# Services Package

Orchestration frameworks and communication layers for agent systems.

## Directory Structure

### agent-manager/
**Contents:**
- Agent lifecycle management
- Task orchestration
- Resource allocation
- Performance monitoring
- Agent discovery and registration

**Key Features:**
- Agent pool management
- Load balancing
- Health monitoring
- Metrics collection
- Configuration management

## Core Services

### Agent Manager Service

```go
type AgentManager interface {
    RegisterAgent(ctx context.Context, agent *AgentConfig) error
    UnregisterAgent(ctx context.Context, agentID string) error
    GetAgent(ctx context.Context, agentID string) (*Agent, error)
    ListAgents(ctx context.Context, filter *AgentFilter) ([]*Agent, error)
    ExecuteTask(ctx context.Context, task *Task) (*TaskResult, error)
    GetMetrics(ctx context.Context, agentID string) (*AgentMetrics, error)
}

type Task struct {
    ID          string
    Type        TaskType
    Input       interface{}
    AgentID     string
    Priority    Priority
    Timeout     time.Duration
    Metadata    map[string]interface{}
}
```

### Orchestration Patterns

#### Sequential Pipeline
```go
func (om *Orchestrator) ExecutePipeline(ctx context.Context, agents []Agent, input interface{}) (interface{}, error) {
    result := input
    for _, agent := range agents {
        task := &Task{
            Input:   result,
            AgentID: agent.ID(),
        }
        output, err := om.ExecuteTask(ctx, task)
        if err != nil {
            return nil, err
        }
        result = output
    }
    return result, nil
}
```

#### MapReduce Pattern
```go
func (om *Orchestrator) ExecuteMapReduce(ctx context.Context, mappers []Agent, reducer Agent, input interface{}) (interface{}, error) {
    // Execute mappers in parallel
    mapperTasks := make([]*Task, len(mappers))
    for i, mapper := range mappers {
        mapperTasks[i] = &Task{
            Input:   input,
            AgentID: mapper.ID(),
        }
    }

    // Collect mapper results
    results, err := om.ExecuteParallel(ctx, mapperTasks)
    if err != nil {
        return nil, err
    }

    // Execute reducer
    reduceTask := &Task{
        Input:   results,
        AgentID: reducer.ID(),
    }
    return om.ExecuteTask(ctx, reduceTask)
}
```

## Monitoring and Metrics

### Metrics Collection
```go
type MetricsCollector interface {
    RecordTaskStart(taskID string, agentID string)
    RecordTaskComplete(taskID string, success bool, duration time.Duration)
    RecordTokenUsage(taskID string, usage *TokenUsage)
    RecordError(taskID string, err error)
    GetAgentMetrics(agentID string, timeRange *TimeRange) (*AgentMetrics, error)
}

type AgentMetrics struct {
    TasksCompleted   int64
    TasksFailed      int64
    AverageDuration  time.Duration
    TotalTokensUsed  int64
    SuccessRate      float64
    LastActive       time.Time
}
```

## Configuration Management

```go
type ServiceConfig struct {
    AgentManager *AgentManagerConfig `yaml:"agent_manager"`
    Monitoring   *MonitoringConfig   `yaml:"monitoring"`
    Caching      *CachingConfig      `yaml:"caching"`
}

type AgentManagerConfig struct {
    MaxConcurrentTasks int           `yaml:"max_concurrent_tasks"`
    TaskTimeout        time.Duration `yaml:"task_timeout"`
    RetryAttempts      int           `yaml:"retry_attempts"`
    HealthCheckInterval time.Duration `yaml:"health_check_interval"`
}
```

## Usage Examples

### Basic Orchestration
```go
// Initialize services
agentMgr := agentmanager.NewManager(config.AgentManager)

// Register agents
agentMgr.RegisterAgent(ctx, &AgentConfig{
    ID:   "summary",
    Type: AgentTypeSummary,
})

// Execute task
result, err := agentMgr.ExecuteTask(ctx, &Task{
    Type:        TaskTypeSummary,
    Input:       contentList,
    AgentID:     "summary",
    Timeout:     30 * time.Second,
})
```
