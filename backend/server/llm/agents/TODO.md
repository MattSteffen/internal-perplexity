# LLM Agents - MVP Tasks

## Overview
Implement agent framework with specialization patterns. Focus on researcher sub-agent and primary main-agent for MVP, integrating with tools and LLM providers.

## Agent Framework

### 1. Agent Interfaces
- [ ] Define `Agent` interface with Execute method
- [ ] Create `AgentInput` and `AgentResult` types
- [ ] Add agent metadata (capabilities, stats)
- [ ] Implement agent configuration structures

```go
type Agent interface {
    Execute(ctx context.Context, input *AgentInput) (*AgentResult, error)
    GetCapabilities() []Capability
    GetStats() AgentStats
}

type AgentResult struct {
    Content     interface{}
    Success     bool
    TokensUsed  TokenUsage
    Duration    time.Duration
    Metadata    map[string]interface{}
}
```

### 2. Summary Sub-Agent (`sub-agents/summary/`)
- [ ] Create `summary.go` with SummaryAgent struct
- [ ] Implement content list processing
- [ ] Add instruction parsing and focus area handling
- [ ] Return structured summary results with stats

```go
type SummaryAgent struct {
    llmClient LLMProvider
}

### 2. Input Validation Framework
- [ ] Create `ValidationError` and `ValidationErrors` types
- [ ] Implement `InputValidator` interface for agents
- [ ] Add validation error codes and messages
- [ ] Create validation helper functions

### 4. Summary Sub-Agent (`sub-agents/summary/`)
- [ ] Create `summary.go` with SummaryAgent struct
- [ ] Implement comprehensive input validation
- [ ] Add content list processing with size limits
- [ ] Add instruction parsing and focus area validation
- [ ] Return structured summary results with stats

### 5. Primary Agent Validation
- [ ] Implement primary agent input validation
- [ ] Add task type validation
- [ ] Add timeout validation
- [ ] Test validation error handling
        Metadata: map[string]interface{}{
            "input_length": len(combinedContent),
            "focus_areas": focusAreas,
        },
    }, nil
}
```

### 3. Primary Main-Agent (`main-agents/primary/`)
- [ ] Create `primary.go` with PrimaryAgent struct
- [ ] Implement agent orchestration logic
- [ ] Integrate with summary sub-agent
- [ ] Handle document summarization workflows

```go
type PrimaryAgent struct {
    llmClient     LLMProvider
    agentManager  *AgentManager
    summaryAgent  *SummaryAgent
}

func (p *PrimaryAgent) Execute(ctx context.Context, input *AgentInput) (*AgentResult, error) {
    task := input.Data["task"].(string)

    switch task {
    case "summarize_documents":
        return p.executeSummary(ctx, input)
    default:
        return p.executeGeneral(ctx, input)
    }
}
```

## Testing Tasks

### 4. Unit Tests
- [ ] Test agent interface implementations
- [ ] Test agent input/output validation
- [ ] Test agent capability detection
- [ ] Test agent statistics tracking

### 5. Validation Tests
- [ ] Test summary agent input validation
- [ ] Test primary agent input validation
- [ ] Test validation error messages and codes
- [ ] Test edge cases and boundary conditions

### 6. Integration Tests
- [ ] Test summary agent with localhost:11434
- [ ] Test primary agent orchestration
- [ ] Test agent result formatting

```go
func TestSummaryAgentIntegration(t *testing.T) {
    llmClient := NewOllamaClient(&LLMConfig{
        BaseURL: "http://localhost:11434/v1",
        Model:   "gpt-oss:20b",
    })

    summary := NewSummaryAgent(llmClient)

    input := &AgentInput{
        Data: map[string]interface{}{
            "contents": []string{
                "First document content about AI systems...",
                "Second document content about agent architectures...",
            },
            "instructions": "Focus on key architectural patterns",
            "focus_areas": []string{"patterns", "benefits"},
        },
    }

    result, err := summary.Execute(context.Background(), input)
    assert.NoError(t, err)
    assert.True(t, result.Success)
    assert.IsType(t, &SummaryResult{}, result.Content)
}
```

### 6. Agent Orchestration
- [ ] Implement sequential pipeline execution
- [ ] Add parallel agent execution support
- [ ] Create agent chaining framework
- [ ] Add agent result aggregation

## Agent Types

### 7. Future Sub-Agents
- [ ] Additional specialized agents can be added following the summary agent pattern
- [ ] Each agent should focus on a specific capability
- [ ] Maintain single LLM call per agent execution

## Implementation Priority

### Phase 1: Core Framework
1. [ ] Implement Agent interface and types
2. [ ] Create basic agent result structures
3. [ ] Add agent capability system
4. [ ] Test framework functionality

### Phase 2: Summary Agent
1. [ ] Build summary agent structure
2. [ ] Implement content list processing
3. [ ] Add instruction and focus area handling
4. [ ] Test with localhost:11434

### Phase 3: Primary Agent
1. [ ] Implement primary agent orchestration
2. [ ] Integrate with summary agent
3. [ ] Test document summarization workflows

## Configuration

### 8. Agent Configuration
- [ ] Add agent enable/disable settings
- [ ] Configure agent-specific parameters
- [ ] Add agent execution timeouts
- [ ] Configure LLM models per agent

```yaml
agents:
  summary:
    enabled: true
    max_content_length: 10000
    default_focus_areas: ["key_points", "conclusions"]
    model: "gpt-oss:20b"
  primary:
    enabled: true
    max_sub_agents: 3
    execution_timeout: 300s
```

## Success Criteria
- [ ] Agents execute with proper LLM integration
- [ ] Summary agent returns structured results from content lists
- [ ] Primary agent orchestrates summary tasks correctly
- [ ] Agent stats are tracked and returned
- [ ] Agents work with localhost:11434 (gpt-oss:20b)
- [ ] Error handling is robust across agents

## Files to Create
- `llm/agents/types.go`
- `llm/agents/sub-agents/summary/summary.go`
- `llm/agents/main-agents/primary/primary.go`
- `llm/agents/sub-agents/summary/summary_test.go`
- `llm/agents/main-agents/primary/primary_test.go`
