package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers"
	"internal-perplexity/server/llm/tools"
	"time"

	"github.com/rs/zerolog"
)

type AgentRegistry struct {
	main    map[string]Agent
	support map[string]Agent
	tools   *tools.Registry
	llms    *providers.Registry
}

func NewAgentRegistry(toolRegistry *tools.Registry, llmRegistry *providers.Registry) *AgentRegistry {
	return &AgentRegistry{
		main:    make(map[string]Agent),
		support: make(map[string]Agent),
		tools:   toolRegistry,
		llms:    llmRegistry,
	}
}

func (r *AgentRegistry) Register(agent Agent) {
	r.main[agent.Name()] = agent
}
func (r *AgentRegistry) RegisterSupport(agent Agent) {
	r.support[agent.Name()] = agent
}
func (r *AgentRegistry) RegisterTool(tool tools.Tool) {
	r.tools.Register(tool)
}
func (r *AgentRegistry) RegisterLLM(llm providers.LLMProvider) {
	r.llms.RegisterProvider(llm.Name(), llm)
}

func (r *AgentRegistry) Get(name string) (Agent, error) {
	agent, exists := r.main[name]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", name)
	}
	return agent, nil
}

func (r *AgentRegistry) List() []Agent {
	agents := make([]Agent, 0, len(r.main))
	for _, agent := range r.main {
		agents = append(agents, agent)
	}
	for _, agent := range r.support {
		agents = append(agents, agent)
	}
	return agents
}

func (r *AgentRegistry) Execute(ctx context.Context, name string, input *AgentInput) (*AgentResult, error) {
	rt := &Runtime{
		Context: ctx,
		Agents:  r.main,
		Tools:   r.tools,
		LLM:     r.llms,
	}

	agent, err := r.Get(name)
	if err != nil {
		return nil, err
	}
	return agent.Execute(input, rt)
}

type Agent interface {
	Name() string
	ValidateInput(input *AgentInput) []ValidationError
	Execute(input *AgentInput, rt *Runtime) (*AgentResult, error)
}

type AgentInput struct {
	Input   json.RawMessage `json:"input"`
	User    User            `json:"user"`
	Data    map[string]any  `json:"data"`
	Meta    map[string]any  `json:"meta"`
	Session SessionInfo     `json:"session"`
}

type User struct {
	APIKey string `json:"api_key"`
}

type AgentResult struct {
	Content  any            `json:"content"`
	Success  bool           `json:"success"`
	Stats    AgentStats     `json:"stats"`
	Metadata map[string]any `json:"metadata"`
}

type AgentStats struct {
	StartedAt   time.Time
	FinishedAt  time.Time
	Duration    time.Duration
	TokensIn    int
	TokensOut   int
	CallsMade   int
	Parallelism int
}

type SessionInfo struct {
	SessionID string
	UserID    string
}

type Runtime struct {
	Context context.Context
	LLM     *providers.Registry
	Broker  Broker
	Agents  map[string]Agent
	Tools   *tools.Registry
	Logger  *zerolog.Logger
}

type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Code    string `json:"code"`
	Value   any    `json:"value,omitempty"`
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error: %s: %s", e.Field, e.Message)
}

type AgentCall struct {
	Name      string
	Arguments struct {
		Support []SupportCalls
		Tools   []api.ToolCall
	}
}

type SupportCalls struct {
	Calls       []SupportCall
	Description string
}

type SupportCall struct {
	ToolCall    api.ToolCall
	Description string
}

type Broker interface {
	// Stores v, returns opaque ref (UUID, key, …)
	Put(ctx context.Context, v any) (DataRef, error)
	Get(ctx context.Context, r DataRef) (any, error)
}

type DataRef string // of the form: `data:uuid` or `data:key`
