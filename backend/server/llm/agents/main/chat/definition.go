package chat

import (
	"encoding/json"
	"fmt"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/api"
	"strings"
	"time"
)

type ChatAgent struct {
	name         string
	DefaultModel string
}

func NewChatAgent() *ChatAgent {
	return &ChatAgent{
		name:         "chat",
		DefaultModel: "gpt-4",
	}
}

func (c *ChatAgent) Name() string {
	return c.name
}

func (c *ChatAgent) Execute(input *agents.AgentInput, rt *agents.Runtime) (*agents.AgentResult, error) {
	start := time.Now()
	var request api.ChatCompletionRequest
	err := json.Unmarshal(input.Input, &request)
	if err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}
	fmt.Printf("request, %+v\n", request)

	// If the agent name contains a slash (e.g., "ollama/gpt-oss"), use only the prefix before the slash.
	baseName := request.Model
	if idx := strings.Index(baseName, "/"); idx != -1 {
		baseName = baseName[:idx]
	}
	llm, err := rt.LLM.GetProvider(strings.TrimSuffix(baseName, "/"))
	if err != nil {
		return nil, err
	}

	request.Model = strings.TrimPrefix(request.Model, llm.Name()+"/")
	if !llm.HasModel(request.Model) {
		return nil, fmt.Errorf("model %s not supported", request.Model)
	}
	response, err := llm.Complete(rt.Context, &request, input.User.APIKey)
	if err != nil {
		return nil, err
	}
	message := response.Choices[0].Message.ChatCompletionMessage.(api.ChatCompletionAssistantMessage).Content
	result := &agents.AgentResult{
		Content: message,
		Success: true,
		Stats: agents.AgentStats{
			FinishedAt:  time.Now(),
			StartedAt:   start,
			Duration:    time.Since(start),
			TokensIn:    response.Usage.PromptTokens,
			TokensOut:   response.Usage.CompletionTokens,
			CallsMade:   1,
			Parallelism: 1,
		},
	}
	return result, nil
}

func (c *ChatAgent) ValidateInput(input *agents.AgentInput) []agents.ValidationError {
	return nil
}
