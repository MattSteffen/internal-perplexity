package ollama

import (
	"context"
	"fmt"
	"strings"

	"internal-perplexity/server/llm/api"
)

// Config holds Ollama provider configuration
type Config struct {
	BaseURL string
}

// Provider implements the unified LLMProvider interface for Ollama
type Provider struct {
	config *Config
	models map[string]Model
}

// NewProvider creates a new Ollama provider
func NewProvider(cfg *Config) (*Provider, error) {
	if cfg == nil || cfg.BaseURL == "" {
		cfg = &Config{
			BaseURL: "http://localhost:11434",
		}
	}

	provider := &Provider{
		config: cfg,
		models: make(map[string]Model),
	}

	if _, err := provider.GetModels(); err != nil {
		return nil, err
	}

	fmt.Printf("provider.models, %+v\n", provider.models)
	return provider, nil
}

// Name returns the provider name
func (p *Provider) Name() string { return "ollama" }

func (p *Provider) HasModel(model string) bool {
	if m, ok := p.models[model]; ok {
		return m.IsSupported(model)
	}
	return false
}

func (p *Provider) GetModels() ([]Model, error) {
	models, err := GetModels(p.config.BaseURL)
	if err != nil {
		return nil, err
	}
	for _, model := range models {
		p.models[model.Name] = model
	}
	return models, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *api.ChatCompletionRequest, apiKey string) (*api.ChatCompletionResponse, error) {
	model, _ := strings.CutPrefix(req.Model, p.Name()+"/")
	req.Model = model
	if !p.HasModel(req.Model) {
		return nil, fmt.Errorf("model %s not supported", req.Model)
	}

	return api.SendRequest(ctx, req, p.config.BaseURL+"/v1/chat/completions", apiKey, 3)
}
