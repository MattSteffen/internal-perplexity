package providers

import (
	"context"
	"fmt"
	"log"
	"sync"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers/ollama"
)

// ProviderConfig holds configuration for creating providers
type ProviderConfig struct {
	Name           string // "openai" | "anthropic" | "ollama" | "openai-compatible"
	DefaultAPIKey  string
	DefaultBaseURL string
	DefaultModel   string
	Extra          map[string]any
}

type Model struct {
	Name  string
	Model string
}

type LLMProvider interface {
	Complete(ctx context.Context, req *api.ChatCompletionRequest, apiKey string) (*api.ChatCompletionResponse, error)
	HasModel(model string) bool
	// StreamComplete(ctx context.Context, req *api.ChatCompletionRequest) (<-chan *api.ChatCompletionStreamResponse, func(), error)
	Name() string
	GetModels() ([]Model, error)
}

// Registry manages provider instances
type Registry struct {
	providers map[string]LLMProvider
	mu        sync.RWMutex
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	providers := make(map[string]LLMProvider)
	// providers["openai"] = openai.NewProvider()
	// providers["anthropic"] = anthropic.NewProvider()
	ollamaModel, err := ollama.NewProvider(&ollama.Config{
		BaseURL: "http://localhost:11434",
	})
	if err != nil {
		log.Fatalf("Failed to create Ollama provider: %v", err)
	}

	providers["ollama"] = ollamaModel
	fmt.Printf("providers, %+v\n", providers)
	return &Registry{
		providers: providers,
	}
}

// RegisterProvider registers a provider instance with a name
func (r *Registry) RegisterProvider(name string, provider LLMProvider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[name] = provider
}

// GetProvider gets a registered provider by name
func (r *Registry) GetProvider(name string) (LLMProvider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	provider, exists := r.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider not found: %s", name)
	}
	return provider, nil
}

// ListProviders returns a list of registered provider names
func (r *Registry) ListProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}
