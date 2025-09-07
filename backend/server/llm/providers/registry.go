package providers

import (
	"fmt"
	"sync"

	"internal-perplexity/server/llm/providers/anthropic"
	"internal-perplexity/server/llm/providers/ollama"
	"internal-perplexity/server/llm/providers/openai"
	"internal-perplexity/server/llm/providers/shared"
)

// ProviderConfig holds configuration for creating providers
type ProviderConfig struct {
	Name    string // "openai" | "anthropic" | "ollama" | "openai-compatible"
	APIKey  string
	BaseURL string
	OrgID   string
	// Additional knobs
	ModelDefault string
	Extra        map[string]any
}

// Registry manages provider instances
type Registry struct {
	providers map[string]shared.LLMProvider
	mu        sync.RWMutex
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]shared.LLMProvider),
	}
}

// RegisterProvider registers a provider instance with a name
func (r *Registry) RegisterProvider(name string, provider shared.LLMProvider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[name] = provider
}

// GetProvider gets a registered provider by name
func (r *Registry) GetProvider(name string) (shared.LLMProvider, error) {
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

// NewProvider creates a new provider instance based on configuration
func NewProvider(cfg ProviderConfig) (shared.LLMProvider, error) {
	switch cfg.Name {
	case "openai", "openai-compatible":
		return NewOpenAIProvider(cfg)
	case "anthropic":
		return NewAnthropicProvider(cfg)
	case "ollama":
		return NewOllamaProvider(cfg)
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Name)
	}
}

// NewOpenAIProvider creates an OpenAI provider instance
func NewOpenAIProvider(cfg ProviderConfig) (shared.LLMProvider, error) {
	return openai.NewProvider(openai.Config{
		APIKey:  cfg.APIKey,
		BaseURL: defaultURL(cfg.BaseURL, "https://api.openai.com/v1"),
		OrgID:   cfg.OrgID,
	})
}

// NewAnthropicProvider creates an Anthropic provider instance
func NewAnthropicProvider(cfg ProviderConfig) (shared.LLMProvider, error) {
	return anthropic.NewProvider(anthropic.Config{
		APIKey:  cfg.APIKey,
		BaseURL: defaultURL(cfg.BaseURL, "https://api.anthropic.com"),
	})
}

// NewOllamaProvider creates an Ollama provider instance
func NewOllamaProvider(cfg ProviderConfig) (shared.LLMProvider, error) {
	return ollama.NewProvider(ollama.Config{
		BaseURL: defaultURL(cfg.BaseURL, "http://localhost:11434"),
	})
}

// defaultURL provides a default URL if none is specified
func defaultURL(baseURL, defaultURL string) string {
	if baseURL != "" {
		return baseURL
	}
	return defaultURL
}

// ProviderFactory provides factory methods for creating providers
type ProviderFactory struct {
	registry *Registry
}

// NewProviderFactory creates a new provider factory
func NewProviderFactory() *ProviderFactory {
	return &ProviderFactory{
		registry: NewRegistry(),
	}
}

// CreateProvider creates a provider from configuration and registers it
func (f *ProviderFactory) CreateProvider(name string, cfg ProviderConfig) (shared.LLMProvider, error) {
	provider, err := NewProvider(cfg)
	if err != nil {
		return nil, err
	}

	f.registry.RegisterProvider(name, provider)
	return provider, nil
}

// GetProvider gets a provider from the registry
func (f *ProviderFactory) GetProvider(name string) (shared.LLMProvider, error) {
	return f.registry.GetProvider(name)
}

// Global registry instance
var globalRegistry = NewRegistry()

// RegisterGlobalProvider registers a provider in the global registry
func RegisterGlobalProvider(name string, provider shared.LLMProvider) {
	globalRegistry.RegisterProvider(name, provider)
}

// GetGlobalProvider gets a provider from the global registry
func GetGlobalProvider(name string) (shared.LLMProvider, error) {
	return globalRegistry.GetProvider(name)
}

// CreateGlobalProvider creates and registers a provider in the global registry
func CreateGlobalProvider(name string, cfg ProviderConfig) (shared.LLMProvider, error) {
	provider, err := NewProvider(cfg)
	if err != nil {
		return nil, err
	}

	RegisterGlobalProvider(name, provider)
	return provider, nil
}
