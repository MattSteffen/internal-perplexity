package llm

import (
	"fmt"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// Factory creates LLM instances based on configuration
type Factory struct{}

// NewFactory creates a new LLM factory
func NewFactory() *Factory {
	return &Factory{}
}

// Create creates an LLM instance based on the configuration
func (f *Factory) Create(cfg *config.LLMConfig) (interfaces.LLM, error) {
	switch cfg.Provider {
	case "ollama":
		return NewOllamaLLM(cfg)
	case "vllm":
		return NewVllmLLM(cfg)
	case "openai":
		return nil, fmt.Errorf("OpenAI LLM not implemented yet")
	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", cfg.Provider)
	}
}

// CreateWithConfig is a standalone function to create LLMs
func CreateWithConfig(cfg *config.LLMConfig) (interfaces.LLM, error) {
	factory := NewFactory()
	return factory.Create(cfg)
}
