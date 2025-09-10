package embeddings

import (
	"fmt"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// Factory creates embedder instances based on configuration
type Factory struct{}

// NewFactory creates a new embedder factory
func NewFactory() *Factory {
	return &Factory{}
}

// Create creates an embedder instance based on the configuration
func (f *Factory) Create(cfg *config.EmbedderConfig) (interfaces.Embedder, error) {
	switch cfg.Provider {
	case "ollama":
		return NewOllamaEmbedder(cfg)
	case "openai":
		return NewOpenAIEmbedder(cfg)
	case "vllm":
		return nil, fmt.Errorf("VLLM embedder not implemented yet")
	default:
		return nil, fmt.Errorf("unsupported embedder provider: %s", cfg.Provider)
	}
}

// CreateWithConfig is a standalone function to create embedders
func CreateWithConfig(cfg *config.EmbedderConfig) (interfaces.Embedder, error) {
	factory := NewFactory()
	return factory.Create(cfg)
}
