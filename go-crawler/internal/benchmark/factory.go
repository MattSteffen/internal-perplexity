package benchmark

import (
	"go-crawler/internal/config"
	"go-crawler/internal/embeddings"
	"go-crawler/internal/storage"
	"go-crawler/pkg/interfaces"
)

// Factory creates benchmark client instances
type Factory struct{}

// NewFactory creates a new benchmark factory
func NewFactory() *Factory {
	return &Factory{}
}

// Create creates a benchmark client instance
func (f *Factory) Create(dbConfig *config.DatabaseConfig, embedConfig *config.EmbedderConfig) (interfaces.BenchmarkClient, error) {
	// Create embedder
	embedderFactory := embeddings.NewFactory()
	embedder, err := embedderFactory.Create(embedConfig)
	if err != nil {
		return nil, err
	}

	// Create database client
	storageFactory := storage.NewFactory()
	database, err := storageFactory.Create(dbConfig, embedder.GetDimension(), nil)
	if err != nil {
		return nil, err
	}

	return NewBenchmarkClient(dbConfig, embedConfig, embedder, database), nil
}

// CreateWithConfig creates a benchmark client with the given configurations
func CreateWithConfig(dbConfig *config.DatabaseConfig, embedConfig *config.EmbedderConfig) (interfaces.BenchmarkClient, error) {
	factory := NewFactory()
	return factory.Create(dbConfig, embedConfig)
}
