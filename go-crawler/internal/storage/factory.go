package storage

import (
	"fmt"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// Factory creates database client instances based on configuration
type Factory struct{}

// NewFactory creates a new storage factory
func NewFactory() *Factory {
	return &Factory{}
}

// Create creates a database client instance based on the configuration
func (f *Factory) Create(cfg *config.DatabaseConfig, embeddingDim int, metadataSchema map[string]interface{}) (interfaces.DatabaseClient, error) {
	switch cfg.Provider {
	case "milvus":
		return NewMilvusClient(cfg, embeddingDim, metadataSchema)
	default:
		return nil, fmt.Errorf("unsupported database provider: %s", cfg.Provider)
	}
}

// CreateWithConfig is a standalone function to create database clients
func CreateWithConfig(cfg *config.DatabaseConfig, embeddingDim int, metadataSchema map[string]interface{}) (interfaces.DatabaseClient, error) {
	factory := NewFactory()
	return factory.Create(cfg, embeddingDim, metadataSchema)
}
