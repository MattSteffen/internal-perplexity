package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.NotNil(t, cfg)
	assert.Equal(t, "ollama", cfg.Embeddings.Provider)
	assert.Equal(t, "all-minilm:v2", cfg.Embeddings.Model)
	assert.Equal(t, "http://localhost:11434", cfg.Embeddings.BaseURL)
	assert.Equal(t, "ollama", cfg.LLM.Provider)
	assert.Equal(t, "llama3.2", cfg.LLM.ModelName)
	assert.Equal(t, "milvus", cfg.Database.Provider)
	assert.Equal(t, "documents", cfg.Database.Collection)
	assert.Equal(t, 10000, cfg.ChunkSize)
	assert.Equal(t, "tmp/", cfg.TempDir)
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		config  *CrawlerConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: &CrawlerConfig{
				Embeddings: EmbedderConfig{
					Provider: "ollama",
					Model:    "test-model",
					BaseURL:  "http://localhost:11434",
				},
				LLM: LLMConfig{
					ModelName:     "test-llm",
					Provider:      "ollama",
					BaseURL:       "http://localhost:11434",
					ContextLength: 1000,
					Timeout:       10.0,
				},
				Database: DatabaseConfig{
					Provider:   "milvus",
					Host:       "localhost",
					Port:       19530,
					Collection: "test-collection",
				},
				Converter: ConverterConfig{
					Type: "markitdown",
				},
				Extractor: ExtractorConfig{
					Type: "basic",
				},
				ChunkSize: 1000,
				MetadataSchema: map[string]interface{}{
					"type": "object",
				},
				NumBenchmarkQuestions: 3,
				MaxConcurrency:        4,
				BatchSize:             100,
			},
			wantErr: false,
		},
		{
			name: "missing embeddings provider",
			config: &CrawlerConfig{
				Embeddings: EmbedderConfig{
					Model:   "test-model",
					BaseURL: "http://localhost:11434",
				},
				LLM: LLMConfig{
					ModelName: "test-llm",
					Provider:  "ollama",
					BaseURL:   "http://localhost:11434",
				},
				Database: DatabaseConfig{
					Provider:   "milvus",
					Collection: "test-collection",
				},
				MetadataSchema: map[string]interface{}{
					"type": "object",
				},
			},
			wantErr: true,
		},
		{
			name: "missing metadata schema",
			config: &CrawlerConfig{
				Embeddings: EmbedderConfig{
					Provider: "ollama",
					Model:    "test-model",
					BaseURL:  "http://localhost:11434",
				},
				LLM: LLMConfig{
					ModelName: "test-llm",
					Provider:  "ollama",
					BaseURL:   "http://localhost:11434",
				},
				Database: DatabaseConfig{
					Provider:   "milvus",
					Collection: "test-collection",
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestEmbedderConfigCreation(t *testing.T) {
	cfg := EmbedderConfig{
		Provider: "ollama",
		Model:    "test-model",
		BaseURL:  "http://localhost:11434",
		APIKey:   "test-key",
	}

	assert.Equal(t, "ollama", cfg.Provider)
	assert.Equal(t, "test-model", cfg.Model)
	assert.Equal(t, "http://localhost:11434", cfg.BaseURL)
	assert.Equal(t, "test-key", cfg.APIKey)
}

func TestLLMConfigCreation(t *testing.T) {
	systemPrompt := "You are a helpful assistant"
	cfg := LLMConfig{
		ModelName:     "test-llm",
		Provider:      "ollama",
		BaseURL:       "http://localhost:11434",
		SystemPrompt:  &systemPrompt,
		ContextLength: 4096,
		Timeout:       60.0,
	}

	assert.Equal(t, "test-llm", cfg.ModelName)
	assert.Equal(t, "ollama", cfg.Provider)
	assert.Equal(t, "http://localhost:11434", cfg.BaseURL)
	assert.Equal(t, "You are a helpful assistant", *cfg.SystemPrompt)
	assert.Equal(t, 4096, cfg.ContextLength)
	assert.Equal(t, 60.0, cfg.Timeout)
}

func TestDatabaseConfigCreation(t *testing.T) {
	partition := "test-partition"
	cfg := DatabaseConfig{
		Provider:   "milvus",
		Host:       "localhost",
		Port:       19530,
		Username:   "test-user",
		Password:   "test-pass",
		Collection: "test-collection",
		Partition:  &partition,
		Recreate:   true,
	}

	assert.Equal(t, "milvus", cfg.Provider)
	assert.Equal(t, "localhost", cfg.Host)
	assert.Equal(t, 19530, cfg.Port)
	assert.Equal(t, "test-user", cfg.Username)
	assert.Equal(t, "test-pass", cfg.Password)
	assert.Equal(t, "test-collection", cfg.Collection)
	assert.Equal(t, "test-partition", *cfg.Partition)
	assert.True(t, cfg.Recreate)
}

func TestDatabaseConfigURI(t *testing.T) {
	cfg := DatabaseConfig{
		Host: "localhost",
		Port: 19530,
	}

	assert.Equal(t, "http://localhost:19530", cfg.GetURI())
}

func TestDatabaseConfigToken(t *testing.T) {
	cfg := DatabaseConfig{
		Username: "test-user",
		Password: "test-pass",
	}

	assert.Equal(t, "test-user:test-pass", cfg.GetToken())
}
