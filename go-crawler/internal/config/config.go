package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/spf13/viper"
)

// CrawlerConfig contains all configuration for the crawler system
type CrawlerConfig struct {
	// Core components
	Embeddings EmbedderConfig  `json:"embeddings" validate:"required"`
	LLM        LLMConfig       `json:"llm" validate:"required"`
	VisionLLM  *LLMConfig      `json:"vision_llm,omitempty"`
	Database   DatabaseConfig  `json:"database" validate:"required"`
	Converter  ConverterConfig `json:"converter" validate:"required"`
	Extractor  ExtractorConfig `json:"extractor" validate:"required"`

	// Processing settings
	ChunkSize      int                    `json:"chunk_size" validate:"min=100,max=50000"`
	TempDir        string                 `json:"temp_dir"`
	MetadataSchema map[string]interface{} `json:"metadata_schema" validate:"required"`

	// Benchmarking
	Benchmark                  bool `json:"benchmark"`
	GenerateBenchmarkQuestions bool `json:"generate_benchmark_questions"`
	NumBenchmarkQuestions      int  `json:"num_benchmark_questions" validate:"min=1,max=20"`

	// Logging
	LogLevel string `json:"log_level"`
	LogFile  string `json:"log_file"`

	// Processing options
	MaxConcurrency int `json:"max_concurrency" validate:"min=1,max=100"`
	BatchSize      int `json:"batch_size" validate:"min=1,max=1000"`
}

// EmbedderConfig configuration for embedding models
type EmbedderConfig struct {
	Provider  string `json:"provider" validate:"required,oneof=ollama openai vllm"`
	Model     string `json:"model" validate:"required"`
	BaseURL   string `json:"base_url" validate:"required,url"`
	APIKey    string `json:"api_key,omitempty"`
	Dimension *int   `json:"dimension,omitempty" validate:"omitempty,min=1,max=4096"`
}

// LLMConfig configuration for language models
type LLMConfig struct {
	ModelName     string  `json:"model_name" validate:"required"`
	Provider      string  `json:"provider" validate:"required,oneof=ollama openai vllm"`
	BaseURL       string  `json:"base_url" validate:"required,url"`
	APIKey        string  `json:"api_key,omitempty"`
	SystemPrompt  *string `json:"system_prompt,omitempty"`
	ContextLength int     `json:"ctx_length" validate:"min=1,max=100000"`
	Timeout       float64 `json:"default_timeout" validate:"min=1,max=3600"`
}

// DatabaseConfig configuration for vector database
type DatabaseConfig struct {
	Provider              string  `json:"provider" validate:"required,oneof=milvus"`
	Host                  string  `json:"host" validate:"required"`
	Port                  int     `json:"port" validate:"min=1,max=65535"`
	Username              string  `json:"username"`
	Password              string  `json:"password"`
	Collection            string  `json:"collection" validate:"required"`
	Partition             *string `json:"partition,omitempty"`
	Recreate              bool    `json:"recreate"`
	CollectionDescription *string `json:"collection_description,omitempty"`
}

// ConverterConfig configuration for document converters
type ConverterConfig struct {
	Type     string                 `json:"type" validate:"required,oneof=pymupdf markitdown docling"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ExtractorConfig configuration for metadata extractors
type ExtractorConfig struct {
	Type           string                 `json:"type" validate:"required,oneof=basic multi_schema"`
	LLM            *LLMConfig             `json:"llm,omitempty"`
	MetadataSchema map[string]interface{} `json:"metadata_schema,omitempty"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *CrawlerConfig {
	return &CrawlerConfig{
		Embeddings: EmbedderConfig{
			Provider: "ollama",
			Model:    "all-minilm:v2",
			BaseURL:  "http://localhost:11434",
			APIKey:   "",
		},
		LLM: LLMConfig{
			ModelName:     "llama3.2",
			Provider:      "ollama",
			BaseURL:       "http://localhost:11434",
			ContextLength: 32000,
			Timeout:       300.0,
		},
		Database: DatabaseConfig{
			Provider:   "milvus",
			Host:       "localhost",
			Port:       19530,
			Username:   "root",
			Password:   "Milvus",
			Collection: "documents",
			Recreate:   false,
		},
		Converter: ConverterConfig{
			Type:     "pymupdf",
			Metadata: map[string]interface{}{},
		},
		Extractor: ExtractorConfig{
			Type: "basic",
		},
		ChunkSize:                  10000,
		TempDir:                    "tmp/",
		MetadataSchema:             map[string]interface{}{},
		Benchmark:                  false,
		GenerateBenchmarkQuestions: false,
		NumBenchmarkQuestions:      3,
		LogLevel:                   "INFO",
		LogFile:                    "",
		MaxConcurrency:             4,
		BatchSize:                  100,
	}
}

// LoadConfigFromFile loads configuration from a JSON or YAML file
func LoadConfigFromFile(filepath string) (*CrawlerConfig, error) {
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file does not exist: %s", filepath)
	}

	viper.SetConfigFile(filepath)
	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	config := DefaultConfig()
	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %v", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	return config, nil
}

// LoadConfigFromMap loads configuration from a map
func LoadConfigFromMap(configMap map[string]interface{}) (*CrawlerConfig, error) {
	config := DefaultConfig()

	// Convert map to JSON then back to struct for validation
	jsonData, err := json.Marshal(configMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal config map: %v", err)
	}

	if err := json.Unmarshal(jsonData, config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %v", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	return config, nil
}

// Validate validates the configuration
func (c *CrawlerConfig) Validate() error {
	validate := validator.New()

	// Custom validation for file paths
	if c.TempDir != "" {
		if !filepath.IsAbs(c.TempDir) {
			c.TempDir = filepath.Clean(c.TempDir)
		}
		if err := os.MkdirAll(c.TempDir, 0755); err != nil {
			return fmt.Errorf("cannot create temp directory: %v", err)
		}
	}

	// Validate metadata schema
	if len(c.MetadataSchema) == 0 {
		return fmt.Errorf("metadata_schema cannot be empty")
	}

	// Validate schema has required fields
	if _, ok := c.MetadataSchema["type"]; !ok {
		return fmt.Errorf("metadata_schema must have 'type' field")
	}

	if c.MetadataSchema["type"] != "object" {
		return fmt.Errorf("metadata_schema type must be 'object'")
	}

	return validate.Struct(c)
}

// SaveToFile saves the configuration to a file
func (c *CrawlerConfig) SaveToFile(filepath string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	return nil
}

// String returns a string representation of the config (with sensitive data masked)
func (c *CrawlerConfig) String() string {
	configCopy := *c

	// Mask sensitive information
	if configCopy.Database.Password != "" {
		configCopy.Database.Password = strings.Repeat("*", len(configCopy.Database.Password))
	}
	if configCopy.Embeddings.APIKey != "" {
		configCopy.Embeddings.APIKey = strings.Repeat("*", len(configCopy.Embeddings.APIKey))
	}
	if configCopy.LLM.APIKey != "" {
		configCopy.LLM.APIKey = strings.Repeat("*", len(configCopy.LLM.APIKey))
	}

	data, _ := json.MarshalIndent(configCopy, "", "  ")
	return string(data)
}

// GetURI returns the database connection URI
func (d *DatabaseConfig) GetURI() string {
	return fmt.Sprintf("http://%s:%d", d.Host, d.Port)
}

// GetToken returns the database authentication token
func (d *DatabaseConfig) GetToken() string {
	return fmt.Sprintf("%s:%s", d.Username, d.Password)
}
