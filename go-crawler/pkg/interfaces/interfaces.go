package interfaces

import (
	"context"
	"time"

	"go-crawler/internal/config"
)

// Document represents a document with text, embeddings, and metadata
type Document struct {
	Text          string                 `json:"text"`
	TextEmbedding []float64              `json:"text_embedding"`
	ChunkIndex    int                    `json:"chunk_index"`
	Source        string                 `json:"source"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// Converter converts various document formats to markdown
type Converter interface {
	Convert(ctx context.Context, filepath string) (string, error)
}

// Extractor extracts metadata from text using LLMs and schemas
type Extractor interface {
	ExtractMetadata(ctx context.Context, text string) (map[string]interface{}, error)
	ChunkText(text string, chunkSize int) ([]string, error)
}

// Embedder generates vector embeddings for text
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
	GetDimension() int
}

// LLM provides language model capabilities
type LLM interface {
	Invoke(ctx context.Context, prompt string, options *LLMOptions) (string, error)
	InvokeWithMessages(ctx context.Context, messages []LLMMessage, options *LLMOptions) (string, error)
	GetContextLength() int
}

// LLMMessage represents a message in a conversation
type LLMMessage struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// LLMOptions contains options for LLM invocations
type LLMOptions struct {
	Temperature    *float64    `json:"temperature,omitempty"`
	MaxTokens      *int        `json:"max_tokens,omitempty"`
	TopP           *float64    `json:"top_p,omitempty"`
	TopK           *int        `json:"top_k,omitempty"`
	ResponseFormat interface{} `json:"response_format,omitempty"`
	StopSequences  []string    `json:"stop_sequences,omitempty"`
}

// DatabaseClient provides vector database operations
type DatabaseClient interface {
	CreateCollection(ctx context.Context, recreate bool) error
	InsertDocuments(ctx context.Context, docs []Document) error
	CheckDuplicate(ctx context.Context, source string, chunkIndex int) (bool, error)
	Search(ctx context.Context, query string, limit int, filters map[string]interface{}) ([]SearchResult, error)
	Close() error
}

// SearchResult represents a search result from the database
type SearchResult struct {
	Document Document `json:"document"`
	Score    float64  `json:"score"`
}

// ImageDescriber describes images within documents
type ImageDescriber interface {
	DescribeImage(ctx context.Context, imageData []byte, imageExt string, prompt string) (string, error)
}

// BenchmarkClient provides benchmarking capabilities
type BenchmarkClient interface {
	RunBenchmark(ctx context.Context, generateQueries bool) (*BenchmarkResults, error)
	Search(ctx context.Context, queries []string) ([]BenchmarkResult, error)
	SaveResults(results *BenchmarkResults, filepath string) error
}

// BenchmarkResult represents a single query result in benchmarking
type BenchmarkResult struct {
	Query          string        `json:"query"`
	ExpectedSource string        `json:"expected_source"`
	PlacementOrder *int          `json:"placement_order,omitempty"`
	Distance       *float64      `json:"distance,omitempty"`
	TimeToSearch   time.Duration `json:"time_to_search"`
	Found          bool          `json:"found"`
}

// BenchmarkResults contains aggregated benchmark results
type BenchmarkResults struct {
	ResultsByDoc           map[string][]BenchmarkResult `json:"results_by_doc"`
	PlacementDistribution  map[int]int                  `json:"placement_distribution"`
	DistanceDistribution   []float64                    `json:"distance_distribution"`
	PercentInTopK          map[int]float64              `json:"percent_in_top_k"`
	SearchTimeDistribution []time.Duration              `json:"search_time_distribution"`
}

// Crawler is the main orchestrator interface
type Crawler interface {
	Crawl(ctx context.Context, path string) error
	Benchmark(ctx context.Context) (*BenchmarkResults, error)
	GetStats() CrawlerStats
}

// CrawlerStats contains statistics about crawler operations
type CrawlerStats struct {
	TotalDocuments     int           `json:"total_documents"`
	ProcessedDocuments int           `json:"processed_documents"`
	FailedDocuments    int           `json:"failed_documents"`
	TotalChunks        int           `json:"total_chunks"`
	ProcessingTime     time.Duration `json:"processing_time"`
	AverageChunkTime   time.Duration `json:"average_chunk_time"`
}

// Factory interfaces for creating components

// ConverterFactory creates converters
type ConverterFactory interface {
	Create(cfg *config.ConverterConfig) (Converter, error)
}

// ExtractorFactory creates extractors
type ExtractorFactory interface {
	Create(cfg *config.ExtractorConfig, llm LLM) (Extractor, error)
}

// EmbedderFactory creates embedders
type EmbedderFactory interface {
	Create(cfg *config.EmbedderConfig) (Embedder, error)
}

// LLMFactory creates LLMs
type LLMFactory interface {
	Create(cfg *config.LLMConfig) (LLM, error)
}

// DatabaseFactory creates database clients
type DatabaseFactory interface {
	Create(cfg *config.DatabaseConfig, embeddingDim int, metadataSchema map[string]interface{}) (DatabaseClient, error)
}

// BenchmarkFactory creates benchmark clients
type BenchmarkFactory interface {
	Create(dbCfg *config.DatabaseConfig, embedCfg *config.EmbedderConfig) (BenchmarkClient, error)
}

// ComponentRegistry manages all component factories
type ComponentRegistry struct {
	Converters ConverterFactory
	Extractors ExtractorFactory
	Embedders  EmbedderFactory
	LLMs       LLMFactory
	Databases  DatabaseFactory
	Benchmarks BenchmarkFactory
}

