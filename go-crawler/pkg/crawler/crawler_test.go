package crawler

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go-crawler/internal/config"
)

func TestNewCrawler(t *testing.T) {
	cfg := &config.CrawlerConfig{
		Embeddings: config.EmbedderConfig{
			Provider: "ollama",
			Model:    "test-model",
			BaseURL:  "http://localhost:11434",
		},
		LLM: config.LLMConfig{
			ModelName: "test-llm",
			Provider:  "ollama",
			BaseURL:   "http://localhost:11434",
		},
		Database: config.DatabaseConfig{
			Provider:   "milvus",
			Collection: "test-collection",
		},
		Converter: config.ConverterConfig{
			Type: "markitdown",
		},
		Extractor: config.ExtractorConfig{
			Type: "basic",
		},
		MetadataSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"title": map[string]interface{}{"type": "string"},
			},
		},
		ChunkSize: 1000,
		TempDir:   "tmp/",
	}

	crawler, err := New(cfg)
	require.NoError(t, err)
	assert.NotNil(t, crawler)

	// Test stats
	stats := crawler.GetStats()
	assert.Equal(t, 0, stats.TotalDocuments)
	assert.Equal(t, 0, stats.ProcessedDocuments)
}

func TestCrawlerStats(t *testing.T) {
	cfg := &config.CrawlerConfig{
		Embeddings: config.EmbedderConfig{
			Provider: "ollama",
			Model:    "test-model",
			BaseURL:  "http://localhost:11434",
		},
		LLM: config.LLMConfig{
			ModelName: "test-llm",
			Provider:  "ollama",
			BaseURL:   "http://localhost:11434",
		},
		Database: config.DatabaseConfig{
			Provider:   "milvus",
			Collection: "test-collection",
		},
		Converter: config.ConverterConfig{
			Type: "markitdown",
		},
		Extractor: config.ExtractorConfig{
			Type: "basic",
		},
		MetadataSchema: map[string]interface{}{
			"type": "object",
		},
	}

	crawler, err := New(cfg)
	require.NoError(t, err)

	// Test initial stats
	stats := crawler.GetStats()
	assert.Equal(t, 0, stats.TotalDocuments)
	assert.Equal(t, 0, stats.ProcessedDocuments)
	assert.Equal(t, 0, stats.FailedDocuments)
	assert.Equal(t, 0, stats.TotalChunks)

	// Test chunk text functionality
	chunks, err := crawler.chunkText("This is a test document with some content that should be chunked.", 10)
	require.NoError(t, err)
	assert.True(t, len(chunks) > 1)

	totalLength := 0
	for _, chunk := range chunks {
		totalLength += len(chunk)
		assert.True(t, len(chunk) <= 10 || chunk == chunks[len(chunks)-1]) // Last chunk can be longer
	}
	assert.Equal(t, len("This is a test document with some content that should be chunked."), totalLength)
}

func TestCrawlerFileDetection(t *testing.T) {
	cfg := &config.CrawlerConfig{
		Embeddings: config.EmbedderConfig{
			Provider: "ollama",
			Model:    "test-model",
			BaseURL:  "http://localhost:11434",
		},
		LLM: config.LLMConfig{
			ModelName: "test-llm",
			Provider:  "ollama",
			BaseURL:   "http://localhost:11434",
		},
		Database: config.DatabaseConfig{
			Provider:   "milvus",
			Collection: "test-collection",
		},
		Converter: config.ConverterConfig{
			Type: "markitdown",
		},
		Extractor: config.ExtractorConfig{
			Type: "basic",
		},
		MetadataSchema: map[string]interface{}{
			"type": "object",
		},
	}

	crawler, err := New(cfg)
	require.NoError(t, err)

	// Test supported file types
	supportedFiles := []string{
		"document.pdf",
		"readme.md",
		"data.txt",
		"config.json",
		"table.csv",
		"content.xml",
		"presentation.pptx",
		"spreadsheet.xlsx",
		"document.docx",
		"webpage.html",
	}

	for _, filename := range supportedFiles {
		assert.True(t, crawler.isSupportedFile(filename), "File %s should be supported", filename)
	}

	// Test unsupported file types
	unsupportedFiles := []string{
		"image.png",
		"audio.mp3",
		"video.mp4",
		"archive.zip",
		"executable.exe",
		"random.xyz",
	}

	for _, filename := range unsupportedFiles {
		assert.False(t, crawler.isSupportedFile(filename), "File %s should not be supported", filename)
	}
}

func TestCrawlerBenchmark(t *testing.T) {
	cfg := &config.CrawlerConfig{
		Embeddings: config.EmbedderConfig{
			Provider: "ollama",
			Model:    "test-model",
			BaseURL:  "http://localhost:11434",
		},
		LLM: config.LLMConfig{
			ModelName: "test-llm",
			Provider:  "ollama",
			BaseURL:   "http://localhost:11434",
		},
		Database: config.DatabaseConfig{
			Provider:   "milvus",
			Collection: "test-collection",
		},
		Converter: config.ConverterConfig{
			Type: "markitdown",
		},
		Extractor: config.ExtractorConfig{
			Type: "basic",
		},
		MetadataSchema: map[string]interface{}{
			"type": "object",
		},
	}

	crawler, err := New(cfg)
	require.NoError(t, err)

	// Test benchmark (should return empty results in current implementation)
	ctx := context.Background()
	results, err := crawler.Benchmark(ctx)
	require.NoError(t, err)
	assert.NotNil(t, results)
	assert.NotNil(t, results.ResultsByDoc)
	assert.NotNil(t, results.PlacementDistribution)
	assert.NotNil(t, results.DistanceDistribution)
	assert.NotNil(t, results.PercentInTopK)
	assert.NotNil(t, results.SearchTimeDistribution)
}
