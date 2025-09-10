package processing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go-crawler/internal/config"
)

func TestMarkItDownConverter(t *testing.T) {
	cfg := &config.ConverterConfig{
		Type: "markitdown",
	}

	converter, err := NewMarkItDownConverter(cfg)
	require.NoError(t, err)
	assert.NotNil(t, converter)

	// Test with non-existent file (should fail)
	_, err = converter.Convert(nil, "/nonexistent/file.txt")
	assert.Error(t, err)
}

func TestBasicExtractor(t *testing.T) {
	cfg := &config.ExtractorConfig{
		Type: "basic",
		MetadataSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"title":  map[string]interface{}{"type": "string"},
				"author": map[string]interface{}{"type": "string"},
			},
			"required": []string{"title"},
		},
	}

	// Mock LLM - in real tests, you'd use a mock
	llm := &mockLLM{}
	extractor, err := NewBasicExtractor(cfg, llm)
	require.NoError(t, err)
	assert.NotNil(t, extractor)

	// Test chunking
	chunks, err := extractor.ChunkText("This is a test document with some content that should be split into chunks.", 20)
	require.NoError(t, err)
	assert.True(t, len(chunks) > 1)

	totalLength := 0
	for _, chunk := range chunks {
		totalLength += len(chunk)
	}
	assert.Equal(t, len("This is a test document with some content that should be split into chunks."), totalLength)
}

func TestMultiSchemaExtractor(t *testing.T) {
	schema1 := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"title": map[string]interface{}{"type": "string"},
		},
		"required": []string{"title"},
	}

	schema2 := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"author": map[string]interface{}{"type": "string"},
		},
		"required": []string{"author"},
	}

	cfg := &config.ExtractorConfig{
		Type: "multi_schema",
		MetadataSchema: map[string]interface{}{
			"schemas": []interface{}{schema1, schema2},
		},
	}

	// Mock LLM
	llm := &mockLLM{}
	extractor, err := NewMultiSchemaExtractor(cfg, llm)
	require.NoError(t, err)
	assert.NotNil(t, extractor)

	// Test chunking
	chunks, err := extractor.ChunkText("Test content for chunking.", 10)
	require.NoError(t, err)
	assert.True(t, len(chunks) > 1)
}

// mockLLM is a mock implementation for testing
type mockLLM struct{}

func (m *mockLLM) Invoke(ctx context.Context, prompt string, options interface{}) (interface{}, error) {
	return map[string]interface{}{
		"title":  "Test Title",
		"author": "Test Author",
	}, nil
}

func (m *mockLLM) InvokeWithMessages(ctx context.Context, messages []interface{}, options interface{}) (interface{}, error) {
	return m.Invoke(ctx, "", options)
}

func (m *mockLLM) GetContextLength() int {
	return 4096
}

func TestConverterFactory(t *testing.T) {
	factory := NewConverterFactory()

	// Test creating MarkItDown converter
	cfg := &config.ConverterConfig{Type: "markitdown"}
	converter, err := factory.Create(cfg, nil)
	require.NoError(t, err)
	assert.NotNil(t, converter)
	assert.IsType(t, &MarkItDownConverter{}, converter)

	// Test unsupported converter type
	cfg = &config.ConverterConfig{Type: "unsupported"}
	_, err = factory.Create(cfg, nil)
	assert.Error(t, err)
}

func TestExtractorFactory(t *testing.T) {
	factory := NewExtractorFactory()

	// Mock LLM
	llm := &mockLLM{}

	// Test creating basic extractor
	cfg := &config.ExtractorConfig{
		Type: "basic",
		MetadataSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"title": map[string]interface{}{"type": "string"},
			},
		},
	}
	extractor, err := factory.Create(cfg, llm)
	require.NoError(t, err)
	assert.NotNil(t, extractor)
	assert.IsType(t, &BasicExtractor{}, extractor)

	// Test unsupported extractor type
	cfg = &config.ExtractorConfig{Type: "unsupported"}
	_, err = factory.Create(cfg, llm)
	assert.Error(t, err)
}
