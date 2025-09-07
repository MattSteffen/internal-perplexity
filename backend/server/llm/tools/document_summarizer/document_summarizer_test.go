package document_summarizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDocumentSummarizer_Name(t *testing.T) {
	summarizer := NewDocumentSummarizer()
	assert.Equal(t, "document_summarizer", summarizer.Name())
}

func TestDocumentSummarizer_Description(t *testing.T) {
	summarizer := NewDocumentSummarizer()
	assert.Contains(t, summarizer.Description(), "Summarizes")
	assert.Contains(t, summarizer.Description(), "intelligent")
}

func TestDocumentSummarizer_Schema(t *testing.T) {
	summarizer := NewDocumentSummarizer()
	schema := summarizer.Schema()

	assert.NotNil(t, schema)
	assert.Equal(t, "object", schema.Type)
	assert.Contains(t, schema.Properties, "content")
	assert.Contains(t, schema.Properties, "max_length")
	assert.Contains(t, schema.Required, "content")
}

func TestDocumentSummarizer_Definition(t *testing.T) {
	summarizer := NewDocumentSummarizer()
	definition := summarizer.Definition()

	assert.NotNil(t, definition)
	assert.Equal(t, "function", definition.Type)
	assert.NotNil(t, definition.Function)
	assert.Equal(t, "document_summarizer", definition.Function.Name)
	assert.Contains(t, definition.Function.Description, "summarize")
	assert.NotNil(t, definition.Function.Parameters)
}

func TestNewDocumentSummarizer(t *testing.T) {
	summarizer := NewDocumentSummarizer()
	assert.NotNil(t, summarizer)
	assert.IsType(t, &DocumentSummarizer{}, summarizer)
}
