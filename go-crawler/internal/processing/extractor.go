package processing

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// BasicExtractor implements the Extractor interface using LLMs for metadata extraction
type BasicExtractor struct {
	config                     *config.ExtractorConfig
	llm                        interfaces.LLM
	documentLibraryContext     string
	generateBenchmarkQuestions bool
	numBenchmarkQuestions      int
}

// NewBasicExtractor creates a new basic extractor
func NewBasicExtractor(cfg *config.ExtractorConfig, llm interfaces.LLM) (*BasicExtractor, error) {
	if cfg.Type != "basic" {
		return nil, fmt.Errorf("invalid extractor type: expected 'basic', got '%s'", cfg.Type)
	}

	return &BasicExtractor{
		config:                     cfg,
		llm:                        llm,
		documentLibraryContext:     "",
		generateBenchmarkQuestions: false,
		numBenchmarkQuestions:      3,
	}, nil
}

// ExtractMetadata extracts metadata from text using LLM and JSON schema
func (e *BasicExtractor) ExtractMetadata(ctx context.Context, text string) (map[string]interface{}, error) {
	if e.config.MetadataSchema == nil || len(e.config.MetadataSchema) == 0 {
		// Return basic metadata if no schema is provided
		return map[string]interface{}{
			"file_size":    len(text),
			"processed_at": time.Now().Format(time.RFC3339),
		}, nil
	}

	// Generate the prompt
	prompt := e.generatePrompt(text)

	// Create LLM options for structured output
	options := &interfaces.LLMOptions{
		ResponseFormat: e.config.MetadataSchema,
		MaxTokens:      getMaxTokens(e.config.MetadataSchema),
		Temperature:    getTemperature(),
	}

	// Make LLM call
	result, err := e.llm.Invoke(ctx, prompt, options)
	if err != nil {
		return nil, fmt.Errorf("failed to extract metadata: %v", err)
	}

	// Parse the result
	var metadata map[string]interface{}
	if strResult, ok := result.(string); ok {
		// Try to parse as JSON
		if err := json.Unmarshal([]byte(strResult), &metadata); err != nil {
			return nil, fmt.Errorf("failed to parse LLM response as JSON: %v", err)
		}
	} else if mapResult, ok := result.(map[string]interface{}); ok {
		metadata = mapResult
	} else {
		return nil, fmt.Errorf("unexpected LLM response type: %T", result)
	}

	// Validate required fields
	if err := e.validateRequiredFields(metadata); err != nil {
		return nil, fmt.Errorf("metadata validation failed: %v", err)
	}

	// Generate benchmark questions if enabled
	if e.generateBenchmarkQuestions {
		if questions, err := e.generateBenchmarkQuestionsFunc(ctx, text); err == nil {
			metadata["benchmark_questions"] = questions
		}
	}

	return metadata, nil
}

// ChunkText splits text into chunks for processing
func (e *BasicExtractor) ChunkText(text string, chunkSize int) ([]string, error) {
	if chunkSize <= 0 {
		chunkSize = 1000
	}

	var chunks []string
	runes := []rune(text)

	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunk := string(runes[i:end])
		if strings.TrimSpace(chunk) != "" {
			chunks = append(chunks, chunk)
		}
	}

	return chunks, nil
}

// generatePrompt creates the LLM prompt for metadata extraction
func (e *BasicExtractor) generatePrompt(text string) string {
	schemaJSON, _ := json.MarshalIndent(e.config.MetadataSchema, "", "  ")

	var prompt strings.Builder
	prompt.WriteString("You are an expert metadata extraction engine. Read the document and output a single JSON object that conforms EXACTLY to the provided JSON Schema.\n\n")

	prompt.WriteString("STRICT RULES:\n")
	prompt.WriteString("- Use ONLY the keys defined in the schema's properties.\n")
	prompt.WriteString("- Include ALL fields required by the schema.\n")
	prompt.WriteString("- Do NOT invent or copy fields that are not in the schema.\n")
	prompt.WriteString("- Normalize values (e.g., convert dates to YYYY-MM-DD, strip markup/artifacts).\n")
	prompt.WriteString("- If a required field is missing or cannot be inferred, set its value to \"Unknown\".\n")
	prompt.WriteString("- Validate each value against the schema types/formats.\n")
	prompt.WriteString("- Output must be ONLY a JSON object (no extra text, no markdown fences).\n\n")

	prompt.WriteString("JSON Schema:\n")
	prompt.WriteString("```json\n")
	prompt.WriteString(string(schemaJSON))
	prompt.WriteString("\n```\n\n")

	if e.documentLibraryContext != "" {
		prompt.WriteString("Document Library Context (do not echo in output; use only for disambiguation):\n")
		prompt.WriteString(e.documentLibraryContext)
		prompt.WriteString("\n\n")
	}

	prompt.WriteString("Document:\n")
	prompt.WriteString("```\n")
	// Truncate text if too long to avoid token limits
	if len(text) > 8000 {
		prompt.WriteString(text[:8000])
		prompt.WriteString("...\n[Content truncated for length]")
	} else {
		prompt.WriteString(text)
	}
	prompt.WriteString("\n```")

	return prompt.String()
}

// validateRequiredFields validates that all required fields are present
func (e *BasicExtractor) validateRequiredFields(metadata map[string]interface{}) error {
	required, ok := e.config.MetadataSchema["required"].([]interface{})
	if !ok {
		return nil // No required fields specified
	}

	var missingFields []string
	for _, req := range required {
		fieldName, ok := req.(string)
		if !ok {
			continue
		}

		if _, exists := metadata[fieldName]; !exists {
			missingFields = append(missingFields, fieldName)
		}
	}

	if len(missingFields) > 0 {
		return fmt.Errorf("missing required fields: %v", missingFields)
	}

	return nil
}

// generateBenchmarkQuestionsFunc generates benchmark questions for the document
func (e *BasicExtractor) generateBenchmarkQuestionsFunc(ctx context.Context, text string) ([]string, error) {
	prompt := fmt.Sprintf(`You are an expert at creating benchmark questions for document retrieval systems.

Given the following document text, generate exactly %d diverse questions that could be answered by this document. Each question should:
- Be answerable using information from the document
- Cover different aspects of the document content
- Be specific and unambiguous
- Vary in complexity and topic coverage

Respond with a JSON array of exactly %d strings, containing only the questions.

Document text:
%s

Questions:`, e.numBenchmarkQuestions, e.numBenchmarkQuestions, truncateText(text, 4000))

	result, err := e.llm.Invoke(ctx, prompt, nil)
	if err != nil {
		return nil, err
	}

	var questions []string
	if strResult, ok := result.(string); ok {
		if err := json.Unmarshal([]byte(strResult), &questions); err != nil {
			// Try to extract questions from text response
			lines := strings.Split(strResult, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if strings.HasSuffix(line, "?") {
					questions = append(questions, line)
				}
			}
		}
	} else {
		return nil, fmt.Errorf("unexpected LLM response type: %T", result)
	}

	if len(questions) != e.numBenchmarkQuestions {
		return questions, fmt.Errorf("expected %d questions, got %d", e.numBenchmarkQuestions, len(questions))
	}

	return questions, nil
}

// MultiSchemaExtractor implements the Extractor interface with support for multiple schemas
type MultiSchemaExtractor struct {
	config             *config.ExtractorConfig
	llm                interfaces.LLM
	schemas            []map[string]interface{}
	libraryDescription string
}

// NewMultiSchemaExtractor creates a new multi-schema extractor
func NewMultiSchemaExtractor(cfg *config.ExtractorConfig, llm interfaces.LLM) (*MultiSchemaExtractor, error) {
	if cfg.Type != "multi_schema" {
		return nil, fmt.Errorf("invalid extractor type: expected 'multi_schema', got '%s'", cfg.Type)
	}

	var schemas []map[string]interface{}
	if cfg.MetadataSchema != nil {
		if schemaArray, ok := cfg.MetadataSchema["schemas"].([]interface{}); ok {
			for _, s := range schemaArray {
				if schemaMap, ok := s.(map[string]interface{}); ok {
					schemas = append(schemas, schemaMap)
				}
			}
		} else {
			// Single schema
			schemas = append(schemas, cfg.MetadataSchema)
		}
	}

	if len(schemas) == 0 {
		return nil, fmt.Errorf("no schemas provided for multi-schema extractor")
	}

	return &MultiSchemaExtractor{
		config:             cfg,
		llm:                llm,
		schemas:            schemas,
		libraryDescription: "",
	}, nil
}

// ExtractMetadata extracts metadata using multiple schemas
func (m *MultiSchemaExtractor) ExtractMetadata(ctx context.Context, text string) (map[string]interface{}, error) {
	metadata := make(map[string]interface{})

	for i, schema := range m.schemas {
		extractor := &BasicExtractor{
			config: &config.ExtractorConfig{
				Type:           "basic",
				MetadataSchema: schema,
			},
			llm:                        m.llm,
			documentLibraryContext:     m.libraryDescription,
			generateBenchmarkQuestions: false,
			numBenchmarkQuestions:      3,
		}

		schemaMetadata, err := extractor.ExtractMetadata(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to extract metadata for schema %d: %v", i, err)
		}

		// Merge metadata from this schema
		for key, value := range schemaMetadata {
			metadata[key] = value
		}
	}

	return metadata, nil
}

// ChunkText splits text into chunks for processing
func (m *MultiSchemaExtractor) ChunkText(text string, chunkSize int) ([]string, error) {
	if chunkSize <= 0 {
		chunkSize = 1000
	}

	var chunks []string
	runes := []rune(text)

	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunk := string(runes[i:end])
		if strings.TrimSpace(chunk) != "" {
			chunks = append(chunks, chunk)
		}
	}

	return chunks, nil
}

// Helper functions

// getMaxTokens calculates maximum tokens based on schema complexity
func getMaxTokens(schema map[string]interface{}) *int {
	// Estimate tokens needed based on schema complexity
	properties, ok := schema["properties"].(map[string]interface{})
	if !ok {
		tokens := 1000
		return &tokens
	}

	complexity := len(properties)
	tokens := 500 + complexity*100
	return &tokens
}

// getTemperature returns temperature for extraction (low for consistency)
func getTemperature() *float64 {
	temp := 0.1
	return &temp
}

// truncateText truncates text to specified length
func truncateText(text string, maxLength int) string {
	if len(text) <= maxLength {
		return text
	}

	runes := []rune(text)
	if len(runes) <= maxLength {
		return text
	}

	return string(runes[:maxLength]) + "..."
}
