package processing

import (
	"fmt"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// ConverterFactory creates converter instances based on configuration
type ConverterFactory struct{}

// NewConverterFactory creates a new converter factory
func NewConverterFactory() *ConverterFactory {
	return &ConverterFactory{}
}

// Create creates a converter instance based on the configuration
func (f *ConverterFactory) Create(cfg *config.ConverterConfig, visionLLM interfaces.LLM) (interfaces.Converter, error) {
	switch cfg.Type {
	case "pymupdf":
		return NewPyMuPDFConverter(cfg)
	case "markitdown":
		return NewMarkItDownConverter(cfg)
	case "docling":
		return NewDoclingConverter(cfg, visionLLM)
	case "docling_vlm":
		return NewDoclingVLMConverter(cfg, visionLLM)
	default:
		return nil, fmt.Errorf("unsupported converter type: %s", cfg.Type)
	}
}

// ExtractorFactory creates extractor instances based on configuration
type ExtractorFactory struct{}

// NewExtractorFactory creates a new extractor factory
func NewExtractorFactory() *ExtractorFactory {
	return &ExtractorFactory{}
}

// Create creates an extractor instance based on the configuration
func (f *ExtractorFactory) Create(cfg *config.ExtractorConfig, llm interfaces.LLM) (interfaces.Extractor, error) {
	switch cfg.Type {
	case "basic":
		return NewBasicExtractor(cfg, llm)
	case "multi_schema":
		return NewMultiSchemaExtractor(cfg, llm)
	default:
		return nil, fmt.Errorf("unsupported extractor type: %s", cfg.Type)
	}
}

// Factory provides both converter and extractor factories
type Factory struct {
	converters *ConverterFactory
	extractors *ExtractorFactory
}

// NewFactory creates a new processing factory
func NewFactory() *Factory {
	return &Factory{
		converters: NewConverterFactory(),
		extractors: NewExtractorFactory(),
	}
}

// CreateConverter creates a converter instance
func (f *Factory) CreateConverter(cfg *config.ConverterConfig, visionLLM interfaces.LLM) (interfaces.Converter, error) {
	return f.converters.Create(cfg, visionLLM)
}

// CreateExtractor creates an extractor instance
func (f *Factory) CreateExtractor(cfg *config.ExtractorConfig, llm interfaces.LLM) (interfaces.Extractor, error) {
	return f.extractors.Create(cfg, llm)
}

// Create is kept for backward compatibility (creates converter)
func (f *Factory) Create(cfg *config.ConverterConfig, visionLLM interfaces.LLM) (interfaces.Converter, error) {
	return f.CreateConverter(cfg, visionLLM)
}

// CreateWithConfig is a standalone function to create converters
func CreateWithConfig(cfg *config.ConverterConfig, visionLLM interfaces.LLM) (interfaces.Converter, error) {
	factory := NewFactory()
	return factory.CreateConverter(cfg, visionLLM)
}

// CreateExtractorWithConfig is a standalone function to create extractors
func CreateExtractorWithConfig(cfg *config.ExtractorConfig, llm interfaces.LLM) (interfaces.Extractor, error) {
	factory := NewFactory()
	return factory.CreateExtractor(cfg, llm)
}
