package processing

import (
	"context"
	"fmt"
	"os"
	"strings"

	"go-crawler/internal/config"
)

// PyMuPDFConverter implements a PDF converter similar to PyMuPDF
type PyMuPDFConverter struct {
	config *config.ConverterConfig
}

// NewPyMuPDFConverter creates a new PyMuPDF-style converter
func NewPyMuPDFConverter(cfg *config.ConverterConfig) (*PyMuPDFConverter, error) {
	if cfg.Type != "pymupdf" {
		return nil, fmt.Errorf("invalid converter type: expected 'pymupdf', got '%s'", cfg.Type)
	}

	return &PyMuPDFConverter{
		config: cfg,
	}, nil
}

// Convert converts a PDF file to markdown format
func (p *PyMuPDFConverter) Convert(ctx context.Context, filepath string) (string, error) {
	// Validate file exists
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", filepath)
	}

	// Check file extension
	if !strings.HasSuffix(strings.ToLower(filepath), ".pdf") {
		return "", fmt.Errorf("file is not a PDF: %s", filepath)
	}

	// Placeholder implementation - in a real implementation, this would use a PDF library
	var markdown strings.Builder

	// Add document header
	markdown.WriteString(fmt.Sprintf("# PDF Document: %s\n\n", filepath.Base(filePath)))

	markdown.WriteString("## Document Analysis\n\n")
	markdown.WriteString("*This is a placeholder implementation for PDF processing.*\n\n")
	markdown.WriteString("*In a full implementation, this would:*\n")
	markdown.WriteString("- Extract text from all pages\n")
	markdown.WriteString("- Preserve formatting and layout\n")
	markdown.WriteString("- Handle images and tables\n")
	markdown.WriteString("- Extract metadata\n\n")

	markdown.WriteString("## Implementation Note\n\n")
	markdown.WriteString("To enable full PDF processing, install a PDF library such as:\n")
	markdown.WriteString("- `github.com/ledongthuc/pdf`\n")
	markdown.WriteString("- `github.com/unidoc/unipdf`\n")
	markdown.WriteString("- `github.com/pdfcpu/pdfcpu`\n\n")

	markdown.WriteString(fmt.Sprintf("File path: %s\n", filepath))

	return strings.TrimSpace(markdown.String()), nil
}

// getBoolConfig gets a boolean value from metadata config with default
func (p *PyMuPDFConverter) getBoolConfig(config map[string]interface{}, key string, defaultValue bool) bool {
	if val, exists := config[key]; exists {
		if boolVal, ok := val.(bool); ok {
			return boolVal
		}
	}
	return defaultValue
}
