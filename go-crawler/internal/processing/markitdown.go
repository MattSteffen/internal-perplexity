package processing

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"go-crawler/internal/config"
)

// MarkItDownConverter implements a general file converter similar to MarkItDown
type MarkItDownConverter struct {
	config *config.ConverterConfig
}

// NewMarkItDownConverter creates a new general file converter
func NewMarkItDownConverter(cfg *config.ConverterConfig) (*MarkItDownConverter, error) {
	if cfg.Type != "markitdown" {
		return nil, fmt.Errorf("invalid converter type: expected 'markitdown', got '%s'", cfg.Type)
	}

	return &MarkItDownConverter{
		config: cfg,
	}, nil
}

// Convert converts various file formats to markdown
func (m *MarkItDownConverter) Convert(ctx context.Context, filePath string) (string, error) {
	// Validate file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", filePath)
	}

	// Determine file type and convert accordingly
	ext := strings.ToLower(filepath.Ext(filePath))
	filename := filepath.Base(filePath)

	switch ext {
	case ".txt":
		return m.convertTextFile(filePath, filename)
	case ".md":
		return m.convertMarkdownFile(filePath, filename)
	case ".html", ".htm":
		return m.convertHTMLFile(filePath, filename)
	case ".json":
		return m.convertJSONFile(filePath, filename)
	case ".csv":
		return m.convertCSVFile(filePath, filename)
	case ".xml":
		return m.convertXMLFile(filePath, filename)
	case ".pdf":
		// For PDF, we'll use a simple text extraction
		return m.convertPDFFile(filePath, filename)
	default:
		return "", fmt.Errorf("unsupported file type: %s", ext)
	}
}

// convertTextFile converts plain text files
func (m *MarkItDownConverter) convertTextFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read text file: %v", err)
	}

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))
	markdown.WriteString("```\n")
	markdown.Write(content)
	markdown.WriteString("\n```\n")

	return markdown.String(), nil
}

// convertMarkdownFile converts markdown files
func (m *MarkItDownConverter) convertMarkdownFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read markdown file: %v", err)
	}

	// For markdown files, we can mostly return as-is but add a header
	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))
	markdown.Write(content)

	return markdown.String(), nil
}

// convertHTMLFile converts HTML files to markdown
func (m *MarkItDownConverter) convertHTMLFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read HTML file: %v", err)
	}

	htmlContent := string(content)

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))

	// Simple HTML to markdown conversion
	markdown.WriteString(m.htmlToMarkdown(htmlContent))

	return markdown.String(), nil
}

// convertJSONFile converts JSON files to markdown
func (m *MarkItDownConverter) convertJSONFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read JSON file: %v", err)
	}

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))
	markdown.WriteString("```json\n")
	markdown.Write(content)
	markdown.WriteString("\n```\n")

	return markdown.String(), nil
}

// convertCSVFile converts CSV files to markdown tables
func (m *MarkItDownConverter) convertCSVFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read CSV file: %v", err)
	}

	csvContent := string(content)

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))

	// Convert CSV to markdown table
	markdown.WriteString(m.csvToMarkdownTable(csvContent))

	return markdown.String(), nil
}

// convertXMLFile converts XML files to markdown
func (m *MarkItDownConverter) convertXMLFile(filePath, filename string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read XML file: %v", err)
	}

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))
	markdown.WriteString("```xml\n")
	markdown.Write(content)
	markdown.WriteString("\n```\n")

	return markdown.String(), nil
}

// convertPDFFile provides a basic PDF text extraction fallback
func (m *MarkItDownConverter) convertPDFFile(filePath, filename string) (string, error) {
	// This is a fallback for when PyMuPDF is not available
	// In practice, you'd want to use a proper PDF library

	var markdown strings.Builder
	markdown.WriteString(fmt.Sprintf("# %s\n\n", filename))
	markdown.WriteString("*PDF file detected. Use PyMuPDF converter for full text extraction.*\n")
	markdown.WriteString(fmt.Sprintf("File path: %s\n", filepath))

	return markdown.String(), nil
}

// htmlToMarkdown performs a simple HTML to markdown conversion
func (m *MarkItDownConverter) htmlToMarkdown(html string) string {
	// Simple HTML to markdown conversion
	result := html

	// Replace common HTML tags
	replacements := map[string]string{
		"<h1>":      "# ",
		"</h1>":     "\n",
		"<h2>":      "## ",
		"</h2>":     "\n",
		"<h3>":      "### ",
		"</h3>":     "\n",
		"<p>":       "",
		"</p>":      "\n\n",
		"<br>":      "\n",
		"<br/>":     "\n",
		"<strong>":  "**",
		"</strong>": "**",
		"<em>":      "*",
		"</em>":     "*",
		"<b>":       "**",
		"</b>":      "**",
		"<i>":       "*",
		"</i>":      "*",
	}

	for old, new := range replacements {
		result = strings.ReplaceAll(result, old, new)
	}

	// Remove remaining HTML tags
	result = regexp.MustCompile(`<[^>]*>`).ReplaceAllString(result, "")

	// Clean up extra whitespace
	result = regexp.MustCompile(`\n{3,}`).ReplaceAllString(result, "\n\n")

	return strings.TrimSpace(result)
}

// csvToMarkdownTable converts CSV content to a markdown table
func (m *MarkItDownConverter) csvToMarkdownTable(csv string) string {
	lines := strings.Split(strings.TrimSpace(csv), "\n")
	if len(lines) == 0 {
		return ""
	}

	var markdown strings.Builder

	// Process header row
	header := strings.Split(lines[0], ",")
	markdown.WriteString("| " + strings.Join(header, " | ") + " |\n")
	markdown.WriteString("| " + strings.Repeat("--- | ", len(header)) + "\n")

	// Process data rows
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "" {
			continue
		}
		row := strings.Split(lines[i], ",")
		markdown.WriteString("| " + strings.Join(row, " | ") + " |\n")
	}

	return markdown.String()
}
