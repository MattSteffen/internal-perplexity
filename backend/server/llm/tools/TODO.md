# LLM Tools - MVP Tasks

## Overview
Build deterministic tools with predictable I/O that integrate with the LLM system. Focus on core tools needed for agent functionality and test them thoroughly.

## Core Tool Development

### 1. Tool Framework
- [ ] Create `Tool` interface with Execute method
- [ ] Implement `ToolRegistry` for tool discovery
- [ ] Add JSON schema validation for inputs
- [ ] Create `ToolInput` and `ToolResult` types
- [ ] Add tool metadata (name, description, schema)

```go
type Tool interface {
    Name() string
    Description() string
    Schema() *ToolSchema
    Execute(ctx context.Context, input *ToolInput) (*ToolResult, error)
}

type ToolRegistry struct {
    tools map[string]Tool
}

func (r *ToolRegistry) Register(tool Tool) {
    r.tools[tool.Name()] = tool
}
```

### 2. Input Validation Framework
- [ ] Create `ToolValidationError` and `ToolValidationErrors` types
- [ ] Implement `ToolValidator` interface for tools
- [ ] Add validation error codes and messages
- [ ] Create validation helper functions

### 3. Document Summarizer Tool
- [ ] Create `document_summarizer/` directory
- [ ] Implement summarizer with LLM integration
- [ ] Add comprehensive input validation
- [ ] Add configurable max length parameter
- [ ] Ensure ≤1 LLM call per execution
- [ ] Add compression ratio tracking

### 4. Calculator Tool
- [ ] Create `calculator/` directory
- [ ] Implement basic arithmetic operations (+, -, *, /)
- [ ] Add expression parsing and validation
- [ ] Handle parentheses and operator precedence
- [ ] Add division by zero protection
- [ ] Implement input validation for expressions
- [ ] Add support for decimal numbers
- [ ] Test edge cases and error conditions

### 5. Retriever Tool
- [x] Create `retriever/` directory
- [x] Implement Milvus query interface
- [x] Add hybrid search capabilities
- [x] Implement input validation for queries
- [x] Add partition support
- [x] Integrate with tool registry
- [x] Create comprehensive tests
- [ ] Add actual Milvus client integration
- [ ] Implement text embedding
- [ ] Add vector search execution

```go
type DocumentSummarizer struct {
    llmClient LLMProvider
}

func (s *DocumentSummarizer) Execute(ctx context.Context, input *ToolInput) (*ToolResult, error) {
    content := input.Data["content"].(string)
    maxLength := input.Data["max_length"].(int)

    // Single LLM call for summarization
    messages := []Message{
        {Role: "system", Content: "You are a document summarizer."},
        {Role: "user", Content: fmt.Sprintf("Summarize this text in %d words or less: %s", maxLength, content)},
    }

    resp, err := s.llmClient.Complete(ctx, messages, CompletionOptions{})
    // Process response...
}
```

### 3. Web Search Tool (Mock)
- [ ] Create `web_search/` directory
- [ ] Implement mock web search functionality
- [ ] Add configurable result limits
- [ ] Return structured search results
- [ ] Add source credibility scoring

```go
type WebSearchTool struct {
    maxResults int
}

func (w *WebSearchTool) Execute(ctx context.Context, input *ToolInput) (*ToolResult, error) {
    query := input.Data["query"].(string)
    maxResults := input.Data["max_results"].(int)

    // Mock search results for testing
    results := []SearchResult{
        {
            Title: "Mock Result 1",
            URL: "https://example.com/1",
            Snippet: "This is a mock search result for testing.",
            Credibility: 0.8,
        },
        // ... more mock results
    }

    return &ToolResult{
        Success: true,
        Data:    map[string]interface{}{"results": results},
    }, nil
}
```

## Testing Tasks

### 4. Unit Tests
- [ ] Test tool registry registration and discovery
- [ ] Test JSON schema validation
- [ ] Test tool input/output types
- [ ] Test error handling for invalid inputs

### 5. Validation Tests
- [ ] Test tool input validation with various error cases
- [ ] Test validation error messages and codes
- [ ] Test boundary conditions and edge cases
- [ ] Test schema-based validation

### 6. Integration Tests
- [ ] Test document summarizer with localhost:11434
- [ ] Test web search tool execution
- [ ] Test tool result formatting
- [ ] Test concurrent tool execution

```go
func TestDocumentSummarizerIntegration(t *testing.T) {
    llmClient := NewMockLLMClient("http://localhost:11434/v1", "gpt-oss:20b")
    summarizer := NewDocumentSummarizer(llmClient)
    registry := NewToolRegistry()
    registry.Register(summarizer)

    input := &ToolInput{
        Name: "document_summarizer",
        Data: map[string]interface{}{
            "content":     "This is a long document that needs to be summarized...",
            "max_length":  50,
        },
    }

    result, err := registry.Execute(context.Background(), input)
    assert.NoError(t, err)
    assert.True(t, result.Success)
    assert.Contains(t, result.Data, "summary")
}
```

### 7. Tool Validation
- [ ] Ensure all tools are deterministic (same input = same output)
- [ ] Verify ≤1 LLM call per tool execution
- [ ] Test tool schemas are valid JSON
- [ ] Validate tool result structures
- [ ] Test input validation returns proper error codes
- [ ] Test validation schema compliance

## Tool Categories

### 8. Content Processing Tools
- [ ] Text analyzer (word count, readability)
- [ ] Format converter (markdown to HTML)
- [ ] Content extractor (key points, topics)

### 9. Data Processing Tools
- [ ] CSV processor (parse, validate, transform)
- [ ] JSON validator and formatter
- [ ] Data cleaner (remove duplicates, normalize)

## Implementation Priority

### Phase 1: Core Framework
1. [ ] Implement Tool interface and registry
2. [ ] Create basic tool types and schemas
3. [ ] Add input validation
4. [ ] Test framework functionality

### Phase 2: Essential Tools
1. [ ] Build document summarizer with LLM integration
2. [ ] Build calculator tool with expression evaluation
3. [ ] Create web search mock tool
4. [ ] Add tool result formatting
5. [ ] Test with localhost:11434

### Phase 3: Extended Tools
1. [ ] Add more content processing tools
2. [ ] Implement data processing tools
3. [ ] Add tool chaining capabilities
4. [ ] Comprehensive testing

## Configuration

### 10. Tool Configuration
- [ ] Add tool enable/disable configuration
- [ ] Configure tool-specific parameters
- [ ] Add tool execution timeouts
- [ ] Configure LLM models per tool

```yaml
tools:
  document_summarizer:
    enabled: true
    max_length: 200
    model: "gpt-oss:20b"
  web_search:
    enabled: true
    max_results: 10
    mock_mode: true
```

## Success Criteria
- [ ] All tools execute deterministically
- [ ] Tools integrate properly with LLM providers
- [ ] JSON schemas are valid and enforced
- [ ] Tools work with localhost:11434 (gpt-oss:20b)
- [ ] Tool registry provides proper discovery
- [ ] Error handling is comprehensive

## Files to Create
- [x] `llm/tools/registry.go`
- [x] `llm/tools/types.go`
- [x] `llm/tools/document_summarizer/document_summarizer.go`
- [x] `llm/tools/retriever/retriever.go`
- [x] `llm/tools/retriever/retriever_test.go`
- [x] `llm/tools/retriever/README.md`
- [x] `llm/tools/retriever/example_usage.go`
- [ ] `llm/tools/web_search/web_search.go`
- [ ] `llm/tools/registry_test.go`
- [ ] `llm/tools/document_summarizer/document_summarizer_test.go`
