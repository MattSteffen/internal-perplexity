# Summary Sub-Agent

Specialized agent for content summarization that takes a list of text contents and instructions to generate structured summaries.

## Purpose
The summary agent handles:
- Multi-document content summarization
- Instruction-based summarization focus
- Structured output generation
- Configurable summary length and format
- Single LLM call per summarization task

## Implementation Structure

### summary.go
**Contents:**
- `SummaryAgent` struct implementing `Agent` interface
- Content processing and instruction parsing
- LLM-powered summarization with structured prompts
- Result formatting with metadata

**Key Methods:**
```go
type SummaryAgent struct {
    llmClient LLMProvider
}

func (s *SummaryAgent) Execute(ctx context.Context, input *AgentInput) (*AgentResult, error) {
    // 1. Parse input parameters
    contents := input.Data["contents"].([]string)
    instructions := input.Data["instructions"].(string)
    focusAreas := input.Data["focus_areas"].([]string)

    // 2. Combine contents for processing
    combinedContent := strings.Join(contents, "\n\n")

    // 3. Build structured summarization prompt
    prompt := s.buildSummaryPrompt(combinedContent, instructions, focusAreas)

    // 4. Execute LLM summarization
    messages := []openai.ChatCompletionMessage{
        {Role: "system", Content: "You are a professional summarizer."},
        {Role: "user", Content: prompt},
    }

    resp, err := s.llmClient.Complete(ctx, messages, CompletionOptions{})
    if err != nil {
        return nil, err
    }

    // 5. Parse and structure response
    summary := s.parseSummary(resp.Content)

    return &AgentResult{
        Content: &SummaryResult{
            Summary:     summary,
            FocusAreas:  focusAreas,
            ContentCount: len(contents),
        },
        Success: true,
        TokensUsed: resp.TokensUsed,
        Duration: time.Since(start),
        Metadata: map[string]interface{}{
            "input_length": len(combinedContent),
            "focus_areas": focusAreas,
        },
    }, nil
}
```

### Input/Output Structure
**Input:**
```go
type AgentInput struct {
    Data map[string]interface{}{
        "contents": []string{
            "First document content...",
            "Second document content...",
        },
        "instructions": "Focus on key findings and recommendations",
        "focus_areas": []string{"findings", "recommendations", "conclusions"},
    }
}
```

**Output:**
```go
type SummaryResult struct {
    Summary      string   `json:"summary"`
    FocusAreas   []string `json:"focus_areas"`
    ContentCount int      `json:"content_count"`
    Sections     []Section `json:"sections,omitempty"`
}

type Section struct {
    Title   string `json:"title"`
    Content string `json:"content"`
}
```

## Input Validation

The summary agent performs comprehensive validation on all input parameters before processing.

### Validation Rules

**Required Fields:**
- `contents`: Non-empty array of strings
- `instructions`: Non-empty string with summarization guidance

**Optional Fields:**
- `focus_areas`: Array of strings (no duplicates allowed)
- `max_length`: Positive integer (default: 500)
- `format`: String enum: "concise", "detailed", "bullet_points"

### Validation Examples

```go
// Valid input
input := &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{
            "Go is a programming language...",
            "It features concurrency...",
        },
        "instructions": "Summarize the key features",
        "focus_areas": []string{"features", "benefits"},
        "max_length": 300,
    },
}
err := summaryAgent.Validate(input)
// Returns: nil (valid)

// Missing contents
input := &AgentInput{
    Data: map[string]interface{}{
        "instructions": "Summarize this",
    },
}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "MISSING_REQUIRED_FIELD", Field: "contents"}

// Empty contents array
input := &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{},
        "instructions": "Summarize this",
    },
}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "EMPTY_CONTENTS", Field: "contents"}

// Invalid content type
input := &AgentInput{
    Data: map[string]interface{}{
        "contents": "not an array",
        "instructions": "Summarize this",
    },
}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "INVALID_FIELD_TYPE", Field: "contents", Value: "string"}

// Duplicate focus areas
input := &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{"content1", "content2"},
        "instructions": "Summarize this",
        "focus_areas": []string{"features", "features", "benefits"},
    },
}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "DUPLICATE_FOCUS_AREAS", Field: "focus_areas"}

// Invalid max_length
input := &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{"content1", "content2"},
        "instructions": "Summarize this",
        "max_length": -100,
    },
}
err := summaryAgent.Validate(input)
// Returns: ValidationError{Code: "INVALID_FIELD_VALUE", Field: "max_length", Value: -100}
```

### Validation Error Types

| Error Code | Description | Field |
|------------|-------------|-------|
| `MISSING_REQUIRED_FIELD` | Required field is missing | varies |
| `EMPTY_CONTENTS` | Contents array is empty | `contents` |
| `INVALID_FIELD_TYPE` | Field has wrong data type | varies |
| `INVALID_FIELD_VALUE` | Field value is invalid | varies |
| `DUPLICATE_FOCUS_AREAS` | Focus areas contain duplicates | `focus_areas` |
| `CONTENT_TOO_LONG` | Individual content exceeds limit | `contents` |
| `INSTRUCTION_TOO_LONG` | Instructions exceed character limit | `instructions` |

### Content Size Limits
- **Individual Content**: Maximum 50,000 characters
- **Combined Contents**: Maximum 200,000 characters
- **Instructions**: Maximum 2,000 characters
- **Focus Areas**: Maximum 10 areas, each ≤ 50 characters

## Usage Example

```go
summaryAgent := NewSummaryAgent(llmClient)

result, err := summaryAgent.Execute(ctx, &AgentInput{
    Data: map[string]interface{}{
        "contents": []string{
            "Go is a statically typed programming language...",
            "It features garbage collection and concurrent execution...",
        },
        "instructions": "Summarize the key features and benefits",
        "focus_areas": []string{"features", "benefits"},
    },
})

if result.Success {
    summaryData := result.Content.(*SummaryResult)
    fmt.Printf("Summary: %s\n", summaryData.Summary)
    fmt.Printf("Processed %d content items\n", summaryData.ContentCount)
}
```

## Performance Characteristics
- **Latency**: 1-5 seconds depending on content length
- **Token Usage**: 200-800 tokens per summarization
- **Success Rate**: >98% for well-formed inputs
- **Deterministic**: Same input produces consistent results

## Configuration Options
- **Content Length**: Maximum combined content length
- **Summary Length**: Target summary length
- **Focus Areas**: Specific areas to emphasize
- **Output Format**: Structured vs. free-form summaries
- **Language**: Target language for summaries

## Monitoring Stats
Returns execution statistics:
- Content items processed
- Input text length
- Token usage breakdown
- Processing duration
- Focus areas covered
