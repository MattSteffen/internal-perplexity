# Summary Agent

The Summary Agent is a specialized sub-agent that processes lists of text content and generates focused, coherent summaries using Large Language Models (LLMs). It handles multi-document summarization with customizable instructions and focus areas.

## Purpose

The Summary Agent provides intelligent document summarization capabilities for:
- **Multi-Document Processing**: Summarize multiple related documents simultaneously
- **Instruction-Based Summarization**: Generate summaries following specific guidelines
- **Focused Summaries**: Emphasize particular areas of interest
- **Structured Output**: Consistent, well-formatted summary results
- **Single LLM Call Efficiency**: Optimized processing with minimal API usage

## Features

- **Multi-Content Processing**: Handle multiple documents in a single operation
- **Instruction Parsing**: Process and apply custom summarization instructions
- **Focus Area Support**: Emphasize specific topics or aspects
- **Comprehensive Validation**: Robust input validation with detailed error messages
- **Statistics Tracking**: Monitor performance and usage metrics
- **Flexible Output**: Adaptable summary formats and lengths

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `contents` | array | Yes | Array of text content strings to summarize |
| `instructions` | string | No | Specific instructions for summarization approach |
| `focus_areas` | array | No | List of areas to emphasize in the summary |

### Content Requirements
- **Type**: Array of strings
- **Minimum Items**: 1 content item required
- **Maximum Items**: No hard limit, but performance may degrade with large arrays
- **Content Length**: Individual items should be reasonable text lengths

### Instructions Format
- **Type**: String
- **Purpose**: Guide the summarization approach and style
- **Examples**: "Focus on key findings", "Summarize technical details", "Highlight benefits and drawbacks"

### Focus Areas
- **Type**: Array of strings
- **Purpose**: Specify areas of emphasis
- **Examples**: `["findings", "recommendations", "conclusions"]`

## Output Schema

```json
{
  "success": true,
  "content": {
    "summary": "Comprehensive summary of all provided content...",
    "metadata": {
      "content_count": 3,
      "combined_length": 2500,
      "focus_areas": ["findings", "recommendations"],
      "instructions": "Focus on key insights and actionable recommendations"
    }
  },
  "tokens_used": 450,
  "duration": "2.1s",
  "metadata": {
    "input_length": 2500,
    "focus_areas": ["findings", "recommendations"],
    "content_count": 3
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | The generated summary text |
| `content_count` | integer | Number of content items processed |
| `combined_length` | integer | Total character count of all input content |
| `focus_areas` | array | Focus areas used in summarization |
| `instructions` | string | Instructions provided for summarization |

## Usage Examples

### Basic Multi-Document Summarization

```go
summaryAgent := summary.NewSummaryAgent(llmClient)

result, err := summaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "contents": []interface{}{
            "First research paper discusses machine learning advances...",
            "Second paper covers neural network architectures...",
            "Third paper presents experimental results...",
        },
    },
})

if result.Success {
    content := result.Content.(map[string]interface{})
    fmt.Printf("Summary: %s\n", content["summary"])
}
```

### API Usage with curl

#### Multi-Document Summarization
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "summarize_documents",
      "contents": [
        "Go is a programming language developed by Google.",
        "It features garbage collection and concurrent execution.",
        "Go is designed for building reliable and efficient software."
      ],
      "instructions": "Focus on key features and benefits",
      "focus_areas": ["features", "benefits"]
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "task": "summarize_documents",
    "result": {
      "summary": "Go is a programming language created by Google that features garbage collection, concurrent execution, and other capabilities designed for building reliable, efficient software with significant development benefits.",
      "metadata": {
        "content_count": 3,
        "combined_length": 142,
        "focus_areas": ["features", "benefits"],
        "instructions": "Focus on key features and benefits"
      }
    },
    "orchestrator": "primary_agent"
  }
}
```

#### Summarization with Custom Focus Areas
```bash
curl -X POST http://localhost:8080/agents/primary \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "task": "summarize_documents",
      "contents": [
        "Machine learning algorithms process data to find patterns.",
        "Neural networks consist of interconnected nodes called neurons.",
        "Deep learning uses multi-layer neural networks for complex tasks."
      ],
      "instructions": "Explain the core concepts and their relationships",
      "focus_areas": ["algorithms", "neural_networks", "applications"]
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "task": "summarize_documents",
    "result": {
      "summary": "Machine learning algorithms identify data patterns through various techniques. Neural networks, composed of interconnected neurons, form the foundation of many algorithms. Deep learning extends this concept with multi-layer networks capable of handling complex tasks and applications.",
      "metadata": {
        "content_count": 3,
        "combined_length": 156,
        "focus_areas": ["algorithms", "neural_networks", "applications"],
        "instructions": "Explain the core concepts and their relationships"
      }
    },
    "orchestrator": "primary_agent"
  }
}
```

### Summarization with Instructions

```go
result, err := summaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "contents": []interface{}{
            "Article about climate change impacts...",
            "Report on renewable energy solutions...",
        },
        "instructions": "Focus on practical solutions and policy recommendations",
    },
})
```

### Focused Summarization with Areas

```go
result, err := summaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "contents": []interface{}{
            "Technical documentation...",
            "User guide content...",
            "API reference...",
        },
        "instructions": "Create a user-friendly overview",
        "focus_areas": []interface{}{"key_features", "getting_started", "examples"},
    },
})
```

## Validation and Error Handling

### Input Validation Rules

**Required Fields:**
- `contents`: Must be present and be an array
- Array must contain at least one item
- Each item must be convertible to string

**Optional Fields:**
- `instructions`: Must be string if provided
- `focus_areas`: Must be array of strings if provided

### Common Validation Errors

**Missing Contents:**
```json
{
  "success": false,
  "error": "contents field is required",
  "duration": "0.001s"
}
```

**Empty Contents Array:**
```json
{
  "success": false,
  "error": "at least one content item is required",
  "duration": "0.001s"
}
```

**Invalid Contents Type:**
```json
{
  "success": false,
  "error": "contents must be an array",
  "duration": "0.001s"
}
```

**LLM Service Error:**
```json
{
  "success": false,
  "error": "LLM completion failed: connection timeout",
  "duration": "30.5s"
}
```

## Capabilities

The Summary Agent provides these capabilities:

```go
capabilities := summaryAgent.GetCapabilities()
// Returns:
[
    {
        "name": "content_summarization",
        "description": "Summarize multiple documents or text contents into coherent summaries"
    },
    {
        "name": "focused_summarization",
        "description": "Generate summaries with specific focus areas and instructions"
    }
]
```

## Prompt Construction

The agent builds comprehensive prompts that include:

1. **Content Section**: All provided content items clearly labeled
2. **Instructions**: Custom summarization guidelines when provided
3. **Focus Areas**: Specific areas to emphasize in the summary
4. **Output Guidelines**: Instructions for comprehensive yet concise summaries

### Example Generated Prompt
```
Please summarize the following content:

Content 1:
First document text about artificial intelligence...

Content 2:
Second document text about machine learning...

Instructions: Focus on practical applications and future trends

Focus areas: applications, trends, challenges

Provide a comprehensive summary that captures the key points and insights.
```

## Performance Characteristics

- **Latency**: 1-5 seconds depending on content length and complexity
- **Token Usage**: 200-800 tokens per summarization
- **Content Processing**: Up to 10,000 characters per content item
- **Batch Processing**: Efficiently handles multiple content items
- **Success Rate**: >98% for well-formed inputs

## Statistics and Monitoring

The Summary Agent tracks execution statistics:

```go
stats := summaryAgent.GetStats()
// Returns:
{
    "total_executions": 250,
    "average_duration": "2.1s",
    "success_rate": 0.98,
    "total_tokens": 75000
}
```

### Metrics Tracked
- **Total Executions**: Number of summarization tasks processed
- **Average Duration**: Mean processing time per task
- **Success Rate**: Percentage of successful summarizations
- **Token Usage**: Total tokens consumed by LLM calls

## Configuration

### LLM Integration
The agent requires an LLM provider implementing the `shared.LLMProvider` interface:

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    CountTokens(messages []Message) (int, error)
}
```

### Model Settings
- **Temperature**: 0.3 (for consistent, focused summaries)
- **Max Tokens**: 1,000 (allows comprehensive but concise summaries)
- **System Prompt**: Professional summarizer persona

### Context Parameters
Additional parameters can be passed via `input.Context`:
- `model`: LLM model to use (default: "gpt-4")
- `api_key`: API key for the LLM provider

## Best Practices

### Content Preparation
1. **Consistent Format**: Ensure similar document formats for best results
2. **Relevant Content**: Only include content relevant to the summarization goal
3. **Size Optimization**: Break very large documents into logical sections

### Instruction Design
1. **Clear Objectives**: Provide specific, actionable instructions
2. **Focus Areas**: Use focus areas to guide attention to key topics
3. **Output Format**: Specify desired summary structure when needed

### Performance Optimization
1. **Batch Processing**: Group related documents for efficient processing
2. **Content Limits**: Respect reasonable content size limits
3. **Caching**: Cache summaries for frequently processed content

## Implementation Details

### Execution Flow
1. **Input Validation**: Comprehensive validation of all input parameters
2. **Parameter Extraction**: Parse contents, instructions, and focus areas
3. **Prompt Construction**: Build structured summarization prompt
4. **LLM Execution**: Single API call to language model
5. **Result Processing**: Format and return structured summary
6. **Statistics Update**: Update execution metrics and counters

### Content Processing
- **Type Conversion**: Safely convert interface{} arrays to strings
- **Length Calculation**: Track combined content length for monitoring
- **Metadata Preservation**: Maintain content count and focus areas in results

### Error Recovery
- **Graceful Degradation**: Return meaningful errors for validation failures
- **Timeout Handling**: Respect context timeouts for long-running operations
- **Resource Cleanup**: Proper cleanup of resources and connections

## Integration with Primary Agent

The Summary Agent is designed to work seamlessly with the Primary Agent:

```go
// Primary agent automatically routes summarization tasks
primaryAgent := primary.NewPrimaryAgent(llmClient, summaryAgent)

result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "task": "summarize_documents",
        "contents": []interface{}{"doc1", "doc2"},
        "instructions": "Focus on key findings",
    },
})
```

## Future Enhancements

### Planned Features
- **Structured Output**: JSON-formatted summaries with sections
- **Summary Length Control**: Configurable summary lengths
- **Multi-Language Support**: Summarization in different languages
- **Domain-Specific Summaries**: Specialized prompts for technical domains
- **Summary Quality Metrics**: Automated quality assessment

### Advanced Capabilities
- **Comparative Summaries**: Summarize differences between documents
- **Timeline Summaries**: Extract and organize temporal information
- **Entity Recognition**: Identify and highlight key entities
- **Citation Tracking**: Preserve source citations in summaries
