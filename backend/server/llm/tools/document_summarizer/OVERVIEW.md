# Document Summarizer Tool

The Document Summarizer tool provides intelligent text summarization capabilities using Large Language Models (LLMs). It can condense long documents into concise summaries while preserving key information and main ideas.

## Purpose

The Document Summarizer tool processes text content and generates focused summaries of specified lengths. It's designed for:
- **Content Condensation**: Reducing long documents to essential information
- **Key Point Extraction**: Identifying and preserving main ideas
- **Flexible Length Control**: Customizable summary lengths
- **Professional Summaries**: High-quality, coherent output using LLMs

## Features

- **LLM-Powered**: Uses advanced language models for intelligent summarization
- **Configurable Length**: Control summary length with `max_length` parameter
- **Single LLM Call**: Efficient processing with minimal API calls
- **Content Statistics**: Provides original and summary length metrics
- **Token Tracking**: Monitors token usage for cost optimization

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | The document text content to summarize |
| `max_length` | integer | No | Maximum length of summary in words (default: 200) |

### Content Requirements
- **Type**: Plain text string
- **Minimum length**: 1 character
- **Maximum length**: No hard limit, but longer content may affect processing time
- **Encoding**: UTF-8 compatible text

### Length Parameter
- **Default**: 200 words
- **Range**: 1-5000 words (recommended)
- **Behavior**: LLM will aim for the specified length but may vary slightly for coherence

## Output Schema

```json
{
  "success": true,
  "data": {
    "summary": "Concise summary of the input content...",
    "original_length": 1250,
    "summary_length": 180
  },
  "stats": {
    "tokens_used": 450
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | The generated summary text |
| `original_length` | integer | Character count of the original content |
| `summary_length` | integer | Character count of the generated summary |
| `tokens_used` | integer | Total tokens consumed by the LLM call |

## Usage Examples

### Basic Summarization

```go
registry := tools.NewRegistry()
summarizer := document_summarizer.NewDocumentSummarizer(llmClient)
registry.Register(summarizer)

result, err := registry.Execute(ctx, &tools.ToolInput{
    Name: "document_summarizer",
    Data: map[string]interface{}{
        "content": "Your long document text here...",
        "max_length": 100,
    },
})

if result.Success {
    data := result.Data.(map[string]interface{})
    fmt.Printf("Summary: %s\n", data["summary"])
}
```

### API Usage with curl

#### Basic Document Summarization
```bash
curl -X POST http://localhost:8080/tools/document_summarizer \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "content": "Go is a programming language developed by Google. It features garbage collection, concurrent programming support, and a rich standard library. Go is designed for building simple, reliable, and efficient software.",
      "max_length": 50
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "summary": "Go is a programming language created by Google, featuring garbage collection, concurrency support, and a rich standard library for building reliable, efficient software.",
    "original_length": 156,
    "summary_length": 89
  },
  "stats": {
    "execution_time": "2.5s",
    "tokens_used": 245
  }
}
```

#### Summarization with Custom Instructions
```bash
curl -X POST http://localhost:8080/tools/document_summarizer \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "content": "Machine learning is a subset of artificial intelligence that involves training algorithms to recognize patterns in data. There are several types of machine learning including supervised learning, unsupervised learning, and reinforcement learning.",
      "instructions": "Focus on practical applications and benefits",
      "max_length": 75
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "summary": "Machine learning enables algorithms to recognize data patterns for practical applications. Types include supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning, offering significant benefits in automation and decision-making.",
    "original_length": 234,
    "summary_length": 156
  },
  "stats": {
    "execution_time": "3.2s",
    "tokens_used": 320
  }
}
```

### Default Length Summarization

```go
input := &tools.ToolInput{
    Name: "document_summarizer",
    Data: map[string]interface{}{
        "content": "Document content without specifying max_length...",
    },
}
// Uses default max_length of 200 words
```

## Example Inputs and Outputs

### Example 1: Short Document Summary
**Input:**
```json
{
  "content": "The Go programming language was created by Google engineers Robert Griesemer, Rob Pike, and Ken Thompson. It was announced in November 2009 and version 1.0 was released in March 2012. Go is designed for building simple, reliable, and efficient software. It features garbage collection, concurrent programming support, and a rich standard library.",
  "max_length": 50
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "summary": "Go is a programming language created by Google engineers in 2009, with version 1.0 released in 2012. It's designed for building reliable, efficient software with features like garbage collection, concurrency support, and a rich standard library.",
    "original_length": 387,
    "summary_length": 156
  },
  "stats": {
    "tokens_used": 245
  }
}
```

### Example 2: Long Document with Custom Length
**Input:**
```json
{
  "content": "Machine learning is a subset of artificial intelligence that involves training algorithms to recognize patterns in data. There are several types of machine learning including supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex data patterns. Applications of machine learning include image recognition, natural language processing, recommendation systems, and autonomous vehicles.",
  "max_length": 75
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "summary": "Machine learning is a subset of AI that trains algorithms to recognize data patterns. Types include supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning. Deep learning uses multi-layer neural networks for complex patterns. Applications span image recognition, NLP, recommendations, and autonomous vehicles.",
    "original_length": 678,
    "summary_length": 234
  },
  "stats": {
    "tokens_used": 420
  }
}
```

## Error Handling

### Missing Content
**Input:**
```json
{
  "max_length": 100
}
```

**Output:**
```json
{
  "success": false,
  "error": "content field is required and must be a string"
}
```

### Invalid Content Type
**Input:**
```json
{
  "content": 12345,
  "max_length": 100
}
```

**Output:**
```json
{
  "success": false,
  "error": "content field is required and must be a string"
}
```

### LLM Service Error
**Input:**
```json
{
  "content": "Valid content...",
  "max_length": 100
}
```

**Output:**
```json
{
  "success": false,
  "error": "LLM completion failed: connection timeout"
}
```

## Configuration

### LLM Integration
The tool requires an LLM provider that implements the `shared.LLMProvider` interface:

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
    CountTokens(messages []Message) (int, error)
}
```

### Model Settings
- **Model**: Default GPT-4 (configurable)
- **Temperature**: 0.3 (for focused, consistent summaries)
- **Max Tokens**: Calculated as `max_length * 5` (rough token-to-word estimate)

## Best Practices

### Content Preparation
1. **Clean Text**: Remove unnecessary formatting, headers, footers
2. **Concatenate Sections**: Combine related content for coherent summaries
3. **Language Consistency**: Ensure consistent language throughout the document

### Length Selection
- **Short summaries (25-50 words)**: For quick overviews or abstracts
- **Medium summaries (100-200 words)**: For detailed summaries with key points
- **Long summaries (300+ words)**: For comprehensive coverage with examples

### Performance Optimization
- **Batch Processing**: Group similar documents for processing
- **Length Limits**: Set reasonable max_length to control costs
- **Caching**: Cache summaries for frequently accessed content

## Limitations

- **Single Language**: Optimized for English content
- **Text Only**: Cannot process images, tables, or structured data
- **Context Window**: Limited by LLM context window size
- **Deterministic Output**: Results may vary slightly between calls due to LLM nature
- **Cost**: Each summarization consumes LLM tokens

## Implementation Details

The summarization process:
1. **Input Validation**: Validates content and parameters
2. **Prompt Construction**: Creates focused summarization prompt
3. **LLM Call**: Single API call to language model
4. **Response Processing**: Extracts and formats summary
5. **Statistics**: Calculates lengths and token usage
6. **Result Packaging**: Returns structured response with metadata
