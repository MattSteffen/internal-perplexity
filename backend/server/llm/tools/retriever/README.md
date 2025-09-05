# Retriever Tool

The Retriever tool provides semantic search and retrieval capabilities against a Milvus vector database. It supports hybrid queries combining text embedding and vector similarity search for intelligent document retrieval.

## Purpose

The Retriever tool enables efficient querying of vectorized documents stored in Milvus. It's designed for:
- **Semantic Search**: Finding documents by meaning, not just keywords
- **Hybrid Queries**: Combining multiple text queries in a single search
- **Partition Targeting**: Querying specific document partitions
- **Relevance Ranking**: Returning results ordered by similarity scores
- **Metadata Preservation**: Maintaining document metadata in results

## Features

- **Vector Search**: Semantic similarity search using embeddings
- **Multi-Query Support**: Process multiple text queries simultaneously
- **Partition Filtering**: Target specific partitions within collections
- **Configurable Results**: Control result count with `top_k` parameter
- **Structured Output**: Consistent result format with scores and metadata
- **Tool Integration**: Seamless integration with the tool registry system

## Usage

### Basic Query

```go
registry := tools.NewRegistry()
retrieverTool := retriever.NewMilvusQueryTool()
registry.Register(retrieverTool)

input := &tools.ToolInput{
    Name: "retriever",
    Data: map[string]interface{}{
        "collection_name": "xmidas",
        "texts":           []interface{}{"your search query"},
        "top_k":           10,
    },
}

result, err := registry.Execute(ctx, input)
```

### Query with Partition

```go
input := &tools.ToolInput{
    Name: "retriever",
    Data: map[string]interface{}{
        "collection_name": "xmidas",
        "partition_name":  "documents",
        "texts":           []interface{}{"multiple", "queries", "supported"},
        "top_k":           5,
    },
}
```

### Multiple Text Queries

```go
input := &tools.ToolInput{
    Name: "retriever",
    Data: map[string]interface{}{
        "collection_name": "xmidas",
        "texts": []interface{}{
            "artificial intelligence",
            "machine learning algorithms",
            "neural networks"
        },
        "top_k": 20,
    },
}
```

### API Usage with curl

#### Single Query Search
```bash
curl -X POST http://localhost:8080/tools/retriever \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "collection_name": "xmidas",
      "texts": ["artificial intelligence"],
      "top_k": 5
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "results": [
      {
        "id": 1001,
        "text": "Artificial Intelligence (AI) is a field of computer science...",
        "score": 0.92,
        "metadata": {
          "source": "wikipedia",
          "category": "technology"
        }
      }
    ],
    "count": 1
  },
  "stats": {
    "execution_time": "1.2s"
  }
}
```

#### Multiple Queries with Partition
```bash
curl -X POST http://localhost:8080/tools/retriever \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "collection_name": "xmidas",
      "partition_name": "documents",
      "texts": [
        "machine learning algorithms",
        "neural network architecture"
      ],
      "top_k": 10
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "results": [
      {
        "id": 2001,
        "text": "Machine learning algorithms process data to find patterns...",
        "score": 0.89,
        "metadata": {
          "source": "research_paper",
          "author": "Dr. Smith"
        }
      },
      {
        "id": 2002,
        "text": "Neural networks consist of interconnected nodes...",
        "score": 0.87,
        "metadata": {
          "source": "tutorial",
          "difficulty": "intermediate"
        }
      }
    ],
    "count": 2
  },
  "stats": {
    "execution_time": "2.1s"
  }
}
```

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `collection_name` | string | Yes | Name of the Milvus collection to query |
| `partition_name` | string | No | Name of the partition to query (searches all partitions if omitted) |
| `texts` | array | Yes | Array of text strings to embed and search |
| `top_k` | integer | No | Number of top results to return (default: 10, max: 100) |

## Output Schema

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": 123,
        "text": "Retrieved document text content...",
        "score": 0.95,
        "metadata": {
          "source": "document_source",
          "timestamp": "2024-01-01T00:00:00Z",
          "author": "John Doe"
        }
      }
    ],
    "count": 1
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Array of retrieved documents |
| `results[].id` | integer | Unique document identifier from Milvus |
| `results[].text` | string | The retrieved document text content |
| `results[].score` | number | Similarity score (higher = more relevant) |
| `results[].metadata` | object | Document metadata (source, timestamp, etc.) |
| `count` | integer | Total number of results returned |

## Example Inputs and Outputs

### Example 1: Single Query
**Input:**
```json
{
  "collection_name": "xmidas",
  "texts": ["artificial intelligence"],
  "top_k": 5
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": 1001,
        "text": "Artificial Intelligence (AI) is a field of computer science...",
        "score": 0.92,
        "metadata": {
          "source": "wikipedia",
          "category": "technology"
        }
      }
    ],
    "count": 1
  }
}
```

### Example 2: Multiple Queries
**Input:**
```json
{
  "collection_name": "xmidas",
  "partition_name": "documents",
  "texts": [
    "machine learning algorithms",
    "neural network architecture"
  ],
  "top_k": 10
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": 2001,
        "text": "Machine learning algorithms process data to find patterns...",
        "score": 0.89,
        "metadata": {
          "source": "research_paper",
          "author": "Dr. Smith"
        }
      },
      {
        "id": 2002,
        "text": "Neural networks consist of interconnected nodes...",
        "score": 0.87,
        "metadata": {
          "source": "tutorial",
          "difficulty": "intermediate"
        }
      }
    ],
    "count": 2
  }
}
```

## Error Handling

### Missing Collection Name
**Input:**
```json
{
  "texts": ["query"],
  "top_k": 5
}
```

**Output:**
```json
{
  "success": false,
  "error": "collection_name field is required and must be a string"
}
```

### Invalid Texts Array
**Input:**
```json
{
  "collection_name": "xmidas",
  "texts": "not an array"
}
```

**Output:**
```json
{
  "success": false,
  "error": "texts field must be an array of strings"
}
```

### Invalid Text Item
**Input:**
```json
{
  "collection_name": "xmidas",
  "texts": ["valid query", 123]
}
```

**Output:**
```json
{
  "success": false,
  "error": "texts[1] must be a string"
}
```

### Database Connection Error
**Input:**
```json
{
  "collection_name": "nonexistent",
  "texts": ["query"]
}
```

**Output:**
```json
{
  "success": false,
  "error": "query failed: collection not found"
}
```

## Configuration

### Milvus Integration
The tool requires a Milvus client connection (to be implemented):

```go
// TODO: Implement Milvus client setup
type MilvusConfig struct {
    Address  string
    Username string
    Password string
    Database string
}

func NewMilvusQueryTool(config *MilvusConfig) *MilvusQueryTool {
    // Initialize Milvus client
    // Setup embedding model
    return &MilvusQueryTool{
        client: client,
        embedder: embedder,
    }
}
```

### Embedding Model
The tool will use text embeddings for semantic search (to be implemented):
- **Model**: Configurable embedding model (e.g., text-embedding-ada-002)
- **Dimensions**: Must match the vector dimensions in Milvus collections
- **Preprocessing**: Text cleaning and normalization before embedding

## Best Practices

### Query Optimization
1. **Specific Queries**: Use descriptive, specific search terms
2. **Multiple Queries**: Combine related queries for better results
3. **Partition Targeting**: Use partitions to narrow search scope
4. **Result Limits**: Set appropriate `top_k` based on use case

### Collection Design
- **Consistent Embeddings**: Use the same embedding model for indexing and querying
- **Metadata Rich**: Include relevant metadata for filtering and context
- **Partition Strategy**: Organize documents by type, date, or domain

### Performance Optimization
- **Batch Queries**: Process multiple queries efficiently
- **Caching**: Cache frequently accessed embeddings
- **Connection Pooling**: Reuse Milvus connections
- **Async Processing**: Handle large result sets asynchronously

## Limitations

- **Mock Implementation**: Currently returns mock data (full implementation pending)
- **Single Collection**: Queries one collection at a time
- **Text Only**: Cannot search images, audio, or other media types
- **Embedding Consistency**: Requires matching embedding models between indexing and querying
- **Memory Usage**: Large result sets may consume significant memory

## Implementation Details

### Query Processing Flow
1. **Input Validation**: Validate collection, partition, texts, and parameters
2. **Text Embedding**: Convert query texts to vector embeddings
3. **Milvus Query**: Execute vector similarity search
4. **Result Processing**: Map Milvus results to structured format
5. **Metadata Preservation**: Include document metadata in results
6. **Response Formatting**: Return consistent JSON structure

### Data Structures

```go
// Core types used by the retriever
type Document struct {
    ID       int64                  `json:"id"`
    Text     string                 `json:"text"`
    Score    float64                `json:"score"`
    Metadata map[string]interface{} `json:"metadata"`
}

type QueryRequest struct {
    CollectionName string   `json:"collection_name"`
    PartitionName  string   `json:"partition_name"`
    Texts          []string `json:"texts"`
    TopK           int      `json:"top_k"`
}

type QueryResponse struct {
    Results []Document `json:"results"`
}
```

## Implementation Status

### Current Implementation
- ✅ Tool interface integration
- ✅ Input validation and schema
- ✅ Mock query implementation
- ✅ Comprehensive test coverage
- ✅ Tool registry integration

### TODO - Full Implementation
- 🔄 Milvus client connection setup
- 🔄 Text embedding integration
- 🔄 Vector search query execution
- 🔄 Result mapping and formatting
- 🔄 Error handling for connection issues
- 🔄 Configuration management

## Dependencies

The tool uses the following Go modules:
- `github.com/milvus-io/milvus-sdk-go/v2` - Milvus Go SDK
- `github.com/tmc/langchaingo/embeddings` - Text embeddings

## Testing

Run the tests with:
```bash
go test ./llm/tools/retriever -v
```

Tests cover:
- Tool interface compliance
- Input validation
- Schema validation
- Error handling
- JSON serialization
