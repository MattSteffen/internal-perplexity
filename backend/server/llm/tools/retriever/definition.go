package retriever

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/tools"
)

// MilvusQueryTool is a concrete implementation
type MilvusQueryTool struct {
	// TODO: Add Milvus client, embedding model, etc.
}

// NewMilvusQueryTool initializes the tool
func NewMilvusQueryTool( /* client config */ ) *MilvusQueryTool {
	return &MilvusQueryTool{
		// TODO: init client
	}
}

// Name returns the tool name
func (m *MilvusQueryTool) Name() string {
	return "retriever"
}

// Description returns the tool description
func (m *MilvusQueryTool) Description() string {
	return "Queries Milvus vector database for semantic search and retrieval of documents"
}

// Schema returns the JSON schema for input validation
func (m *MilvusQueryTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]interface{}{
			"collection_name": map[string]interface{}{
				"type":        "string",
				"description": "Name of the Milvus collection to query",
			},
			"partition_name": map[string]interface{}{
				"type":        "string",
				"description": "Name of the partition to query (optional)",
			},
			"texts": map[string]interface{}{
				"type":        "array",
				"items":       map[string]interface{}{"type": "string"},
				"description": "List of text queries to embed and search",
			},
			"top_k": map[string]interface{}{
				"type":        "integer",
				"description": "Number of top results to return",
				"default":     10,
				"minimum":     1,
				"maximum":     100,
			},
		},
		"required": []string{"collection_name", "texts"},
	}
}

// Definition returns the OpenAI tool definition
func (m *MilvusQueryTool) Definition() *api.ToolDefinition {
	return &api.ToolDefinition{
		Type: "function",
		Function: api.FunctionDefinition{
			Name:        "retriever",
			Description: "Perform semantic search and retrieval against a Milvus vector database. Use this tool when you need to find relevant documents by meaning rather than exact keywords. It supports hybrid queries with multiple text inputs, partition targeting, and configurable result limits. Returns documents with similarity scores and metadata for intelligent document retrieval and analysis.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"collection_name": map[string]interface{}{
						"type":        "string",
						"description": "The name of the Milvus collection to search in. This specifies which dataset to query.",
					},
					"partition_name": map[string]interface{}{
						"type":        "string",
						"description": "Optional partition name within the collection. If not specified, searches all partitions in the collection.",
					},
					"texts": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "string",
						},
						"description": "Array of text queries to search for. Each text will be embedded and used for semantic similarity search.",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"description": "Number of top results to return, ordered by relevance score (1-100, default: 10)",
						"default":     10,
						"minimum":     1,
						"maximum":     100,
					},
				},
				"required": []string{"collection_name", "texts"},
			},
		},
	}
}

// Execute performs the Milvus query using the tool interface
func (m *MilvusQueryTool) Execute(ctx context.Context, input *tools.ToolInput) (*tools.ToolResult, error) {
	// Parse and validate input
	req, err := m.parseInput(input)
	if err != nil {
		return &tools.ToolResult{
			Success: false,
			Error:   err.Error(),
		}, nil
	}

	// Execute query
	response, err := m.HybridQuery(ctx, req)
	if err != nil {
		return &tools.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("query failed: %v", err),
		}, nil
	}

	// Convert results to tool result format
	results := m.convertResults(response)
	return &tools.ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"results": results,
			"count":   len(results),
		},
	}, nil
}
