package retriever

import (
	"context"
	"fmt"

	"internal-perplexity/server/llm/tools"
)

// Document represents a single result from Milvus
type Document struct {
	ID       int64                  `json:"id"`
	Text     string                 `json:"text"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
}

// QueryRequest defines the input for a hybrid query
type QueryRequest struct {
	CollectionName string   `json:"collection_name"`
	PartitionName  string   `json:"partition_name"`
	Texts          []string `json:"texts"`
	TopK           int      `json:"top_k"`
}

// QueryResponse wraps the results
type QueryResponse struct {
	Results []Document `json:"results"`
}

// QueryTool defines the interface for querying Milvus
type QueryTool interface {
	HybridQuery(ctx context.Context, req QueryRequest) (QueryResponse, error)
}

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
	return "Queries Milvus vector database for semantic search and retrieval"
}

// Schema returns the JSON schema for input validation
func (m *MilvusQueryTool) Schema() *tools.ToolSchema {
	return &tools.ToolSchema{
		Type: "object",
		Properties: map[string]interface{}{
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
		Required: []string{"collection_name", "texts"},
	}
}

// Execute performs the Milvus query using the tool interface
func (m *MilvusQueryTool) Execute(ctx context.Context, input *tools.ToolInput) (*tools.ToolResult, error) {
	// Parse input data
	collectionName, ok := input.Data["collection_name"].(string)
	if !ok {
		return &tools.ToolResult{
			Success: false,
			Error:   "collection_name field is required and must be a string",
		}, nil
	}

	textsRaw, ok := input.Data["texts"]
	if !ok {
		return &tools.ToolResult{
			Success: false,
			Error:   "texts field is required",
		}, nil
	}

	textsInterface, ok := textsRaw.([]interface{})
	if !ok {
		return &tools.ToolResult{
			Success: false,
			Error:   "texts field must be an array of strings",
		}, nil
	}

	texts := make([]string, len(textsInterface))
	for i, textInterface := range textsInterface {
		text, ok := textInterface.(string)
		if !ok {
			return &tools.ToolResult{
				Success: false,
				Error:   fmt.Sprintf("texts[%d] must be a string", i),
			}, nil
		}
		texts[i] = text
	}

	partitionName := ""
	if partitionRaw, exists := input.Data["partition_name"]; exists {
		if partition, ok := partitionRaw.(string); ok {
			partitionName = partition
		}
	}

	topK := 10 // default
	if topKRaw, exists := input.Data["top_k"]; exists {
		if topKFloat, ok := topKRaw.(float64); ok {
			topK = int(topKFloat)
		}
	}

	// Build query request
	req := QueryRequest{
		CollectionName: collectionName,
		PartitionName:  partitionName,
		Texts:          texts,
		TopK:           topK,
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
	results := make([]map[string]interface{}, len(response.Results))
	for i, doc := range response.Results {
		results[i] = map[string]interface{}{
			"id":       doc.ID,
			"text":     doc.Text,
			"score":    doc.Score,
			"metadata": doc.Metadata,
		}
	}

	return &tools.ToolResult{
		Success: true,
		Data: map[string]interface{}{
			"results": results,
			"count":   len(results),
		},
	}, nil
}

// HybridQuery executes a hybrid search against Milvus
func (m *MilvusQueryTool) HybridQuery(
	ctx context.Context,
	req QueryRequest,
) (QueryResponse, error) {
	// TODO: Implement actual Milvus query logic
	// 1. Embed input texts using embedding model
	// 2. Build Milvus search request with vectors
	// 3. Execute query against specified collection/partition
	// 4. Map results into []Document

	// For now, return mock data
	return QueryResponse{
		Results: []Document{
			{
				ID:    123,
				Text:  "This is a sample document retrieved from Milvus",
				Score: 0.98,
				Metadata: map[string]interface{}{
					"source":      "mock_data",
					"collection":  req.CollectionName,
					"partition":   req.PartitionName,
					"query_texts": req.Texts,
				},
			},
		},
	}, nil
}
