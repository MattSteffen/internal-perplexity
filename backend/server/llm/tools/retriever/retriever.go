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

// parseInput validates and parses the tool input
func (m *MilvusQueryTool) parseInput(input *tools.ToolInput) (QueryRequest, error) {
	collectionName, ok := input.Data["collection_name"].(string)
	if !ok {
		return QueryRequest{}, fmt.Errorf("collection_name field is required and must be a string")
	}

	textsRaw, ok := input.Data["texts"]
	if !ok {
		return QueryRequest{}, fmt.Errorf("texts field is required")
	}

	textsInterface, ok := textsRaw.([]interface{})
	if !ok {
		return QueryRequest{}, fmt.Errorf("texts field must be an array of strings")
	}

	texts := make([]string, len(textsInterface))
	for i, textInterface := range textsInterface {
		text, ok := textInterface.(string)
		if !ok {
			return QueryRequest{}, fmt.Errorf("texts[%d] must be a string", i)
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

	return QueryRequest{
		CollectionName: collectionName,
		PartitionName:  partitionName,
		Texts:          texts,
		TopK:           topK,
	}, nil
}

// convertResults converts query response to tool result format
func (m *MilvusQueryTool) convertResults(response QueryResponse) []map[string]interface{} {
	results := make([]map[string]interface{}, len(response.Results))
	for i, doc := range response.Results {
		results[i] = map[string]interface{}{
			"id":       doc.ID,
			"text":     doc.Text,
			"score":    doc.Score,
			"metadata": doc.Metadata,
		}
	}
	return results
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
