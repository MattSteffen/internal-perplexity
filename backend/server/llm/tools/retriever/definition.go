package retriever

import (
	"context"
	"fmt"
	"log"

	"internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/services/embeddings"
	toolshared "internal-perplexity/server/llm/tools/shared"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

/*
import (
    "context"
    "fmt"

    "github.com/milvus-io/milvus/client/v2/column"
    "github.com/milvus-io/milvus/client/v2/entity"
    "github.com/milvus-io/milvus/client/v2/index"
)

ctx, cancel := context.WithCancel(context.Background())
defer cancel()

milvusAddr := "localhost:19530"
client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
    Address: milvusAddr,
})
if err != nil {
    fmt.Println(err.Error())
    // handle error
}
defer client.Close(ctx)

function := entity.NewFunction().
    WithName("text_bm25_emb").
    WithInputFields("text").
    WithOutputFields("text_sparse").
    WithType(entity.FunctionTypeBM25)

schema := entity.NewSchema()

schema.WithField(entity.NewField().
    WithName("id").
    WithDataType(entity.FieldTypeInt64).
    WithIsPrimaryKey(true),
).WithField(entity.NewField().
    WithName("text").
    WithDataType(entity.FieldTypeVarChar).
    WithEnableAnalyzer(true).
    WithMaxLength(1000),
).WithField(entity.NewField().
    WithName("text_dense").
    WithDataType(entity.FieldTypeFloatVector).
    WithDim(768),
).WithField(entity.NewField().
    WithName("text_sparse").
    WithDataType(entity.FieldTypeSparseVector),
).WithField(entity.NewField().
    WithName("image_dense").
    WithDataType(entity.FieldTypeFloatVector).
    WithDim(512),
).WithFunction(function)

*/
// MilvusQueryTool is a concrete implementation
type MilvusQueryTool struct {
	client            *milvusclient.Client
	embedder          embeddings.EmbeddingsRegistry
	config            *milvusclient.ClientConfig
	DefaultCollection string
	DefaultPartition  string
}

// NewMilvusQueryTool initializes the tool
func NewMilvusQueryTool(ctx context.Context, cfg *milvusclient.ClientConfig, embedder embeddings.EmbeddingsRegistry, defaultCollection string, defaultPartition string) *MilvusQueryTool {
	var client *milvusclient.Client
	if cfg != nil && cfg.Address != "" {
		c, err := milvusclient.New(ctx, cfg)
		if err != nil {
			log.Printf("Milvus client init failed, running retriever in mock mode: %v", err)
		} else {
			client = c
		}
	}
	return &MilvusQueryTool{
		client:            client,
		embedder:          embedder,
		config:            cfg,
		DefaultCollection: defaultCollection,
		DefaultPartition:  defaultPartition,
	}
}

// Name returns the tool name
func (m *MilvusQueryTool) Name() string {
	return "retriever"
}

// Description returns the tool description
func (m *MilvusQueryTool) Description() string {
	// TODO: construct description from the collections
	return "Queries Milvus vector database for semantic search and retrieval of documents"
}

// Schema returns the JSON schema for input validation
func (m *MilvusQueryTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"collection_name": map[string]any{
				"type":        "string",
				"description": "Name of the Milvus collection to query",
			},
			"partition_name": map[string]any{
				"type":        "string",
				"description": "Name of the partition to query (optional)",
			},
			"texts": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "List of text queries to search",
			},
			"filters": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "List of boolean expressions; joined with AND",
			},
			"top_k": map[string]any{
				"type":        "integer",
				"description": "Number of results to return",
				"default":     10,
				"minimum":     1,
				"maximum":     1000,
			},
		},
		"required": []string{"collection_name", "texts"},
	}
}

// Definition returns the OpenAI tool definition
func (m *MilvusQueryTool) Definition() *api.ToolDefinition {
	return toolshared.ToOpenAISchema(
		"retriever",
		"Perform semantic search and retrieval against a Milvus vector database. Use this tool when you need to find relevant documents by meaning rather than exact keywords. It supports hybrid queries with multiple text inputs, partition targeting, and configurable result limits. Returns documents with similarity scores and metadata for intelligent document retrieval and analysis.",
		m.Schema(),
	)
}

// Execute performs the Milvus query using the tool interface
func (m *MilvusQueryTool) Execute(ctx context.Context, input *toolshared.ToolInput) (*toolshared.ToolResult, error) {
	// Parse and validate input
	req, err := m.parseInput(input)
	if err != nil {
		return &toolshared.ToolResult{
			Success: false,
			Error:   err.Error(),
		}, nil
	}

	// Execute query
	response, err := m.HybridQuery(ctx, req)
	if err != nil {
		return &toolshared.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("query failed: %v", err),
		}, nil
	}

	// Convert results to tool result format
	results := m.convertResults(response)
	return &toolshared.ToolResult{
		Success: true,
		Data: map[string]any{
			"results": results,
			"count":   len(results),
		},
	}, nil
}
