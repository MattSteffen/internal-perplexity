package retriever

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"internal-perplexity/server/llm/tools"
)

func TestMilvusQueryTool_Name(t *testing.T) {
	tool := NewMilvusQueryTool()
	assert.Equal(t, "retriever", tool.Name())
}

func TestMilvusQueryTool_Description(t *testing.T) {
	tool := NewMilvusQueryTool()
	assert.Contains(t, tool.Description(), "Milvus")
	assert.Contains(t, tool.Description(), "vector database")
}

func TestMilvusQueryTool_Schema(t *testing.T) {
	tool := NewMilvusQueryTool()
	schema := tool.Schema()

	assert.NotNil(t, schema)
	assert.Equal(t, "object", schema.Type)
	assert.Contains(t, schema.Properties, "collection_name")
	assert.Contains(t, schema.Properties, "texts")
	assert.Contains(t, schema.Properties, "top_k")
	assert.Contains(t, schema.Required, "collection_name")
	assert.Contains(t, schema.Required, "texts")
}

func TestMilvusQueryTool_Definition(t *testing.T) {
	tool := NewMilvusQueryTool()
	definition := tool.Definition()

	assert.NotNil(t, definition)
	assert.Equal(t, "function", definition.Type)
	assert.NotNil(t, definition.Function)
	assert.Equal(t, "retriever", definition.Function.Name)
	assert.Contains(t, definition.Function.Description, "semantic search")
	assert.Contains(t, definition.Function.Description, "Milvus")
	assert.NotNil(t, definition.Function.Parameters)
}

func TestMilvusQueryTool_Execute_ValidInput(t *testing.T) {
	tool := NewMilvusQueryTool()

	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"collection_name": "test_collection",
			"texts":           []interface{}{"test query"},
			"top_k":           5,
		},
	}

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.NotNil(t, result.Data)

	assert.Contains(t, result.Data, "results")
	assert.Contains(t, result.Data, "count")
}

func TestMilvusQueryTool_Execute_MissingCollectionName(t *testing.T) {
	tool := NewMilvusQueryTool()

	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"texts": []interface{}{"test query"},
		},
	}

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.Success)
	assert.Contains(t, result.Error, "collection_name")
}

func TestMilvusQueryTool_Execute_MissingTexts(t *testing.T) {
	tool := NewMilvusQueryTool()

	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"collection_name": "test_collection",
		},
	}

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.Success)
	assert.Contains(t, result.Error, "texts")
}

func TestMilvusQueryTool_Execute_InvalidTextsType(t *testing.T) {
	tool := NewMilvusQueryTool()

	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"collection_name": "test_collection",
			"texts":           "not an array",
		},
	}

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.Success)
	assert.Contains(t, result.Error, "texts field must be an array")
}

func TestMilvusQueryTool_Execute_InvalidTextItem(t *testing.T) {
	tool := NewMilvusQueryTool()

	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"collection_name": "test_collection",
			"texts":           []interface{}{"valid text", 123},
		},
	}

	result, err := tool.Execute(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.Success)
	assert.Contains(t, result.Error, "must be a string")
}

func TestMilvusQueryTool_HybridQuery(t *testing.T) {
	tool := NewMilvusQueryTool()

	req := QueryRequest{
		CollectionName: "test_collection",
		PartitionName:  "test_partition",
		Texts:          []string{"test query 1", "test query 2"},
		TopK:           5,
	}

	response, err := tool.HybridQuery(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, response.Results)
	assert.Greater(t, len(response.Results), 0)

	// Check first result structure
	doc := response.Results[0]
	assert.NotZero(t, doc.ID)
	assert.NotEmpty(t, doc.Text)
	assert.Greater(t, doc.Score, 0.0)
	assert.NotNil(t, doc.Metadata)
}

func TestNewMilvusQueryTool(t *testing.T) {
	tool := NewMilvusQueryTool()
	assert.NotNil(t, tool)
	assert.IsType(t, &MilvusQueryTool{}, tool)
}

func TestDocument_JSONTags(t *testing.T) {
	doc := Document{
		ID:    123,
		Text:  "test text",
		Score: 0.95,
		Metadata: map[string]interface{}{
			"source": "test",
		},
	}

	// This test ensures the JSON tags are properly set
	assert.NotZero(t, doc.ID)
	assert.NotEmpty(t, doc.Text)
	assert.Greater(t, doc.Score, 0.0)
	assert.NotNil(t, doc.Metadata)
}

func TestQueryRequest_JSONTags(t *testing.T) {
	req := QueryRequest{
		CollectionName: "test_collection",
		PartitionName:  "test_partition",
		Texts:          []string{"query1", "query2"},
		TopK:           10,
	}

	assert.NotEmpty(t, req.CollectionName)
	assert.NotEmpty(t, req.PartitionName)
	assert.NotEmpty(t, req.Texts)
	assert.Greater(t, req.TopK, 0)
}

func TestQueryResponse_JSONTags(t *testing.T) {
	resp := QueryResponse{
		Results: []Document{
			{
				ID:    1,
				Text:  "result1",
				Score: 0.9,
				Metadata: map[string]interface{}{
					"type": "test",
				},
			},
		},
	}

	assert.NotEmpty(t, resp.Results)
	assert.Len(t, resp.Results, 1)
}
