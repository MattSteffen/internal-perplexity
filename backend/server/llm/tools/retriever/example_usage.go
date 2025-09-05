package retriever

import (
	"context"
	"fmt"
	"log"

	"internal-perplexity/server/llm/tools"
)

func main() {
	// Create tool registry
	registry := tools.NewRegistry()

	// Create and register the retriever tool
	retrieverTool := NewMilvusQueryTool()
	registry.Register(retrieverTool)

	// Example query
	input := &tools.ToolInput{
		Name: "retriever",
		Data: map[string]interface{}{
			"collection_name": "xmidas",    // Using the collection name from memory
			"partition_name":  "documents", // Optional partition
			"texts":           []interface{}{"artificial intelligence", "machine learning"},
			"top_k":           5,
		},
	}

	// Execute the query
	ctx := context.Background()
	result, err := registry.Execute(ctx, input)
	if err != nil {
		log.Fatalf("Failed to execute retriever: %v", err)
	}

	if result.Success {
		fmt.Println("Query successful!")
		fmt.Printf("Results: %+v\n", result.Data)
	} else {
		fmt.Printf("Query failed: %s\n", result.Error)
	}
}
