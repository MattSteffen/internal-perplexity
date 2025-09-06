package retriever

// This file previously contained example usage code that has been removed
// to eliminate the unused main function linting error.
//
// Example usage:
//   registry := tools.NewRegistry()
//   retrieverTool := NewMilvusQueryTool()
//   registry.Register(retrieverTool)
//
//   input := &tools.ToolInput{
//       Name: "retriever",
//       Data: map[string]interface{}{
//           "collection_name": "xmidas",
//           "partition_name":  "documents",
//           "texts":           []interface{}{"artificial intelligence", "machine learning"},
//           "top_k":           5,
//       },
//   }
//
//   result, err := registry.Execute(ctx, input)
