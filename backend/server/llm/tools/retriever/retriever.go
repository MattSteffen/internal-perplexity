package retriever

import (
	"context"
	"fmt"
	"sort"
	"strings"

	toolshared "internal-perplexity/server/llm/tools/shared"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// QueryRequest defines the input for a hybrid query
type QueryRequest struct {
	CollectionName string   `json:"collection_name"`
	PartitionName  string   `json:"partition_name"`
	Texts          []string `json:"texts"`
	Filters        []string `json:"filters"`
}

// QueryResponse wraps the results
type QueryResponse struct {
	Results []MilvusDocument `json:"results"`
}

const (
	denseField          = "text_embedding"
	sparseFieldText     = "text_sparse_embedding"
	sparseFieldMeta     = "metadata_sparse_embedding"
	defaultRRFK         = 100
	defaultNProbe       = 10
	defaultDropRatio    = 0.2
	defaultEmbedModel   = "nomic-embed-text"
	outputTextField     = "text" // optional if exists
	outputMetadataField = "metadata"
)

// parseInput validates and parses the tool input
func (m *MilvusQueryTool) parseInput(
	input *toolshared.ToolInput,
) (QueryRequest, error) {
	collectionName, ok := input.Data["collection_name"].(string)
	if !ok || strings.TrimSpace(collectionName) == "" {
		if m.DefaultCollection == "" {
			return QueryRequest{},
				fmt.Errorf(
					"collection_name field is required and must be a string",
				)
		}
		collectionName = m.DefaultCollection
	}

	textsRaw, ok := input.Data["texts"]
	if !ok {
		return QueryRequest{}, fmt.Errorf("texts field is required")
	}
	textsInterface, ok := textsRaw.([]interface{})
	if !ok {
		return QueryRequest{},
			fmt.Errorf("texts field must be an array of strings")
	}
	texts := make([]string, len(textsInterface))
	for i, v := range textsInterface {
		s, ok := v.(string)
		if !ok {
			return QueryRequest{}, fmt.Errorf("texts[%d] must be a string", i)
		}
		texts[i] = s
	}

	partitionName := m.DefaultPartition
	if partitionRaw, exists := input.Data["partition_name"]; exists {
		if partition, ok := partitionRaw.(string); ok {
			partitionName = partition
		}
	}

	var filters []string
	if fr, exists := input.Data["filters"]; exists {
		if arr, ok := fr.([]interface{}); ok {
			for i, v := range arr {
				s, ok := v.(string)
				if !ok {
					return QueryRequest{}, fmt.Errorf(
						"filters[%d] must be a string", i,
					)
				}
				if t := strings.TrimSpace(s); t != "" {
					filters = append(filters, t)
				}
			}
		} else {
			return QueryRequest{}, fmt.Errorf(
				"filters must be an array of strings",
			)
		}
	}

	return QueryRequest{
		CollectionName: collectionName,
		PartitionName:  partitionName,
		Texts:          texts,
		Filters:        filters,
	}, nil
}

// convertResults converts query response to tool result format
// TODO: fix this
func (m *MilvusQueryTool) convertResults(
	response QueryResponse,
) []map[string]interface{} {
	results := make([]map[string]interface{}, len(response.Results))
	for i, doc := range response.Results {
		results[i] = map[string]interface{}{
			"id":       doc.ID,
			"text":     doc.Text,
			"score":    doc.Distance,
			"metadata": doc.Metadata,
		}
	}
	return results
}

// HybridQuery executes a hybrid search against Milvus following the provided
// Python-style workflow:
// - For each query q:
//   - Dense: search on "text_embedding" with IVF nprobe
//   - Sparse: search on "text_sparse_embedding" with BM25 drop_ratio
//   - Sparse: search on "metadata_sparse_embedding" with BM25 drop_ratio
//
// - Apply filters (joined with AND) as expr on each request
// - Use RRFReranker to merge, then return topK results
func (m *MilvusQueryTool) HybridQuery(
	ctx context.Context,
	req QueryRequest,
) (QueryResponse, error) {
	if m.client == nil {
		return QueryResponse{}, fmt.Errorf("milvus client not initialized")
	}
	if len(req.Texts) == 0 {
		return QueryResponse{Results: []MilvusDocument{}}, nil
	}

	// Ensure collection is loaded
	if err := m.loadCollectionIfNeeded(ctx, req.CollectionName); err != nil {
		return QueryResponse{}, fmt.Errorf("load collection: %w", err)
	}

	// Determine embedding model from index if possible
	embedModel := ""
	var err error
	embedModel, err = m.embeddingModelFromIndex(
		ctx, req.CollectionName, denseField,
	)
	if strings.TrimSpace(embedModel) == "" {
		embedModel = defaultEmbedModel
	}

	// Pre-embed all texts for dense if available
	var denseVecs [][]float32
	denseVecs, err = m.embedTexts(ctx, embedModel, req.Texts)
	if err != nil {
		return QueryResponse{}, fmt.Errorf("embed texts: %w", err)
	}
	if len(denseVecs) != len(req.Texts) {
		return QueryResponse{}, fmt.Errorf(
			"embedding output mismatch: got %d vectors for %d texts",
			len(denseVecs), len(req.Texts),
		)
	}

	// Build all AnnSearchRequests
	var requests []*milvusclient.AnnRequest
	filterExpr := ""
	if len(req.Filters) > 0 {
		filterExpr = strings.Join(req.Filters, " and ")
	}

	for i, q := range req.Texts {
		dReq := milvusclient.NewAnnRequest(
			denseField,
			defaultRRFK,
			entity.FloatVector(denseVecs[i]),
		).WithAnnParam(index.NewIvfAnnParam(defaultNProbe))
		if filterExpr != "" {
			dReq = dReq.WithFilter(filterExpr)
		}
		requests = append(requests, dReq)

		sp := index.NewSparseAnnParam()
		sp.WithDropRatio(defaultDropRatio)
		sReq := milvusclient.NewAnnRequest(
			sparseFieldText,
			defaultRRFK,
			entity.Text(q),
		).WithAnnParam(sp)
		if filterExpr != "" {
			sReq = sReq.WithFilter(filterExpr)
		}
		requests = append(requests, sReq)

		sp = index.NewSparseAnnParam()
		sp.WithDropRatio(defaultDropRatio)
		mReq := milvusclient.NewAnnRequest(
			sparseFieldMeta,
			defaultRRFK,
			entity.Text(q),
		).WithAnnParam(sp)
		if filterExpr != "" {
			mReq = mReq.WithFilter(filterExpr)
		}
		requests = append(requests, mReq)
	}

	// If no valid requests (e.g., only filters were provided with no fields),
	// we could perform a pure boolean filter query. For now, return empty.
	if len(requests) == 0 {
		return QueryResponse{Results: []MilvusDocument{}}, nil
	}

	// RRF ranker
	reranker := milvusclient.NewRRFReranker().WithK(defaultRRFK)

	// Build HybridSearch option
	opt := milvusclient.NewHybridSearchOption(
		req.CollectionName, defaultRRFK, requests...,
	).WithReranker(reranker).
		WithOutputFields(outputTextField)
	if strings.TrimSpace(req.PartitionName) != "" {
		opt = opt.WithPartitions(req.PartitionName)
	}

	resultSets, err := m.client.HybridSearch(ctx, opt)
	if err != nil {
		return QueryResponse{}, fmt.Errorf("hybrid search failed: %w", err)
	}
	if len(resultSets) == 0 {
		return QueryResponse{Results: []MilvusDocument{}}, nil
	}

	// Aggregate results from all returned result sets
	agg := map[int64]MilvusDocument{}
	for _, rs := range resultSets {
		c := rs.GetColumn(outputTextField)
		ids, err := m.extractIDs(rs.IDs)
		if err != nil {
			return QueryResponse{}, fmt.Errorf("extract ids: %w", err)
		}
		scores := rs.Scores

		// TODO: fix this
		var textsByIndex []string
		if rs.Fields != nil {
			if col := rs.Fields[outputTextField]; col != nil {
				if vc, ok := col.(*entity.ColumnVarchar); ok {
					textsByIndex = vc.Data()
				}
			}
		}

		for i := range ids {
			id := ids[i]
			score := float64(scores[i])

			var textVal string
			if i < len(textsByIndex) {
				textVal = textsByIndex[i]
			}

			meta := map[string]any{
				"collection": req.CollectionName,
				"partition":  req.PartitionName,
				"filters":    req.Filters,
			}

			if prev, ok := agg[id]; !ok || score > prev.Score {
				agg[id] = Document{
					ID:       id,
					Text:     textVal,
					Score:    score,
					Metadata: meta,
				}
			}
		}
	}

	// Flatten, sort, truncate
	out := make([]Document, 0, len(agg))
	for _, d := range agg {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > req.TopK {
		out = out[:req.TopK]
	}

	return QueryResponse{Results: out}, nil
}

// loadCollectionIfNeeded loads the collection (best-effort).
func (m *MilvusQueryTool) loadCollectionIfNeeded(
	ctx context.Context,
	collection string,
) error {
	if m.client == nil {
		return fmt.Errorf("milvus client not initialized")
	}
	lt, err := m.client.LoadCollection(
		ctx,
		milvusclient.NewLoadCollectionOption(collection),
	)
	if err != nil {
		return err
	}

	return lt.Await(ctx)
}

// embeddingModelFromIndex attempts to infer the embedding model from index params.
func (m *MilvusQueryTool) embeddingModelFromIndex(
	ctx context.Context,
	collection, field string,
) (string, error) {
	idxDesc, err := m.client.DescribeIndex(
		ctx,
		milvusclient.NewDescribeIndexOption(collection).WithFieldName(field),
	)
	if err != nil {
		return "", err
	}
	if idxDesc == nil || len(idxDesc.Indexes) == 0 {
		return "", nil
	}
	keys := []string{"embedding_model", "model", "embedder"}
	for _, idx := range idxDesc.Indexes {
		for _, p := range idx.IndexParams {
			for _, k := range keys {
				if p.Key == k && strings.TrimSpace(p.Value) != "" {
					return p.Value, nil
				}
			}
		}
	}
	return "", nil
}

// embedTexts uses the embeddings registry to embed query texts with the model.
func (m *MilvusQueryTool) embedTexts(
	ctx context.Context,
	model string,
	texts []string,
) ([][]float32, error) {
	if m.embedder == nil {
		return nil, fmt.Errorf("embeddings registry not configured")
	}
	return m.embedder.Embed(ctx, model, texts)
}

/*
Multi-Vector Hybrid Search

In many applications, an object can be searched by a rich set of information such as title and description, or with multiple modalities such as text, images, and audio. For example, a tweet with a piece of text and an image shall be searched if either the text or the image matches the semantic of the search query. Hybrid search enhances search experience by combining searches across these diverse fields. Milvus supports this by allowing search on multiple vector fields, conducting several Approximate Nearest Neighbor (ANN) searches simultaneously. Multi-vector hybrid search is particularly useful if you want to search both text and images, multiple text fields that describe the same object, or dense and sparse vectors to improve search quality.

Hybrid Search Workflow
Hybrid Search Workflow

The multi-vector hybrid search integrates different search methods or spans embeddings from various modalities:

Sparse-Dense Vector Search: Dense Vector are excellent for capturing semantic relationships, while Sparse Vector are highly effective for precise keyword matching. Hybrid search combines these approaches to provide both a broad conceptual understanding and exact term relevance, thus improving search results. By leveraging the strengths of each method, hybrid search overcomes the limitations of indiviual approaches, offering better performance for complex queries. Here is more detailed guide on hybrid retrieval that combines semantic search with full-text search.
Multimodal Vector Search: Multimodal vector search is a powerful technique that allows you to search across various data types, including text, images, audio, and others. The main advantage of this approach is its ability to unify different modalities into a seamless and cohesive search experience. For instance, in product search, a user might input a text query to find products described with both text and images. By combining these modalities through a hybrid search method, you can enhance search accuracy or enrich the search results.
Example

Let’s consider a real world use case where each product includes a text description and an image. Based on the available data, we can conduct three types of searches:

Semantic Text Search: This involves querying the text description of the product using dense vectors. Text embeddings can be generated using models such as BERT and Transformers or services like OpenAI.
Full-Text Search: Here, we query the text description of the product using a keyword match with sparse vectors. Algorithms like BM25 or sparse embedding models such as BGE-M3 or SPLADE can be utilized for this purpose.
Multimodal Image Search: This method queries over the image using a text query with dense vectors. Image embeddings can be generated with models like CLIP.
This guide will walk you through an example of a multimodal hybrid search combining the above search methods, given the raw text description and image embeddings of products. We will demonstrate how to store multi-vector data and perform hybrid searches with a reranking strategy.

Create a collection with multiple vector fields

The process of creating a collection involves three key steps: defining the collection schema, configuring the index parameters, and creating the collection.

Define schema
For multi-vector hybrid search, we should define multiple vector fields within a collection schema. By default, each collection can accommodate up to 4 vector fields. However, if necessary, you can adjust the proxy.maxVectorFieldNum to include up to 10 vector fields in a collection as needed.

This example incorporates the following fields into the schema:

id: Serves as the primary key for storing text IDs. This field is of data type INT64.
text: Used for storing textual content. This field is of the data type VARCHAR with a maximum length of 1000 bytes. The enable_analyzer option is set to True to facilitate full-text search.
text_dense: Used to store dense vectors of the texts. This field is of the data type FLOAT_VECTOR with a vector dimension of 768.
text_sparse: Used to store sparse vectors of the texts. This field is of the data type SPARSE_FLOAT_VECTOR.
image_dense: Used to store dense vectors of the product images. This field is of the data type FLOAT_VETOR with a vector dimension of 512.
Since we will use the built-in BM25 algorithm to perform a full-text search on the text field, it is necessary to add the Milvus Function to the schema. For further details, please refer to Full Text Search.

import (
    "context"
    "fmt"

    "github.com/milvus-io/milvus/client/v2/column"
    "github.com/milvus-io/milvus/client/v2/entity"
    "github.com/milvus-io/milvus/client/v2/index"
    "github.com/milvus-io/milvus/client/v2/milvusclient"
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

Create index
indexOption1 := milvusclient.NewCreateIndexOption("my_collection", "text_dense",
    index.NewAutoIndex(index.MetricType(entity.IP)))
indexOption2 := milvusclient.NewCreateIndexOption("my_collection", "text_sparse",
    index.NewSparseInvertedIndex(entity.BM25, 0.2))
indexOption3 := milvusclient.NewCreateIndexOption("my_collection", "image_dense",
    index.NewAutoIndex(index.MetricType(entity.IP)))
)

Create collection
Create a collection named demo with the collection schema and indexes configured in the previous two steps.

err = client.CreateCollection(ctx,
    milvusclient.NewCreateCollectionOption("my_collection", schema).
        WithIndexOptions(indexOption1, indexOption2))
if err != nil {
    fmt.Println(err.Error())
    // handle error
}

Insert data

This section inserts data into the my_collection collection based on the schema defined earlier. During insert, ensure all fields, except those with auto-generated values, are provided with data in the correct format. In this example:

id: an integer representing the product ID
text: a string containing the product description
text_dense: a list of 768 floating-point values representing the dense embedding of the text description
image_dense: a list of 512 floating-point values representing the dense embedding of the product image
You may use the same or different models to generate dense embeddings for each field. In this example, the two dense embeddings have different dimensions, suggesting they were generated by different models. When defining each search later, be sure to use the corresponding model to generate the appropriate query embedding.

Since this example uses the built-in BM25 function to generate sparse embeddings from the text field, you do not need to supply sparse vectors manually. However, if you opt not to use BM25, you must precompute and provide the sparse embeddings yourself.

_, err = client.Insert(ctx, milvusclient.NewColumnBasedInsertOption("my_collection").
    WithInt64Column("id", []int64{0, 1, 2}).
    WithVarcharColumn("text", []string{
        "Red cotton t-shirt with round neck",
        "Wireless noise-cancelling over-ear headphones",
        "Stainless steel water bottle, 500ml",
    }).
    WithFloatVectorColumn("text_dense", 768, [][]float32{
        {0.3580376395471989, -0.6023495712049978, 0.18414012509913835, ...},
        {0.19886812562848388, 0.06023560599112088, 0.6976963061752597, ...},
        {0.43742130801983836, -0.5597502546264526, 0.6457887650909682, ...},
    }).
    WithFloatVectorColumn("image_dense", 512, [][]float32{
        {0.6366019600530924, -0.09323198122475052, ...},
        {0.6414180010301553, 0.8976979978567611, ...},
        {-0.6901259768402174, 0.6100500332193755, ...},
    }).
if err != nil {
    fmt.Println(err.Error())
    // handle err
}

Perform Hybrid Search

Create multiple AnnSearchRequest instances
Hybrid Search is implemented by creating multiple AnnSearchRequest in the hybrid_search() function, where each AnnSearchRequest represents a basic ANN search request for a specific vector field. Therefore, before conducting a Hybrid Search, it is necessary to create an AnnSearchRequest for each vector field.

In addition, by configuring the expr parameter in an AnnSearchRequest, you can set the filtering conditions for your hybrid search. Please refer to Filtered Search and Filtering.

In Hybrid Search, each AnnSearchRequest supports only one query data.
To demonstrate the capabilities of various search vector fields, we will construct three AnnSearchRequest search requests using a sample query. We will also use its pre-computed dense vectors for this process. The search requests will target the following vector fields:

text_dense for semantic text search, allowing for contextual understanding and retrieval based on meaning rather than direct keyword matching.
text_sparsefor full-text search or keyword matching, focusing on exact word or phrase matches within the text.
image_densefor multimodal text-to-image search, to retrieve relevant product images based on the semantic content of the query.
queryText := entity.Text({"white headphones, quiet and comfortable"})
queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, ...}
queryMultimodalVector := []float32{0.015829865178701663, 0.5264158340734488, ...}

request1 := milvusclient.NewAnnRequest("text_dense", 2, entity.FloatVector(queryVector)).
    WithAnnParam(index.NewIvfAnnParam(10))

annParam := index.NewSparseAnnParam()
annParam.WithDropRatio(0.2)
request2 := milvusclient.NewAnnRequest("text_sparse", 2, queryText).
    WithAnnParam(annParam)

request3 := milvusclient.NewAnnRequest("image_dense", 2, entity.FloatVector(queryMultimodalVector)).
    WithAnnParam(index.NewIvfAnnParam(10))

Given that the parameter limit is set to 2, each AnnSearchRequest returns 2 search results. In this example, 3 AnnSearchRequest instances are created, resulting in a total of 6 search results.

Configure a reranking strategy
To merge and rerank the sets of ANN search results, selecting an appropriate reranking strategy is essential. Milvus offers two types of reranking strategies:

WeightedRanker: Use this strategy if the results need to emphasize a particular vector field. WeightedRanker allows you to assign greater weight to certain vector fields, highlighting them more prominently.
RRFRanker (Reciprocal Rank Fusion Ranker): Choose this strategy when no specific emphasis is required. RRFRanker effectively balances the importance of each vector field.
For more details about the mechanisms of these two reranking strategies, refer to Reranking.

In this example, since there is no particular emphasis on specific search queries, we will proceed with the RRFRanker strategy.

reranker := milvusclient.NewRRFReranker().WithK(100)

Perform a Hybrid Search
Before initiating a Hybrid Search, ensure that the collection is loaded. If any vector fields within the collection lack an index or are not loaded into memory, an error will occur upon executing the Hybrid Search method.

resultSets, err := client.HybridSearch(ctx, milvusclient.NewHybridSearchOption(
    "my_collection",
    2,
    request1,
    request2,
    request3,
).WithReranker(reranker))
if err != nil {
    fmt.Println(err.Error())
    // handle error
}

for _, resultSet := range resultSets {
    fmt.Println("IDs: ", resultSet.IDs.FieldData().GetScalars())
    fmt.Println("Scores: ", resultSet.Scores)
}

The following is the output:

With the limit=2 parameter specified for the Hybrid Search, Milvus will rerank the six results obtained from the three searches. Ultimately, they will return only the top two most similar results.
*/
