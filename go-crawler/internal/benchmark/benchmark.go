package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// BenchmarkClient implements the BenchmarkClient interface for performance evaluation
type BenchmarkClient struct {
	dbConfig    *config.DatabaseConfig
	embedConfig *config.EmbedderConfig
	embedder    interfaces.Embedder
	database    interfaces.DatabaseClient
}

// NewBenchmarkClient creates a new benchmark client
func NewBenchmarkClient(dbConfig *config.DatabaseConfig, embedConfig *config.EmbedderConfig, embedder interfaces.Embedder, database interfaces.DatabaseClient) *BenchmarkClient {
	return &BenchmarkClient{
		dbConfig:    dbConfig,
		embedConfig: embedConfig,
		embedder:    embedder,
		database:    database,
	}
}

// RunBenchmark executes comprehensive benchmark tests
func (b *BenchmarkClient) RunBenchmark(ctx context.Context, generateQueries bool) (*interfaces.BenchmarkResults, error) {
	results := &interfaces.BenchmarkResults{
		ResultsByDoc:           make(map[string][]interfaces.BenchmarkResult),
		PlacementDistribution:  make(map[int]int),
		DistanceDistribution:   []float64{},
		PercentInTopK:          make(map[int]float64),
		SearchTimeDistribution: []time.Duration{},
	}

	// Generate benchmark queries from existing documents
	queries, err := b.generateBenchmarkQueries(ctx, generateQueries)
	if err != nil {
		return nil, fmt.Errorf("failed to generate benchmark queries: %v", err)
	}

	if len(queries) == 0 {
		return results, fmt.Errorf("no benchmark queries generated")
	}

	// Run searches for each query
	for _, query := range queries {
		result, err := b.runSingleQuery(ctx, query)
		if err != nil {
			fmt.Printf("Warning: failed to run query '%s': %v\n", query.Query, err)
			continue
		}

		// Store result
		docID := result.ExpectedSource
		results.ResultsByDoc[docID] = append(results.ResultsByDoc[docID], result)

		// Update distributions
		if result.PlacementOrder != nil {
			results.PlacementDistribution[*result.PlacementOrder]++
		}

		if result.Distance != nil {
			results.DistanceDistribution = append(results.DistanceDistribution, *result.Distance)
		}

		results.SearchTimeDistribution = append(results.SearchTimeDistribution, result.TimeToSearch)
	}

	// Calculate top-K accuracy
	results.PercentInTopK = b.calculateTopKAccuracy(results)

	return results, nil
}

// Search performs a search with timing and result analysis
func (b *BenchmarkClient) Search(ctx context.Context, queries []string) ([]interfaces.BenchmarkResult, error) {
	var results []interfaces.BenchmarkResult

	for _, query := range queries {
		result := interfaces.BenchmarkResult{
			Query: query,
		}

		// Time the search
		startTime := time.Now()
		searchResults, err := b.database.Search(ctx, query, 10, nil) // Search top 10
		result.TimeToSearch = time.Since(startTime)

		if err != nil {
			result.Found = false
		} else if len(searchResults) > 0 {
			result.Found = true
			result.Distance = &searchResults[0].Score
			placement := 1 // Default to first position
			result.PlacementOrder = &placement
		} else {
			result.Found = false
		}

		results = append(results, result)
	}

	return results, nil
}

// SaveResults saves benchmark results to a file
func (b *BenchmarkClient) SaveResults(results *interfaces.BenchmarkResults, filepath string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal results: %v", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write results file: %v", err)
	}

	return nil
}

// generateBenchmarkQueries generates queries for benchmarking
func (b *BenchmarkClient) generateBenchmarkQueries(ctx context.Context, generateFromDocs bool) ([]BenchmarkQuery, error) {
	// For now, return some sample queries
	// In a full implementation, this would:
	// 1. Query the database for existing documents
	// 2. Generate questions based on document content
	// 3. Create ground truth mappings

	queries := []BenchmarkQuery{
		{
			Query:          "What is the main topic of this document?",
			ExpectedSource: "sample_doc_1",
		},
		{
			Query:          "Who is the author of this paper?",
			ExpectedSource: "sample_doc_2",
		},
		{
			Query:          "What are the key findings?",
			ExpectedSource: "sample_doc_3",
		},
		{
			Query:          "What methodology was used?",
			ExpectedSource: "sample_doc_4",
		},
		{
			Query:          "What are the conclusions?",
			ExpectedSource: "sample_doc_5",
		},
	}

	return queries, nil
}

// BenchmarkQuery represents a query for benchmarking
type BenchmarkQuery struct {
	Query          string
	ExpectedSource string
}

// runSingleQuery runs a single benchmark query
func (b *BenchmarkClient) runSingleQuery(ctx context.Context, query BenchmarkQuery) (interfaces.BenchmarkResult, error) {
	result := interfaces.BenchmarkResult{
		Query:          query.Query,
		ExpectedSource: query.ExpectedSource,
	}

	// Generate embedding for the query
	embedding, err := b.embedder.Embed(ctx, query.Query)
	if err != nil {
		return result, fmt.Errorf("failed to embed query: %v", err)
	}

	// Create a document for search (in a real implementation, you'd search by embedding)
	searchQuery := query.Query

	// Time the search
	startTime := time.Now()
	searchResults, err := b.database.Search(ctx, searchQuery, 10, nil)
	result.TimeToSearch = time.Since(startTime)

	if err != nil {
		result.Found = false
		return result, fmt.Errorf("search failed: %v", err)
	}

	// Analyze results
	result.Found = len(searchResults) > 0

	if len(searchResults) > 0 {
		result.Distance = &searchResults[0].Score

		// Find the placement of the expected source
		for i, searchResult := range searchResults {
			if searchResult.Document.Source == query.ExpectedSource {
				placement := i + 1
				result.PlacementOrder = &placement
				break
			}
		}
	}

	return result, nil
}

// calculateTopKAccuracy calculates accuracy metrics for different top-K values
func (b *BenchmarkClient) calculateTopKAccuracy(results *interfaces.BenchmarkResults) map[int]float64 {
	accuracy := make(map[int]float64)
	totalQueries := 0

	// Count total queries
	for _, docResults := range results.ResultsByDoc {
		totalQueries += len(docResults)
	}

	if totalQueries == 0 {
		return accuracy
	}

	// Calculate accuracy for different K values
	for k := 1; k <= 10; k++ {
		correct := 0

		for _, docResults := range results.ResultsByDoc {
			for _, result := range docResults {
				if result.PlacementOrder != nil && *result.PlacementOrder <= k {
					correct++
				}
			}
		}

		accuracy[k] = float64(correct) / float64(totalQueries) * 100.0
	}

	return accuracy
}

// Helper functions for statistical analysis

// calculateMean calculates the mean of a slice of float64
func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// calculateMedian calculates the median of a slice of float64
func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}

// calculateStdDev calculates the standard deviation of a slice of float64
func calculateStdDev(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += math.Pow(v-mean, 2)
	}
	return math.Sqrt(sum / float64(len(values)-1))
}

// calculatePercentile calculates the p-th percentile of a slice of float64
func calculatePercentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	index := int(math.Round(float64(len(sorted)-1) * p / 100.0))
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}


