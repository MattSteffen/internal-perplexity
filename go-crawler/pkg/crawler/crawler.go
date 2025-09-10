package crawler

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go-crawler/internal/config"
	"go-crawler/internal/embeddings"
	"go-crawler/internal/llm"
	"go-crawler/internal/processing"
	"go-crawler/internal/storage"
	"go-crawler/pkg/interfaces"
)

// Crawler is the main orchestrator for document processing
type Crawler struct {
	config    *config.CrawlerConfig
	converter interfaces.Converter
	extractor interfaces.Extractor
	embedder  interfaces.Embedder
	database  interfaces.DatabaseClient
	llm       interfaces.LLM

	// Stats
	stats      interfaces.CrawlerStats
	statsMutex sync.RWMutex

	// Processing
	maxConcurrency int
	batchSize      int
}

// New creates a new crawler instance
func New(cfg *config.CrawlerConfig) (*Crawler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}

	c := &Crawler{
		config:         cfg,
		maxConcurrency: cfg.MaxConcurrency,
		batchSize:      cfg.BatchSize,
		stats: interfaces.CrawlerStats{
			TotalDocuments:     0,
			ProcessedDocuments: 0,
			FailedDocuments:    0,
			TotalChunks:        0,
			ProcessingTime:     0,
			AverageChunkTime:   0,
		},
	}

	// Initialize components
	if err := c.initializeComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize components: %v", err)
	}

	return c, nil
}

// initializeComponents sets up all the required components
func (c *Crawler) initializeComponents() error {
	// Initialize embedder
	embedderFactory := embeddings.NewFactory()
	embedder, err := embedderFactory.Create(&c.config.Embeddings)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %v", err)
	}
	c.embedder = embedder

	// Initialize LLM
	llmFactory := llm.NewFactory()
	llmClient, err := llmFactory.Create(&c.config.LLM)
	if err != nil {
		return fmt.Errorf("failed to create LLM: %v", err)
	}
	c.llm = llmClient

	// Initialize database
	storageFactory := storage.NewFactory()
	database, err := storageFactory.Create(&c.config.Database, c.embedder.GetDimension(), c.config.MetadataSchema)
	if err != nil {
		return fmt.Errorf("failed to create database client: %v", err)
	}
	c.database = database

	// Create collection
	ctx := context.Background()
	if err := c.database.CreateCollection(ctx, c.config.Database.Recreate); err != nil {
		return fmt.Errorf("failed to create collection: %v", err)
	}

	// Initialize vision LLM if configured
	var visionLLM interfaces.LLM
	if c.config.VisionLLM != nil {
		visionLLM, err = llmFactory.Create(c.config.VisionLLM)
		if err != nil {
			return fmt.Errorf("failed to create vision LLM: %v", err)
		}
	}

	// Initialize converter
	converterFactory := processing.NewFactory()
	converter, err := converterFactory.Create(&c.config.Converter, visionLLM)
	if err != nil {
		return fmt.Errorf("failed to create converter: %v", err)
	}
	c.converter = converter

	// Initialize extractor
	extractor, err := converterFactory.CreateExtractor(&c.config.Extractor, c.llm)
	if err != nil {
		return fmt.Errorf("failed to create extractor: %v", err)
	}
	c.extractor = extractor

	return nil
}

// Crawl processes files or directories
func (c *Crawler) Crawl(ctx context.Context, path string) error {
	startTime := time.Now()

	// Get all files to process
	files, err := c.getFilesToProcess(path)
	if err != nil {
		return fmt.Errorf("failed to get files to process: %v", err)
	}

	c.statsMutex.Lock()
	c.stats.TotalDocuments = len(files)
	c.statsMutex.Unlock()

	if len(files) == 0 {
		return fmt.Errorf("no files found to process")
	}

	// Create processing channels
	fileChan := make(chan string, len(files))
	resultChan := make(chan ProcessResult, len(files))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < c.maxConcurrency; i++ {
		wg.Add(1)
		go c.worker(ctx, fileChan, resultChan, &wg)
	}

	// Send files to workers
	go func() {
		defer close(fileChan)
		for _, file := range files {
			select {
			case fileChan <- file:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Wait for all workers to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for result := range resultChan {
		c.statsMutex.Lock()
		if result.Error != nil {
			c.stats.FailedDocuments++
		} else {
			c.stats.ProcessedDocuments++
			c.stats.TotalChunks += result.ChunksProcessed
		}
		c.statsMutex.Unlock()
	}

	// Update timing stats
	c.statsMutex.Lock()
	c.stats.ProcessingTime = time.Since(startTime)
	if c.stats.TotalChunks > 0 {
		c.stats.AverageChunkTime = c.stats.ProcessingTime / time.Duration(c.stats.TotalChunks)
	}
	c.statsMutex.Unlock()

	return nil
}

// worker processes files from the channel
func (c *Crawler) worker(ctx context.Context, fileChan <-chan string, resultChan chan<- ProcessResult, wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		select {
		case file, ok := <-fileChan:
			if !ok {
				return
			}

			result := c.processFile(ctx, file)
			select {
			case resultChan <- result:
			case <-ctx.Done():
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

// ProcessResult contains the result of processing a single file
type ProcessResult struct {
	File            string
	ChunksProcessed int
	Error           error
}

// processFile processes a single file
func (c *Crawler) processFile(ctx context.Context, filePath string) ProcessResult {
	result := ProcessResult{File: filePath}

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		result.Error = fmt.Errorf("file does not exist: %s", filePath)
		return result
	}

	// Convert file to markdown (placeholder)
	markdown, err := c.convertFile(ctx, filePath)
	if err != nil {
		result.Error = fmt.Errorf("failed to convert file: %v", err)
		return result
	}

	// Extract metadata (placeholder)
	metadata, err := c.extractMetadata(ctx, markdown)
	if err != nil {
		result.Error = fmt.Errorf("failed to extract metadata: %v", err)
		return result
	}

	// Chunk text
	chunks, err := c.chunkText(markdown, c.config.ChunkSize)
	if err != nil {
		result.Error = fmt.Errorf("failed to chunk text: %v", err)
		return result
	}

	// Process chunks
	documents := make([]interfaces.Document, len(chunks))
	for i, chunk := range chunks {
		// Generate embedding
		embedding, err := c.embedder.Embed(ctx, chunk)
		if err != nil {
			result.Error = fmt.Errorf("failed to embed chunk %d: %v", i, err)
			return result
		}

		documents[i] = interfaces.Document{
			Text:          chunk,
			TextEmbedding: embedding,
			ChunkIndex:    i,
			Source:        filePath,
			Metadata:      metadata,
		}
	}

	// Store documents
	if err := c.database.InsertDocuments(ctx, documents); err != nil {
		result.Error = fmt.Errorf("failed to store documents: %v", err)
		return result
	}

	result.ChunksProcessed = len(chunks)
	return result
}

// convertFile converts a file to markdown using the configured converter
func (c *Crawler) convertFile(ctx context.Context, filePath string) (string, error) {
	if c.converter == nil {
		return "", fmt.Errorf("converter not initialized")
	}

	return c.converter.Convert(ctx, filePath)
}

// extractMetadata extracts metadata from text using the configured extractor
func (c *Crawler) extractMetadata(ctx context.Context, text string) (map[string]interface{}, error) {
	if c.extractor == nil {
		// Fallback to basic metadata if extractor is not initialized
		return map[string]interface{}{
			"file_size":    len(text),
			"processed_at": time.Now().Format(time.RFC3339),
		}, nil
	}

	return c.extractor.ExtractMetadata(ctx, text)
}

// chunkText splits text into chunks
func (c *Crawler) chunkText(text string, chunkSize int) ([]string, error) {
	if chunkSize <= 0 {
		chunkSize = 1000
	}

	var chunks []string
	runes := []rune(text)

	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunk := string(runes[i:end])
		if strings.TrimSpace(chunk) != "" {
			chunks = append(chunks, chunk)
		}
	}

	return chunks, nil
}

// getFilesToProcess gets all files that need to be processed
func (c *Crawler) getFilesToProcess(path string) ([]string, error) {
	var files []string

	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		// Walk directory
		err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if !info.IsDir() && c.isSupportedFile(filePath) {
				files = append(files, filePath)
			}

			return nil
		})
		if err != nil {
			return nil, err
		}
	} else {
		// Single file
		if c.isSupportedFile(path) {
			files = append(files, path)
		}
	}

	return files, nil
}

// isSupportedFile checks if a file type is supported
func (c *Crawler) isSupportedFile(filePath string) bool {
	ext := strings.ToLower(filepath.Ext(filePath))

	// Basic file type support - this would be expanded based on converter capabilities
	supportedExts := []string{
		".txt", ".md", ".pdf", ".docx", ".html", ".xml",
		".json", ".csv", ".xlsx", ".pptx",
	}

	for _, supportedExt := range supportedExts {
		if ext == supportedExt {
			return true
		}
	}

	return false
}

// Benchmark runs benchmarking on the processed documents
func (c *Crawler) Benchmark(ctx context.Context) (*interfaces.BenchmarkResults, error) {
	// Create benchmark client
	benchmarkClient := &struct{}{} // Placeholder - would create actual benchmark client
	_ = benchmarkClient            // Prevent unused variable warning

	// For now, return empty results - in full implementation would use benchmark client
	return &interfaces.BenchmarkResults{
		ResultsByDoc:           make(map[string][]interfaces.BenchmarkResult),
		PlacementDistribution:  make(map[int]int),
		DistanceDistribution:   []float64{},
		PercentInTopK:          make(map[int]float64),
		SearchTimeDistribution: []time.Duration{},
	}, nil
}

// GetStats returns current crawler statistics
func (c *Crawler) GetStats() interfaces.CrawlerStats {
	c.statsMutex.RLock()
	defer c.statsMutex.RUnlock()
	return c.stats
}

// Close closes all resources
func (c *Crawler) Close() error {
	if c.database != nil {
		return c.database.Close()
	}
	return nil
}
