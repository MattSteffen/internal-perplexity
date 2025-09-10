package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"go-crawler/internal/config"
	"go-crawler/pkg/crawler"
)

var (
	configFile string
	verbose    bool
)

func main() {
	var rootCmd = &cobra.Command{
		Use:   "crawler",
		Short: "Go Crawler - Document processing and vector database system",
		Long: `A high-performance document processing system that converts various file formats,
extracts structured metadata using LLMs, generates embeddings, and stores everything
in a vector database for retrieval and analysis.`,
	}

	rootCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "configuration file path")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	// Crawl command
	var crawlCmd = &cobra.Command{
		Use:   "crawl [path]",
		Short: "Crawl and process documents",
		Long:  `Process documents from the specified path (file or directory) and store them in the vector database.`,
		Args:  cobra.ExactArgs(1),
		RunE:  runCrawl,
	}
	crawlCmd.Flags().StringP("output", "o", "", "output directory for processed files")

	// Benchmark command
	var benchmarkCmd = &cobra.Command{
		Use:   "benchmark",
		Short: "Run benchmarks on processed documents",
		Long:  `Run performance benchmarks on the vector database to evaluate search quality and speed.`,
		RunE:  runBenchmark,
	}

	// Config command
	var configCmd = &cobra.Command{
		Use:   "config",
		Short: "Configuration management",
		Long:  `Manage crawler configuration files.`,
	}

	var configInitCmd = &cobra.Command{
		Use:   "init [filename]",
		Short: "Create a default configuration file",
		Long:  `Generate a default configuration file with all available options.`,
		Args:  cobra.MaximumNArgs(1),
		RunE:  runConfigInit,
	}

	var configValidateCmd = &cobra.Command{
		Use:   "validate [filename]",
		Short: "Validate a configuration file",
		Long:  `Validate the syntax and values of a configuration file.`,
		Args:  cobra.ExactArgs(1),
		RunE:  runConfigValidate,
	}

	configCmd.AddCommand(configInitCmd, configValidateCmd)

	// Stats command
	var statsCmd = &cobra.Command{
		Use:   "stats",
		Short: "Show crawler statistics",
		Long:  `Display statistics about processed documents and system performance.`,
		RunE:  runStats,
	}

	rootCmd.AddCommand(crawlCmd, benchmarkCmd, configCmd, statsCmd)

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runCrawl(cmd *cobra.Command, args []string) error {
	path := args[0]

	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	crawler, err := crawler.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to create crawler: %v", err)
	}
	defer crawler.Close()

	if verbose {
		fmt.Printf("Starting crawl of: %s\n", path)
		fmt.Printf("Configuration: %s\n", cfg.String())
	}

	// Set up signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nReceived signal, shutting down gracefully...")
		cancel()
	}()

	// Start crawling
	startTime := time.Now()
	err = crawler.Crawl(ctx, path)
	duration := time.Since(startTime)

	if err != nil {
		return fmt.Errorf("crawl failed: %v", err)
	}

	// Show results
	stats := crawler.GetStats()
	fmt.Printf("\nCrawl completed successfully!\n")
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Total files: %d\n", stats.TotalDocuments)
	fmt.Printf("Processed: %d\n", stats.ProcessedDocuments)
	fmt.Printf("Failed: %d\n", stats.FailedDocuments)
	fmt.Printf("Total chunks: %d\n", stats.TotalChunks)
	fmt.Printf("Average chunk time: %v\n", stats.AverageChunkTime)

	return nil
}

func runBenchmark(cmd *cobra.Command, args []string) error {
	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	crawler, err := crawler.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to create crawler: %v", err)
	}
	defer crawler.Close()

	ctx := context.Background()
	results, err := crawler.Benchmark(ctx)
	if err != nil {
		return fmt.Errorf("benchmark failed: %v", err)
	}

	// Display benchmark results
	fmt.Println("Benchmark Results:")
	fmt.Printf("Total queries: %d\n", len(results.ResultsByDoc))

	if len(results.PlacementDistribution) > 0 {
		fmt.Println("\nPlacement Distribution:")
		for placement, count := range results.PlacementDistribution {
			fmt.Printf("  Position %d: %d queries\n", placement, count)
		}
	}

	if len(results.PercentInTopK) > 0 {
		fmt.Println("\nTop-K Accuracy:")
		for k, accuracy := range results.PercentInTopK {
			fmt.Printf("  Top-%d: %.2f%%\n", k, accuracy)
		}
	}

	if len(results.DistanceDistribution) > 0 {
		fmt.Printf("\nAverage Distance: %.4f\n", average(results.DistanceDistribution))
	}

	if len(results.SearchTimeDistribution) > 0 {
		avgTime := averageDuration(results.SearchTimeDistribution)
		fmt.Printf("Average Search Time: %v\n", avgTime)
	}

	return nil
}

func runConfigInit(cmd *cobra.Command, args []string) error {
	filename := "crawler-config.json"
	if len(args) > 0 {
		filename = args[0]
	}

	// Create default configuration
	cfg := config.DefaultConfig()

	// Save to file
	if err := cfg.SaveToFile(filename); err != nil {
		return fmt.Errorf("failed to save config file: %v", err)
	}

	fmt.Printf("Default configuration saved to: %s\n", filename)
	fmt.Printf("Edit this file to customize your crawler settings.\n")

	return nil
}

func runConfigValidate(cmd *cobra.Command, args []string) error {
	filename := args[0]

	cfg, err := config.LoadConfigFromFile(filename)
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	fmt.Printf("Configuration file '%s' is valid!\n", filename)

	if verbose {
		fmt.Printf("\nConfiguration details:\n")
		configJSON, _ := json.MarshalIndent(cfg, "", "  ")
		fmt.Println(string(configJSON))
	}

	return nil
}

func runStats(cmd *cobra.Command, args []string) error {
	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	crawler, err := crawler.New(cfg)
	if err != nil {
		return fmt.Errorf("failed to create crawler: %v", err)
	}
	defer crawler.Close()

	stats := crawler.GetStats()

	fmt.Println("Crawler Statistics:")
	fmt.Printf("Total documents: %d\n", stats.TotalDocuments)
	fmt.Printf("Processed documents: %d\n", stats.ProcessedDocuments)
	fmt.Printf("Failed documents: %d\n", stats.FailedDocuments)
	fmt.Printf("Total chunks: %d\n", stats.TotalChunks)
	fmt.Printf("Processing time: %v\n", stats.ProcessingTime)
	fmt.Printf("Average chunk time: %v\n", stats.AverageChunkTime)

	return nil
}

func loadConfig() (*config.CrawlerConfig, error) {
	if configFile == "" {
		// Try to find config file in current directory or common locations
		possibleFiles := []string{
			"crawler-config.json",
			"crawler-config.yaml",
			"config.json",
			"config.yaml",
		}

		for _, file := range possibleFiles {
			if _, err := os.Stat(file); err == nil {
				configFile = file
				break
			}
		}

		if configFile == "" {
			return nil, fmt.Errorf("no configuration file found. Use --config flag or create a config file")
		}
	}

	// Determine file type and load accordingly
	ext := filepath.Ext(configFile)
	switch ext {
	case ".json":
		return config.LoadConfigFromFile(configFile)
	case ".yaml", ".yml":
		return nil, fmt.Errorf("YAML config files not yet supported")
	default:
		return nil, fmt.Errorf("unsupported config file format: %s", ext)
	}
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func averageDuration(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}

	sum := time.Duration(0)
	for _, v := range values {
		sum += v
	}
	return sum / time.Duration(len(values))
}
