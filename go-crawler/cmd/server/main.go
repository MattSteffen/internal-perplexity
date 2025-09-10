package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/gorilla/mux"

	"go-crawler/internal/config"
	"go-crawler/pkg/crawler"
)

type Server struct {
	crawler *crawler.Crawler
	config  *config.CrawlerConfig
	router  *mux.Router
	server  *http.Server
}

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type CrawlRequest struct {
	Path string `json:"path"`
}

func main() {
	configFile := os.Getenv("CRAWLER_CONFIG")
	if configFile == "" {
		configFile = "crawler-config.json"
	}

	cfg, err := config.LoadConfigFromFile(configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	crawler, err := crawler.New(cfg)
	if err != nil {
		log.Fatalf("Failed to create crawler: %v", err)
	}
	defer crawler.Close()

	server := &Server{
		crawler: crawler,
		config:  cfg,
		router:  mux.NewRouter(),
	}

	server.setupRoutes()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	server.server = &http.Server{
		Addr:         ":" + port,
		Handler:      server.router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	// Setup graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.server.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
	}()

	log.Printf("Starting Go Crawler API server on port %s", port)
	log.Printf("Configuration loaded from: %s", configFile)

	if err := server.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed to start: %v", err)
	}

	log.Println("Server stopped")
}

func (s *Server) setupRoutes() {
	// API v1 routes
	api := s.router.PathPrefix("/api/v1").Subrouter()
	api.Use(s.loggingMiddleware)
	api.Use(s.corsMiddleware)

	// Health check
	api.HandleFunc("/health", s.handleHealth).Methods("GET")

	// Crawler operations
	api.HandleFunc("/crawl", s.handleCrawl).Methods("POST")
	api.HandleFunc("/benchmark", s.handleBenchmark).Methods("POST")

	// Statistics
	api.HandleFunc("/stats", s.handleStats).Methods("GET")

	// Configuration
	api.HandleFunc("/config", s.handleGetConfig).Methods("GET")

	// Static file serving for docs
	s.router.PathPrefix("/docs/").Handler(http.StripPrefix("/docs/", http.FileServer(http.Dir("docs/"))))
}

// Middleware for logging requests
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Log the request
		log.Printf("%s %s %s", r.Method, r.RequestURI, r.RemoteAddr)

		next.ServeHTTP(w, r)

		// Log the response time
		log.Printf("Completed %s %s in %v", r.Method, r.RequestURI, time.Since(start))
	})
}

// CORS middleware
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Health check endpoint
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().Format(time.RFC3339),
			"version":   "1.0.0",
		},
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// Crawl endpoint
func (s *Server) handleCrawl(w http.ResponseWriter, r *http.Request) {
	var req CrawlRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "Invalid JSON payload")
		return
	}

	if req.Path == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "Path is required")
		return
	}

	// Start crawl in background
	go func() {
		ctx := context.Background()
		if err := s.crawler.Crawl(ctx, req.Path); err != nil {
			log.Printf("Crawl failed: %v", err)
		}
	}()

	response := APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"message": "Crawl started successfully",
			"path":    req.Path,
			"status":  "running",
		},
	}

	s.writeJSONResponse(w, http.StatusAccepted, response)
}

// Benchmark endpoint
func (s *Server) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	results, err := s.crawler.Benchmark(ctx)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, fmt.Sprintf("Benchmark failed: %v", err))
		return
	}

	response := APIResponse{
		Success: true,
		Data:    results,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// Stats endpoint
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := s.crawler.GetStats()

	response := APIResponse{
		Success: true,
		Data:    stats,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// Configuration endpoint
func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	// Return a sanitized version of the config (without sensitive data)
	configCopy := *s.config

	// Mask sensitive information
	if configCopy.Database.Password != "" {
		configCopy.Database.Password = "***"
	}
	if configCopy.Embeddings.APIKey != "" {
		configCopy.Embeddings.APIKey = "***"
	}
	if configCopy.LLM.APIKey != "" {
		configCopy.LLM.APIKey = "***"
	}

	response := APIResponse{
		Success: true,
		Data:    configCopy,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// Helper methods

func (s *Server) writeJSONResponse(w http.ResponseWriter, status int, response APIResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode JSON response: %v", err)
	}
}

func (s *Server) writeErrorResponse(w http.ResponseWriter, status int, message string) {
	response := APIResponse{
		Success: false,
		Error:   message,
	}

	s.writeJSONResponse(w, status, response)
}

// Utility functions

func getQueryParamInt(r *http.Request, key string, defaultValue int) int {
	value := r.URL.Query().Get(key)
	if value == "" {
		return defaultValue
	}

	if intValue, err := strconv.Atoi(value); err == nil {
		return intValue
	}

	return defaultValue
}

func getQueryParamBool(r *http.Request, key string, defaultValue bool) bool {
	value := r.URL.Query().Get(key)
	if value == "" {
		return defaultValue
	}

	if boolValue, err := strconv.ParseBool(value); err == nil {
		return boolValue
	}

	return defaultValue
}
