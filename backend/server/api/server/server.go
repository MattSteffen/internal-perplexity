package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"internal-perplexity/server/api/handlers"
	"internal-perplexity/server/llm/agents/main-agents/primary"
	subagentsummary "internal-perplexity/server/llm/agents/sub-agents/summary"
	"internal-perplexity/server/llm/models/openai"
	"internal-perplexity/server/llm/models/shared"
	"internal-perplexity/server/llm/tools"
	"internal-perplexity/server/llm/tools/calculator"
	"internal-perplexity/server/llm/tools/document_summarizer"
)

// Server represents the agent server
type Server struct {
	httpServer   *http.Server
	llmProvider  shared.LLMProvider
	toolRegistry *tools.Registry
	primaryAgent *primary.PrimaryAgent
	summaryAgent *subagentsummary.SummaryAgent
}

// Config holds server configuration
type Config struct {
	Port         string
	LLMBaseURL   string
	LLMModel     string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	IdleTimeout  time.Duration
}

// NewServer creates a new server instance
func NewServer(config *Config) (*Server, error) {
	if config == nil {
		config = &Config{
			Port:         ":8080",
			LLMBaseURL:   "http://localhost:11434/v1",
			LLMModel:     "gpt-oss:20b",
			ReadTimeout:  30 * time.Second,
			WriteTimeout: 30 * time.Second,
			IdleTimeout:  60 * time.Second,
		}
	}

	server := &Server{}

	// Initialize LLM provider
	llmConfig := &shared.LLMConfig{
		BaseURL: config.LLMBaseURL,
		Model:   config.LLMModel,
	}
	log.Println("Initializing LLM provider...")
	server.llmProvider = openai.NewClient(llmConfig)

	// Initialize tool registry and register tools
	server.toolRegistry = tools.NewRegistry()
	server.toolRegistry.Register(document_summarizer.NewDocumentSummarizer(server.llmProvider))
	server.toolRegistry.Register(calculator.NewCalculator())

	// Initialize agents
	server.summaryAgent = subagentsummary.NewSummaryAgent(server.llmProvider)
	server.primaryAgent = primary.NewPrimaryAgent(server.llmProvider, server.summaryAgent)

	// Setup HTTP server
	if err := server.setupHTTPServer(config); err != nil {
		return nil, fmt.Errorf("failed to setup HTTP server: %w", err)
	}

	return server, nil
}

// setupHTTPServer configures the HTTP server with routes and middleware
func (s *Server) setupHTTPServer(config *Config) error {
	// Initialize handlers
	agentHandler := handlers.NewAgentHandler(s.primaryAgent)
	subAgentHandler := handlers.NewSubAgentHandler(s.summaryAgent)
	toolHandler := handlers.NewToolHandler(s.toolRegistry)

	// Set up HTTP routes
	mux := http.NewServeMux()

	// Agent routes
	mux.HandleFunc("/agents/", agentHandler.ExecuteAgent)

	// Sub-agent routes
	mux.HandleFunc("/sub-agents/", subAgentHandler.ExecuteSubAgent)

	// Tool routes
	mux.HandleFunc("/tools/", toolHandler.ExecuteTool)
	mux.HandleFunc("/tools", toolHandler.ListTools)

	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "healthy"}`))
	})

	// CORS middleware
	corsHandler := func(next http.Handler) http.Handler {
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

	// Logging middleware
	loggingHandler := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			next.ServeHTTP(w, r)
			log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
		})
	}

	// Wrap handlers with middleware
	handler := corsHandler(loggingHandler(mux))

	// Configure server
	s.httpServer = &http.Server{
		Addr:         config.Port,
		Handler:      handler,
		ReadTimeout:  config.ReadTimeout,
		WriteTimeout: config.WriteTimeout,
		IdleTimeout:  config.IdleTimeout,
	}

	return nil
}

// Start starts the server and blocks until shutdown signal
func (s *Server) Start() error {
	log.Printf("Starting agent server on %s", s.httpServer.Addr)
	log.Println("Available endpoints:")
	log.Println("  POST /agents/{name} - Execute main agents")
	log.Println("  POST /sub-agents/{name} - Execute sub-agents")
	log.Println("  POST /tools/{name} - Execute tools")
	log.Println("  GET /tools - List available tools")
	log.Println("  GET /health - Health check")

	// Start server in a goroutine
	serverErr := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- fmt.Errorf("server failed to start: %w", err)
		}
	}()

	// Wait for interrupt signal or server error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		return err
	case <-quit:
		log.Println("Shutting down server...")
		return s.Shutdown()
	}
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown() error {
	// Give outstanding requests 5 seconds to complete
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.httpServer.Shutdown(ctx); err != nil {
		return fmt.Errorf("server forced to shutdown: %w", err)
	}

	log.Println("Server exited")
	return nil
}

// GetLLMProvider returns the LLM provider (for testing or external access)
func (s *Server) GetLLMProvider() shared.LLMProvider {
	return s.llmProvider
}

// GetToolRegistry returns the tool registry (for testing or external access)
func (s *Server) GetToolRegistry() *tools.Registry {
	return s.toolRegistry
}

// GetPrimaryAgent returns the primary agent (for testing or external access)
func (s *Server) GetPrimaryAgent() *primary.PrimaryAgent {
	return s.primaryAgent
}
