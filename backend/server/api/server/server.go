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
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/shared"
	"internal-perplexity/server/llm/tools"
	"internal-perplexity/server/llm/tools/document_summarizer"
)

// Server represents the agent server
type Server struct {
	httpServer *http.Server
	LLMs       shared.LLMProvider
	Tools      *tools.Registry
	Agents     map[string]agents.Agent
}

// Config holds server configuration
type Config struct {
	Address string
}

var defaultConfig = &Config{
	Address: "http://localhost:8080",
}

// NewServer creates a new server instance
func NewServer(config *Config) (*Server, error) {
	if config == nil {
		config = defaultConfig
	}

	server := &Server{}

	// Initialize tool registry and register tools
	server.Tools = tools.NewRegistry()
	server.Tools.Register(document_summarizer.NewDocumentSummarizer())

	// Initialize agents
	// server.Agents = map[string]agents.Agent{
	// 	"summary": subagentsummary.NewSummaryAgent(),
	// }

	// Create sub-agents map for both primary agent and sub-agent handler

	// Setup HTTP server
	if err := server.setupHTTPServer(config); err != nil {
		return nil, fmt.Errorf("failed to setup HTTP server: %w", err)
	}

	return server, nil
}

// setupHTTPServer configures the HTTP server with routes and middleware
func (s *Server) setupHTTPServer(config *Config) error {
	// Initialize handlers
	// agentHandler := handlers.NewAgentHandler(s.Agents)
	toolHandler := handlers.NewToolHandler(s.Tools)

	// Set up HTTP routes
	mux := http.NewServeMux()

	// // Agent routes
	// mux.HandleFunc("/agents/", func(w http.ResponseWriter, r *http.Request) {
	// 	if strings.Contains(r.URL.Path, "/capabilities") {
	// 		agentHandler.GetAgentCapabilities(w, r)
	// 	} else if strings.Contains(r.URL.Path, "/stats") {
	// 		agentHandler.GetAgentStats(w, r)
	// 	} else {
	// 		agentHandler.ExecuteAgent(w, r)
	// 	}
	// })

	// // Sub-agent routes
	// mux.HandleFunc("/sub-agents/", func(w http.ResponseWriter, r *http.Request) {
	// 	if strings.Contains(r.URL.Path, "/capabilities") {
	// 		subAgentHandler.GetSubAgentCapabilities(w, r)
	// 	} else if strings.Contains(r.URL.Path, "/stats") {
	// 		subAgentHandler.GetSubAgentStats(w, r)
	// 	} else {
	// 		subAgentHandler.ExecuteSubAgent(w, r)
	// 	}
	// })
	// mux.HandleFunc("/sub-agents", subAgentHandler.ListSubAgents)

	// Tool routes
	mux.HandleFunc("/tools/", toolHandler.ExecuteTool)
	mux.HandleFunc("/tools", toolHandler.ListTools)

	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte(`{"status": "healthy"}`)); err != nil {
			http.Error(w, "Failed to write response", http.StatusInternalServerError)
			return
		}
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
		Addr:         config.Address,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 300 * time.Second,
		IdleTimeout:  600 * time.Second,
	}

	return nil
}

// Start starts the server and blocks until shutdown signal
func (s *Server) Start() error {
	log.Printf("Starting agent server on %s", s.httpServer.Addr)
	log.Println("Available endpoints:")
	// log.Println("  POST /agents/{name} - Execute main agents")
	// log.Println("  GET /agents/{name}/capabilities - Get agent capabilities")
	// log.Println("  GET /agents/{name}/stats - Get agent statistics")
	// log.Println("  POST /sub-agents/{name} - Execute sub-agents")
	// log.Println("  GET /sub-agents - List available sub-agents")
	// log.Println("  GET /sub-agents/{name}/capabilities - Get sub-agent capabilities")
	// log.Println("  GET /sub-agents/{name}/stats - Get sub-agent statistics")
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
	return s.LLMs
}

// GetToolRegistry returns the tool registry (for testing or external access)
func (s *Server) GetToolRegistry() *tools.Registry {
	return s.Tools
}

// GetPrimaryAgent returns the primary agent (for testing or external access)
func (s *Server) GetPrimaryAgent() agents.IntelligentAgent {
	return s.Agents["primary"].(agents.IntelligentAgent)
}
