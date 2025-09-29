package server

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"internal-perplexity/server/api/handlers"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/agents/main/chat"
	"internal-perplexity/server/llm/providers"
	"internal-perplexity/server/llm/tools"
	"internal-perplexity/server/llm/tools/document_summarizer"
)

// Server represents the agent server
type Server struct {
	handler   http.Handler
	LLMs      *providers.Registry
	Tools     *tools.Registry
	Agents    *agents.AgentRegistry
	listeners []net.Listener
	shutdown  chan struct{}
}

// Config holds server configuration
type Config struct {
	Address string
}

var defaultConfig = &Config{
	Address: ":8080",
}

// NewServer creates a new server instance
func NewServer(config *Config, envFunc func(string) string) (*Server, error) {
	if config == nil {
		config = defaultConfig
	}

	server := &Server{
		shutdown: make(chan struct{}),
	}

	_ = envFunc
	// Initialize LLM registry
	server.LLMs = providers.NewRegistry()

	fmt.Printf("server.LLMs, %+v\n", server.LLMs)

	// Initialize tool registry and register tools
	server.Tools = tools.NewRegistry()
	fmt.Printf("server.Tools, %+v\n", server.Tools)
	server.Tools.Register(document_summarizer.NewDocumentSummarizer())

	// Initialize agents
	server.Agents = agents.NewAgentRegistry(server.Tools, server.LLMs)
	fmt.Printf("server.Agents, %+v\n", server.Agents)
	server.Agents.Register(chat.NewChatAgent())
	fmt.Printf("server.Agents, %+v\n", server.Agents)

	// Setup HTTP server
	if err := server.setupHTTPServer(config); err != nil {
		return nil, fmt.Errorf("failed to setup HTTP server: %w", err)
	}

	return server, nil
}

// setupHTTPServer configures the HTTP server with routes and middleware
func (s *Server) setupHTTPServer(config *Config) error {
	// Initialize handlers
	agentHandler := handlers.NewAgentHandler(s.Agents)
	toolHandler := handlers.NewToolHandler(s.Tools)
	chatHandler := handlers.NewChatHandler(s.LLMs)
	availableHandler := handlers.NewAvailableHandler(s.LLMs, s.Agents)

	// Set up HTTP routes
	mux := http.NewServeMux()

	// Agent routes
	mux.HandleFunc("/agents/", func(w http.ResponseWriter, r *http.Request) {
		agentHandler.ExecuteAgent(w, r)
	})

	// OpenAI-compatible chat completions
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		chatHandler.ChatCompletions(w, r)
	})

	// OpenAI-compatible list models
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		availableHandler.ListModels(w, r)
	})

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
	s.handler = corsHandler(loggingHandler(mux))

	// Set up dual-stack listeners
	return s.setupListeners(config.Address)
}

// setupListeners creates IPv4 and IPv6 listeners
func (s *Server) setupListeners(address string) error {
	// Parse the address to get port
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("invalid address format: %w", err)
	}

	// IPv4 listener
	ipv4Listener, err := net.Listen("tcp4", "127.0.0.1:"+port)
	if err != nil {
		return fmt.Errorf("failed to create IPv4 listener: %w", err)
	}
	s.listeners = append(s.listeners, ipv4Listener)

	// IPv6 listener
	ipv6Listener, err := net.Listen("tcp6", "[::1]:"+port)
	if err != nil {
		log.Printf("IPv6 bind failed: %v (continuing with IPv4 only)", err)
		// Don't fail if IPv6 is not available
	} else {
		s.listeners = append(s.listeners, ipv6Listener)
	}

	return nil
}

// Start starts the server and blocks until shutdown signal
func (s *Server) Start() error {
	if len(s.listeners) == 0 {
		return fmt.Errorf("no listeners configured")
	}

	// Get port from first listener for logging
	_, port, err := net.SplitHostPort(s.listeners[0].Addr().String())
	if err != nil {
		return fmt.Errorf("failed to parse listener address: %w", err)
	}

	log.Printf("Starting agent server on port %s", port)
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
	log.Println("  POST /v1/chat/completions - OpenAI-compatible chat completions")
	log.Println("  GET /v1/models - List available model callers")
	log.Println("  GET /health - Health check")

	// Start listeners in separate goroutines
	serverErr := make(chan error, len(s.listeners))
	var wg sync.WaitGroup

	for i, listener := range s.listeners {
		wg.Add(1)
		go func(l net.Listener, index int) {
			defer wg.Done()
			networkType := "IPv4"
			if index > 0 {
				networkType = "IPv6"
			}
			log.Printf("Starting %s listener on %s", networkType, l.Addr())

			if err := http.Serve(l, s.handler); err != nil && err != http.ErrServerClosed {
				serverErr <- fmt.Errorf("%s server failed: %w", networkType, err)
			}
		}(listener, i)
	}

	// Wait for interrupt signal or server error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		log.Println("Shutting down server due to error...")
		return err
	case <-quit:
		log.Println("Shutting down server...")
		return s.Shutdown()
	}
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown() error {
	// Give outstanding requests 5 seconds to complete
	time.Sleep(5 * time.Second)

	var shutdownErrors []error

	// Close all listeners
	for _, listener := range s.listeners {
		if tcpListener, ok := listener.(*net.TCPListener); ok {
			if err := tcpListener.Close(); err != nil {
				shutdownErrors = append(shutdownErrors, fmt.Errorf("failed to close listener %s: %w", listener.Addr(), err))
			}
		}
	}

	// Wait a bit for graceful shutdown
	time.Sleep(100 * time.Millisecond)

	if len(shutdownErrors) > 0 {
		log.Printf("Shutdown completed with %d errors", len(shutdownErrors))
		for _, err := range shutdownErrors {
			log.Printf("Shutdown error: %v", err)
		}
		return fmt.Errorf("server shutdown completed with errors: %v", shutdownErrors[0])
	}

	log.Println("Server exited")
	return nil
}

// GetLLMProvider returns the LLM provider (for testing or external access)
func (s *Server) GetLLMProvider() *providers.Registry {
	return s.LLMs
}

// GetToolRegistry returns the tool registry (for testing or external access)
func (s *Server) GetToolRegistry() *tools.Registry {
	return s.Tools
}
