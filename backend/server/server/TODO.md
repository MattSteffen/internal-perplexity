# HTTP Server - MVP Tasks

## Overview
Set up the HTTP server infrastructure with proper configuration, routing, and production-ready features. Focus on clean startup, graceful shutdown, and monitoring.

## Server Setup

### 1. Server Configuration
- [ ] Create server configuration structure using standard library
- [ ] Add environment-based configuration loading
- [ ] Implement configuration validation
- [ ] Add server startup/shutdown logging

```go
type ServerConfig struct {
    Host         string        // default: "0.0.0.0"
    Port         int           // default: 8080
    ReadTimeout  time.Duration // default: 30s
    WriteTimeout time.Duration // default: 30s
    IdleTimeout  time.Duration // default: 120s
    MaxHeaderBytes int         // default: 1048576
}

func loadConfig() (*ServerConfig, error) {
    config := &ServerConfig{
        Host:          getEnv("SERVER_HOST", "0.0.0.0"),
        Port:          getEnvInt("SERVER_PORT", 8080),
        ReadTimeout:   getEnvDuration("READ_TIMEOUT", 30*time.Second),
        WriteTimeout:  getEnvDuration("WRITE_TIMEOUT", 30*time.Second),
        IdleTimeout:   getEnvDuration("IDLE_TIMEOUT", 120*time.Second),
        MaxHeaderBytes: getEnvInt("MAX_HEADER_BYTES", 1048576),
    }
    return config, nil
}
```

### 2. net/http Server Setup (`server.go`)
- [ ] Create http.Server instance with configuration
- [ ] Add server middleware (logging, CORS, recovery)
- [ ] Implement route setup using http.ServeMux
- [ ] Add health check endpoints

```go
type Server struct {
    config      *ServerConfig
    mux         *http.ServeMux
    httpServer  *http.Server

    // Dependencies
    agentHandler *AgentHandler
    toolHandler  *ToolHandler
    taskHandler  *TaskHandler
}

func NewServer(config *ServerConfig, handlers *Handlers) *Server {
    mux := http.NewServeMux()

    // Health check
    mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
    })

    // API routes
    mux.HandleFunc("/api/v1/agents/", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        handlers.AgentHandler.ExecuteAgent(w, r)
    })

    mux.HandleFunc("/api/v1/tools/", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case http.MethodGet:
            handlers.ToolHandler.ListTools(w, r)
        case http.MethodPost:
            handlers.ToolHandler.ExecuteTool(w, r)
        default:
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        }
    })

    // Add middleware
    handler := LoggingMiddleware(mux)
    handler = CORSMiddleware(handler)

    httpServer := &http.Server{
        Addr:           fmt.Sprintf("%s:%d", config.Host, config.Port),
        Handler:        handler,
        ReadTimeout:    config.ReadTimeout,
        WriteTimeout:   config.WriteTimeout,
        IdleTimeout:    config.IdleTimeout,
        MaxHeaderBytes: config.MaxHeaderBytes,
    }

    return &Server{
        config:     config,
        mux:        mux,
        httpServer: httpServer,
    }
}
```

### 3. Server Lifecycle Management
- [ ] Implement graceful server startup
- [ ] Add graceful shutdown with timeout
- [ ] Implement signal handling (SIGTERM, SIGINT)
- [ ] Add server status monitoring

```go
func (s *Server) Start(ctx context.Context) error {
    log.Printf("Starting server on %s", s.httpServer.Addr)

    // Start server in goroutine
    go func() {
        if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Printf("Server failed to start: %v", err)
        }
    }()

    // Wait for shutdown signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("Shutting down server...")

    // Graceful shutdown with timeout
    shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
        log.Printf("Server forced to shutdown: %v", err)
        return err
    }

    log.Println("Server exited")
    return nil
}
```

## Middleware

### 4. Server Middleware
- [ ] Add request logging middleware
- [ ] Implement CORS middleware
- [ ] Create recovery middleware
- [ ] Add security headers middleware

```go
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        path := r.URL.Path
        raw := r.URL.RawQuery

        // Wrap response writer to capture status
        wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
        next.ServeHTTP(wrapped, r)

        end := time.Now()
        latency := end.Sub(start)

        if raw != "" {
            path = path + "?" + raw
        }

        log.Printf("HTTP Request: %s %s %d %v", r.Method, path, wrapped.statusCode, latency)
    })
}

func CORSMiddleware() func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Access-Control-Allow-Origin", "*")
            w.Header().Set("Access-Control-Allow-Credentials", "true")
            w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
            w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusNoContent)
                return
            }

            next.ServeHTTP(w, r)
        })
    }
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

## Health Checks

### 5. Health Monitoring
- [ ] Add health check endpoint (`/health`)
- [ ] Implement dependency health checks
- [ ] Add metrics endpoint (`/metrics`)
- [ ] Create readiness/liveness probes

```go
func (s *Server) healthCheck(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "status":    "ok",
        "timestamp": time.Now().Unix(),
        "version":   "1.0.0",
    }

    // Check dependencies
    checks := map[string]bool{
        "llm_provider": s.checkLLMHealth(),
        "database":     s.checkDatabaseHealth(),
    }

    allHealthy := true
    for _, healthy := range checks {
        if !healthy {
            allHealthy = false
            break
        }
    }

    health["checks"] = checks

    statusCode := http.StatusOK
    if !allHealthy {
        statusCode = http.StatusServiceUnavailable
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    json.NewEncoder(w).Encode(health)
}

func (s *Server) checkLLMHealth() bool {
    // Simple health check - just verify the service is running
    // In a real implementation, you might ping the LLM provider
    return true
}
```

## Testing Tasks

### 7. Server Tests
- [ ] Test server startup and shutdown
- [ ] Test configuration loading
- [ ] Test health check endpoints
- [ ] Test middleware functionality

```go
func TestServerStartup(t *testing.T) {
    config := &ServerConfig{
        Host: "localhost",
        Port: 0, // Use random port
    }

    server := NewServer(config, &Handlers{})

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    go func() {
        time.Sleep(100 * time.Millisecond)
        cancel() // Stop server after short time
    }()

    err := server.Start(ctx)
    assert.NoError(t, err)
}
```

## Implementation Priority

### Phase 1: Basic Server
1. [ ] Create server configuration
2. [ ] Set up net/http server with basic routes
3. [ ] Add health check endpoint
4. [ ] Test server startup

### Phase 2: Production Features
1. [ ] Add comprehensive middleware
2. [ ] Implement graceful shutdown
3. [ ] Add environment variable support
4. [ ] Test server lifecycle

### Phase 3: Monitoring
1. [ ] Add health checks for dependencies
2. [ ] Implement metrics collection
3. [ ] Add logging and monitoring
4. [ ] Comprehensive testing

## Success Criteria
- [ ] Server starts up cleanly
- [ ] All routes are properly configured
- [ ] Graceful shutdown works correctly
- [ ] Health checks report accurate status
- [ ] Environment variable support works

## Files to Create
- `server/server.go`
- `server/config.go`
- `server/middleware.go`
- `server/health.go`
- `server/server_test.go`
- `config/default.yaml`
- `config/test.yaml`
