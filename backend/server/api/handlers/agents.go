package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/openai"
)

// AgentHandler handles agent-related HTTP requests
type AgentHandler struct {
	primaryAgent agents.IntelligentAgent
}

// NewAgentHandler creates a new agent handler
func NewAgentHandler(primaryAgent agents.IntelligentAgent) *AgentHandler {
	return &AgentHandler{
		primaryAgent: primaryAgent,
	}
}

// ExecuteAgent handles POST /agents/{name}
func (h *AgentHandler) ExecuteAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use POST method")
		return
	}

	// Extract agent name from URL path
	agentName := strings.TrimPrefix(r.URL.Path, "/agents/")
	if agentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid agent name", "Agent name is required")
		return
	}

	// Only support primary agent for now
	if agentName != "primary" {
		h.writeJSONError(w, http.StatusNotFound, "Agent not found", "Only 'primary' agent is supported")
		return
	}

	var req api.ExecuteAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid JSON request", err.Error())
		return
	}

	// Validate request
	if req.Input == nil {
		h.writeJSONError(w, http.StatusBadRequest, "MISSING_REQUIRED_FIELD", "input field is required")
		return
	}

	// Extract API key from header
	apiKey := r.Header.Get("X-API-KEY")
	if apiKey == "" {
		apiKey = r.Header.Get("Authorization") // fallback to Authorization header
		if after, ok := strings.CutPrefix(apiKey, "Bearer "); ok {
			apiKey = after
		}
	}

	// Extract model from request body (with default)
	model := req.Model
	if model == "" {
		model = "gpt-4" // default model
	}

	// Build AgentInput structure to match current agent interface
	agentInput := &agents.AgentInput{
		Data:    req.Input,
		Context: req.Context,
	}

	// Check if there's a natural language query in the input
	if query, ok := req.Input["query"].(string); ok {
		agentInput.Query = query
	}

	// Add model and API key to context if not already present
	if agentInput.Context == nil {
		agentInput.Context = make(map[string]interface{})
	}
	if _, exists := agentInput.Context["model"]; !exists {
		agentInput.Context["model"] = model
	}
	if _, exists := agentInput.Context["api_key"]; !exists && apiKey != "" {
		agentInput.Context["api_key"] = apiKey
	}

	// Create LLM provider dynamically based on request
	llmProvider, err := openai.NewProvider(openai.Config{
		BaseURL: "https://api.openai.com/v1",
		APIKey:  apiKey,
	})
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to create LLM provider", err.Error())
		return
	}

	// Execute agent with dynamic provider
	result, err := h.primaryAgent.Execute(r.Context(), agentInput, llmProvider)
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Agent execution failed", err.Error())
		return
	}

	// Build response matching the AgentResult structure
	response := api.AgentResponse{
		Success:      result.Success,
		Result:       result.Content,
		Stats:        result.Metadata,
		ExecutionLog: result.ExecutionLog,
		Duration:     result.Duration.String(),
	}

	// Add additional fields if they exist in the result
	if result.TokensUsed != nil {
		if tokens, ok := result.TokensUsed.(int); ok {
			response.TokensUsed = tokens
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// GetAgentCapabilities handles GET /agents/{name}/capabilities
func (h *AgentHandler) GetAgentCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	// Extract agent name from URL path
	agentName := strings.TrimPrefix(r.URL.Path, "/agents/")
	agentName = strings.TrimSuffix(agentName, "/capabilities")

	if agentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid agent name", "Agent name is required")
		return
	}

	// Only support primary agent for now
	if agentName != "primary" {
		h.writeJSONError(w, http.StatusNotFound, "Agent not found", "Only 'primary' agent is supported")
		return
	}

	capabilities := h.primaryAgent.GetCapabilities()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"agent":        agentName,
		"capabilities": capabilities,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// GetAgentStats handles GET /agents/{name}/stats
func (h *AgentHandler) GetAgentStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	// Extract agent name from URL path
	agentName := strings.TrimPrefix(r.URL.Path, "/agents/")
	agentName = strings.TrimSuffix(agentName, "/stats")

	if agentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid agent name", "Agent name is required")
		return
	}

	// Only support primary agent for now
	if agentName != "primary" {
		h.writeJSONError(w, http.StatusNotFound, "Agent not found", "Only 'primary' agent is supported")
		return
	}

	stats := h.primaryAgent.GetStats()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"agent": agentName,
		"stats": stats,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// writeJSONError writes a JSON error response
func (h *AgentHandler) writeJSONError(w http.ResponseWriter, status int, message string, details string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	errorResp := api.ErrorResponse{
		Error: message,
		Code:  http.StatusText(status),
		Details: map[string]interface{}{
			"details": details,
		},
	}

	if err := json.NewEncoder(w).Encode(errorResp); err != nil {
		http.Error(w, "Failed to encode error response", http.StatusInternalServerError)
		return
	}
}
