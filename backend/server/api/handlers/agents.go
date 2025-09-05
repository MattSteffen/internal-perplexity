package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/agents/main-agents/primary"
)

// AgentHandler handles agent-related HTTP requests
type AgentHandler struct {
	primaryAgent *primary.PrimaryAgent
}

// NewAgentHandler creates a new agent handler
func NewAgentHandler(primaryAgent *primary.PrimaryAgent) *AgentHandler {
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
		h.writeJSONError(w, http.StatusBadRequest, "Invalid request", err.Error())
		return
	}

	// Validate request
	if req.Input == nil {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid request", "input field is required")
		return
	}

	// Extract API key from header
	apiKey := r.Header.Get("X-API-KEY")

	// Extract model from request body (with default)
	model := req.Model
	if model == "" {
		model = "gpt-4" // default model
	}

	// Convert to AgentInput with context
	agentInput := &agents.AgentInput{
		Data: req.Input,
		Context: map[string]interface{}{
			"model":   model,
			"api_key": apiKey,
		},
	}

	// Execute agent directly
	result, err := h.primaryAgent.Execute(r.Context(), agentInput)
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Agent execution failed", err.Error())
		return
	}

	response := api.AgentResponse{
		Success: result.Success,
		Result:  result.Content,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
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

	json.NewEncoder(w).Encode(errorResp)
}
