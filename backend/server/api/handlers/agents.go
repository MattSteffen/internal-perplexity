package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
)

// AgentHandler handles agent-related HTTP requests
type AgentHandler struct {
	AgentRegistry *agents.AgentRegistry
}

// NewAgentHandler creates a new agent handler
func NewAgentHandler(registry *agents.AgentRegistry) *AgentHandler {
	return &AgentHandler{
		AgentRegistry: registry,
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

	var req agents.AgentInput
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

	if req.User.APIKey == "" && apiKey != "" {
		req.User.APIKey = apiKey
	}

	fmt.Printf("req, %+v\n", req)

	// Execute agent with dynamic provider
	response, err := h.AgentRegistry.Execute(r.Context(), agentName, &req)
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Agent execution failed", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
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
