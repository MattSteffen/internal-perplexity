package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers/openai"
)

// SubAgentHandler handles sub-agent-related HTTP requests
type SubAgentHandler struct {
	subAgents map[string]agents.Agent
}

// NewSubAgentHandler creates a new sub-agent handler
func NewSubAgentHandler(subAgents map[string]agents.Agent) *SubAgentHandler {
	return &SubAgentHandler{
		subAgents: subAgents,
	}
}

// ExecuteSubAgent handles POST /sub-agents/{name}
func (h *SubAgentHandler) ExecuteSubAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use POST method")
		return
	}

	// Extract sub-agent name from URL path
	subAgentName := strings.TrimPrefix(r.URL.Path, "/sub-agents/")
	if subAgentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid sub-agent name", "Sub-agent name is required")
		return
	}

	var req api.ExecuteSubAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid request", err.Error())
		return
	}

	// Validate request
	if req.Input == nil {
		h.writeJSONError(w, http.StatusBadRequest, "MISSING_REQUIRED_FIELD", "input field is required")
		return
	}

	// Extract API key and model from request
	apiKey := r.Header.Get("X-API-KEY")
	if apiKey == "" {
		apiKey = r.Header.Get("Authorization") // fallback to Authorization header
		if after, ok := strings.CutPrefix(apiKey, "Bearer "); ok {
			apiKey = after
		}
	}

	model := "gpt-4" // default model
	if req.Model != "" {
		model = req.Model
	}

	// Create LLM provider dynamically
	llmProvider, err := openai.NewProvider(openai.Config{
		BaseURL: "https://api.openai.com/v1",
		APIKey:  apiKey,
	})
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to create LLM provider", err.Error())
		return
	}

	// Find the requested sub-agent
	subAgent, exists := h.subAgents[subAgentName]
	if !exists {
		h.writeJSONError(w, http.StatusNotFound, "Sub-agent not found", "Unknown sub-agent: "+subAgentName)
		return
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

	// Execute the sub-agent
	result, err := subAgent.Execute(r.Context(), agentInput, llmProvider)

	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Sub-agent execution failed", err.Error())
		return
	}

	// Build response matching the AgentResult structure
	response := api.SubAgentResponse{
		Success:  result.Success,
		Result:   result.Content,
		Stats:    result.Metadata,
		Duration: result.Duration.String(),
	}

	// Add additional fields if they exist in the result
	if result.TokensUsed != nil {
		if tokens, ok := result.TokensUsed.(int); ok {
			response.TokensUsed = tokens
		}
	}

	if !result.Success {
		response.Error = "Sub-agent execution failed"
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// GetSubAgentCapabilities handles GET /sub-agents/{name}/capabilities
func (h *SubAgentHandler) GetSubAgentCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	// Extract sub-agent name from URL path
	subAgentName := strings.TrimPrefix(r.URL.Path, "/sub-agents/")
	subAgentName = strings.TrimSuffix(subAgentName, "/capabilities")

	if subAgentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid sub-agent name", "Sub-agent name is required")
		return
	}

	// Find the requested sub-agent
	subAgent, exists := h.subAgents[subAgentName]
	if !exists {
		h.writeJSONError(w, http.StatusNotFound, "Sub-agent not found", "Unknown sub-agent: "+subAgentName)
		return
	}

	capabilities := subAgent.GetCapabilities()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"sub_agent":    subAgentName,
		"capabilities": capabilities,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// GetSubAgentStats handles GET /sub-agents/{name}/stats
func (h *SubAgentHandler) GetSubAgentStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	// Extract sub-agent name from URL path
	subAgentName := strings.TrimPrefix(r.URL.Path, "/sub-agents/")
	subAgentName = strings.TrimSuffix(subAgentName, "/stats")

	if subAgentName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid sub-agent name", "Sub-agent name is required")
		return
	}

	// Find the requested sub-agent
	subAgent, exists := h.subAgents[subAgentName]
	if !exists {
		h.writeJSONError(w, http.StatusNotFound, "Sub-agent not found", "Unknown sub-agent: "+subAgentName)
		return
	}

	stats := subAgent.GetStats()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"sub_agent": subAgentName,
		"stats":     stats,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// ListSubAgents handles GET /sub-agents
func (h *SubAgentHandler) ListSubAgents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	subAgents := make(map[string]interface{})
	for name, agent := range h.subAgents {
		subAgents[name] = map[string]interface{}{
			"name":         name,
			"capabilities": agent.GetCapabilities(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"sub_agents": subAgents,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// writeJSONError writes a JSON error response
func (h *SubAgentHandler) writeJSONError(w http.ResponseWriter, status int, message string, details string) {
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
