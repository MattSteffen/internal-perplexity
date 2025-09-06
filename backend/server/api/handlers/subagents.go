package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
	subagentsummary "internal-perplexity/server/llm/agents/sub-agents/summary"
)

// SubAgentHandler handles sub-agent-related HTTP requests
type SubAgentHandler struct {
	summaryAgent *subagentsummary.SummaryAgent
}

// NewSubAgentHandler creates a new sub-agent handler
func NewSubAgentHandler(summaryAgent *subagentsummary.SummaryAgent) *SubAgentHandler {
	return &SubAgentHandler{
		summaryAgent: summaryAgent,
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
		h.writeJSONError(w, http.StatusBadRequest, "Invalid request", "input field is required")
		return
	}

	var result *agents.AgentResult
	var err error

	// Route to appropriate sub-agent
	switch subAgentName {
	case "summary":
		result, err = h.summaryAgent.Execute(r.Context(), &agents.AgentInput{
			Data:    req.Input,
			Context: req.Context,
		})
	default:
		h.writeJSONError(w, http.StatusNotFound, "Sub-agent not found", "Unknown sub-agent: "+subAgentName)
		return
	}

	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Sub-agent execution failed", err.Error())
		return
	}

	response := api.SubAgentResponse{
		Success: result.Success,
		Result:  result.Content,
		Stats:   result.Metadata,
	}

	if !result.Success {
		response.Error = "Sub-agent execution failed"
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
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
