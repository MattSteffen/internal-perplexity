package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/providers/openai"
	"internal-perplexity/server/llm/tools"
	"internal-perplexity/server/llm/tools/shared"
)

// ToolHandler handles tool-related HTTP requests
type ToolHandler struct {
	toolRegistry *tools.Registry
}

// NewToolHandler creates a new tool handler
func NewToolHandler(toolRegistry *tools.Registry) *ToolHandler {
	return &ToolHandler{
		toolRegistry: toolRegistry,
	}
}

// ExecuteTool handles POST /tools/{name}
func (h *ToolHandler) ExecuteTool(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use POST method")
		return
	}

	// Extract tool name from URL path
	toolName := strings.TrimPrefix(r.URL.Path, "/tools/")
	if toolName == "" {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid tool name", "Tool name is required")
		return
	}

	var req api.ExecuteToolRequest
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

	// Create LLM provider dynamically
	llmProvider, err := openai.NewProvider(openai.Config{
		BaseURL: "https://api.openai.com/v1",
		APIKey:  apiKey,
	})
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to create LLM provider", err.Error())
		return
	}

	// Execute tool with dynamic provider
	input := &shared.ToolInput{
		Name: toolName,
		Data: req.Input,
	}

	result, err := h.toolRegistry.Execute(r.Context(), input, llmProvider)
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Tool execution failed", err.Error())
		return
	}

	response := api.ToolResponse{
		Success: result.Success,
		Output:  result.Data,
		Stats:   result.Stats,
	}

	if !result.Success {
		response.Error = result.Error
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// ListTools handles GET /tools
func (h *ToolHandler) ListTools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	tools := h.toolRegistry.List()
	toolList := make(map[string]interface{})

	for name, tool := range tools {
		toolList[name] = map[string]interface{}{
			"name":        tool.Name(),
			"description": tool.Description(),
			"schema":      tool.Schema(),
			"definition":  tool.Definition(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"tools": toolList,
	}); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

// writeJSONError writes a JSON error response
func (h *ToolHandler) writeJSONError(w http.ResponseWriter, status int, message string, details string) {
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
