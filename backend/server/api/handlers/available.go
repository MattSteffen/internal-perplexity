package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"internal-perplexity/server/api"
	"internal-perplexity/server/llm/agents"
	"internal-perplexity/server/llm/providers"
)

// ModelsResponse is an OpenAI-compatible response wrapper for list endpoints
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ModelInfo describes a single model entry (OpenAI-compatible shape)
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// AvailableHandler exposes discovery endpoints (e.g., /v1/models)
type AvailableHandler struct {
	LLMs   *providers.Registry
	Agents *agents.AgentRegistry
}

// NewAvailableHandler creates a new AvailableHandler
func NewAvailableHandler(llms *providers.Registry, agents *agents.AgentRegistry) *AvailableHandler {
	return &AvailableHandler{LLMs: llms, Agents: agents}
}

//	curl -X GET http://localhost:8080/v1/models \
//	  -H "Accept: application/json"
func (h *AvailableHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use GET method")
		return
	}

	now := time.Now().Unix()
	entries := make([]ModelInfo, 0, 16)

	// Providers as top-level model callers (e.g., "ollama")
	for _, name := range h.LLMs.ListProviders() {
		provider, err := h.LLMs.GetProvider(name)
		if err != nil {
			h.writeJSONError(w, http.StatusInternalServerError, "Failed to get provider", err.Error())
			return
		}
		models, err := provider.GetModels()
		if err != nil {
			h.writeJSONError(w, http.StatusInternalServerError, "Failed to get models", err.Error())
			return
		}
		for _, model := range models {
			entries = append(entries, ModelInfo{
				ID:      model.Name,
				Object:  "model",
				Created: now,
				OwnedBy: "provider",
			})
		}
		entries = append(entries, ModelInfo{
			ID:      name,
			Object:  "model",
			Created: now,
			OwnedBy: "provider",
		})
	}

	// Agents (main and support) are also model callers
	for _, a := range h.Agents.List() {
		entries = append(entries, ModelInfo{
			ID:      a.Name(),
			Object:  "model",
			Created: now,
			OwnedBy: "agent",
		})
	}

	resp := ModelsResponse{Object: "list", Data: entries}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

func (h *AvailableHandler) writeJSONError(w http.ResponseWriter, status int, message string, details string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	errorResp := api.ErrorResponse{
		Error: message,
		Code:  http.StatusText(status),
		Details: map[string]any{
			"details": details,
		},
	}

	_ = json.NewEncoder(w).Encode(errorResp)
}
