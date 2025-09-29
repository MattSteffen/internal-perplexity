package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"internal-perplexity/server/api"
	llmapi "internal-perplexity/server/llm/api"
	"internal-perplexity/server/llm/providers"
)

// ChatHandler handles OpenAI-compatible chat completion requests using the LLM registry
type ChatHandler struct {
	LLMs *providers.Registry
}

// NewChatHandler creates a new chat handler backed by the LLM registry
func NewChatHandler(registry *providers.Registry) *ChatHandler {
	return &ChatHandler{LLMs: registry}
}

//	curl -X POST http://localhost:8080/v1/chat/completions \
//	  -H "Content-Type: application/json" \
//	  -H "Authorization: Bearer $OPENAI_API_KEY" \
//	  -d '{
//	        "model": "ollama/llama3.1:8b",
//	        "messages": [
//	          {"role":"user","content":"Hello!"}
//	        ]
//	      }'
func (h *ChatHandler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		h.writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed", "Use POST method")
		return
	}

	var req llmapi.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeJSONError(w, http.StatusBadRequest, "Invalid JSON request", err.Error())
		return
	}

	if req.Stream {
		h.writeJSONError(w, http.StatusBadRequest, "Streaming not supported", "Set stream=false or omit the field")
		return
	}

	// Extract API key from headers (X-API-KEY or Authorization: Bearer ...)
	apiKey := r.Header.Get("X-API-KEY")
	if apiKey == "" {
		apiKey = r.Header.Get("Authorization")
		if after, ok := strings.CutPrefix(apiKey, "Bearer "); ok {
			apiKey = after
		}
	}

	// Determine provider from model prefix (e.g., "ollama/llama3", "openai/gpt-4o")
	baseName := req.Model
	if idx := strings.Index(baseName, "/"); idx != -1 {
		baseName = baseName[:idx]
	}
	provider, err := h.LLMs.GetProvider(strings.TrimSuffix(baseName, "/"))
	if err != nil {
		h.writeJSONError(w, http.StatusBadRequest, "Unknown provider", err.Error())
		return
	}

	// Normalize model to provider local name (strip provider prefix)
	req.Model = strings.TrimPrefix(req.Model, provider.Name()+"/")

	// Optional guard: verify model supported
	if !provider.HasModel(req.Model) {
		h.writeJSONError(w, http.StatusBadRequest, "Model not supported", fmt.Sprintf("model %s not supported by provider %s", req.Model, provider.Name()))
		return
	}

	// Execute completion via provider
	completion, err := provider.Complete(r.Context(), &req, apiKey)
	if err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Completion failed", err.Error())
		return
	}

	// Return provider response directly (OpenAI-compatible)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(completion); err != nil {
		h.writeJSONError(w, http.StatusInternalServerError, "Failed to encode response", err.Error())
		return
	}
}

func (h *ChatHandler) writeJSONError(w http.ResponseWriter, status int, message string, details string) {
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
