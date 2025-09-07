package shared

import (
	"fmt"
)

// ProviderFactory creates LLM providers based on configuration
type ProviderFactory struct {
	providers map[string]LLMProvider
}

// NewProviderFactory creates a new provider factory
func NewProviderFactory() *ProviderFactory {
	return &ProviderFactory{
		providers: make(map[string]LLMProvider),
	}
}

// RegisterProvider registers a custom provider
func (f *ProviderFactory) RegisterProvider(name string, provider LLMProvider) {
	f.providers[name] = provider
}

// GetProvider gets a registered provider by name
func (f *ProviderFactory) GetProvider(name string) (LLMProvider, error) {
	provider, exists := f.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider not found: %s", name)
	}
	return provider, nil
}

// ListProviders returns a list of registered provider names
func (f *ProviderFactory) ListProviders() []string {
	names := make([]string, 0, len(f.providers))
	for name := range f.providers {
		names = append(names, name)
	}
	return names
}

// NormalizeError normalizes different error types to ProviderError
func NormalizeError(err error) *ProviderError {
	if err == nil {
		return nil
	}

	if pe, ok := err.(*ProviderError); ok {
		return pe
	}

	// Handle specific error types from providers
	return &ProviderError{
		Code:    ErrUnknown,
		Message: err.Error(),
	}
}

// ValidateCompletionRequest validates a completion request
func ValidateCompletionRequest(req *CompletionRequest) error {
	if req == nil {
		return &ProviderError{
			Code:    ErrInvalidRequest,
			Message: "request cannot be nil",
		}
	}

	if len(req.Messages) == 0 {
		return &ProviderError{
			Code:    ErrInvalidRequest,
			Message: "messages cannot be empty",
		}
	}

	for i, msg := range req.Messages {
		if msg.Role == "" {
			return &ProviderError{
				Code:    ErrInvalidRequest,
				Message: fmt.Sprintf("message %d: role cannot be empty", i),
			}
		}
		if msg.Role != RoleSystem && msg.Role != RoleUser && msg.Role != RoleAssistant && msg.Role != RoleTool {
			return &ProviderError{
				Code:    ErrInvalidRequest,
				Message: fmt.Sprintf("message %d: invalid role '%s'", i, msg.Role),
			}
		}
	}

	if req.Options.Model == "" {
		return &ProviderError{
			Code:    ErrInvalidRequest,
			Message: "model cannot be empty",
		}
	}

	return nil
}
