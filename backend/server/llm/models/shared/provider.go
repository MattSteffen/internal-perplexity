package shared

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
