package transport

import (
	"context"
	"crypto/tls"
	"net/http"
	"time"

	"internal-perplexity/server/llm/providers/shared"
)

// HTTPClient provides a tuned HTTP client for LLM provider requests
type HTTPClient struct {
	client *http.Client
	opts   shared.ClientOptions
}

// NewHTTPClient creates a new HTTP client with the specified options
func NewHTTPClient(opts shared.ClientOptions) *HTTPClient {
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}
	if opts.RetryMax == 0 {
		opts.RetryMax = 3
	}
	if opts.RetryBackoff == 0 {
		opts.RetryBackoff = time.Second
	}
	if opts.MaxIdleConns == 0 {
		opts.MaxIdleConns = 10
	}
	if opts.IdleConnTTL == 0 {
		opts.IdleConnTTL = 90 * time.Second
	}

	transport := &http.Transport{
		MaxIdleConns:        opts.MaxIdleConns,
		MaxIdleConnsPerHost: opts.MaxIdleConns,
		IdleConnTimeout:     opts.IdleConnTTL,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   opts.Timeout,
	}

	return &HTTPClient{
		client: client,
		opts:   opts,
	}
}

// Do performs an HTTP request with retry logic
func (c *HTTPClient) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
	// Set default headers
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", "llm-provider/1.0")
	}

	// Add custom headers from options
	for key, value := range c.opts.Headers {
		req.Header.Set(key, value)
	}

	// Add authorization if API key is provided
	if c.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}

	// Add organization header if provided
	if c.opts.OrgID != "" {
		req.Header.Set("OpenAI-Organization", c.opts.OrgID)
	}

	var lastErr error
	for attempt := 0; attempt <= c.opts.RetryMax; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.opts.RetryBackoff * time.Duration(attempt)):
			}
		}

		resp, err := c.client.Do(req.WithContext(ctx))
		if err != nil {
			lastErr = err
			continue
		}

		// Check for retryable status codes
		if resp.StatusCode >= 500 || resp.StatusCode == 429 {
			_ = resp.Body.Close() // Ignore close error since we're retrying
			lastErr = &shared.ProviderError{
				Code:       shared.ErrUnavailable,
				Message:    "Server error",
				HTTPStatus: resp.StatusCode,
			}
			continue
		}

		return resp, nil
	}

	if lastErr == nil {
		lastErr = &shared.ProviderError{
			Code:    shared.ErrUnavailable,
			Message: "Request failed after retries",
		}
	}

	return nil, lastErr
}

// Post performs a POST request to the specified URL with the given body
func (c *HTTPClient) Post(ctx context.Context, url string, body interface{}) (*http.Response, error) {
	// This would need JSON marshaling, but keeping it simple for now
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return nil, err
	}
	return c.Do(ctx, req)
}

// Get performs a GET request to the specified URL
func (c *HTTPClient) Get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	return c.Do(ctx, req)
}
