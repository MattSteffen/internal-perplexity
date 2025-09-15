package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"time"
)

// APIError represents an error returned from the OpenAI API.
type APIError struct {
	StatusCode int    `json:"-"`
	Message    string `json:"message"`
	Type       string `json:"type"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API error (status %d): %s (%s)", e.StatusCode, e.Message, e.Type)
}

// SendRequest sends a chat completion request to the OpenAI API and returns the standard response.
// It includes a retry mechanism for transient errors.
func SendRequest(ctx context.Context, request *ChatCompletionRequest, endpointURL string, apiKey string, retries int) (*ChatCompletionResponse, error) {
	if endpointURL == "" {
		return nil, fmt.Errorf("baseURL is required")
	}

	// Marshal the request body
	reqBodyBytes, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	client := &http.Client{
		Timeout: 90 * time.Second, // Set a reasonable timeout
	}

	var finalResp *http.Response

	for i := 0; i <= retries; i++ {
		// Create the HTTP request
		req, err := http.NewRequestWithContext(ctx, "POST", endpointURL, bytes.NewBuffer(reqBodyBytes))
		if err != nil {
			return nil, fmt.Errorf("error creating request: %w", err)
		}

		// Set headers
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		// Send the request
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Request failed: %v. Retry %d/%d...", err, i, retries)
			if i < retries {
				time.Sleep(time.Duration(math.Pow(2, float64(i))) * time.Second) // Exponential backoff
				continue
			}
			return nil, fmt.Errorf("request failed after %d retries: %w", retries, err)
		}

		finalResp = resp

		// Check for retryable status codes
		if (resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= http.StatusInternalServerError) && i < retries {
			backoff := time.Duration(math.Pow(2, float64(i))) * time.Second
			log.Printf("Request failed with status %s. Retrying in %v...", resp.Status, backoff)
			_ = resp.Body.Close()
			time.Sleep(backoff)
			continue
		}

		// If status is OK or it's a non-retryable error, break the loop
		break
	}

	defer finalResp.Body.Close()

	// Read the response body
	respBody, err := io.ReadAll(finalResp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	// Check if the request was successful
	if finalResp.StatusCode != http.StatusOK {
		var apiErr APIError
		if err := json.Unmarshal(respBody, &apiErr); err != nil {
			// If we can't parse the error, return a generic one
			return nil, fmt.Errorf("request failed with status %s and unparsable body: %s", finalResp.Status, string(respBody))
		}
		apiErr.StatusCode = finalResp.StatusCode
		return nil, &apiErr
	}

	// Unmarshal the successful response
	var completionResponse ChatCompletionResponse
	if err := json.Unmarshal(respBody, &completionResponse); err != nil {
		return nil, fmt.Errorf("error unmarshaling successful response: %w", err)
	}

	return &completionResponse, nil
}

// SendRequestStructured sends a request and attempts to unmarshal the response content
// into a specific generic type T. It's designed for use with JSON Schema mode.
func SendRequestStructured[T any](ctx context.Context, request *ChatCompletionRequest, baseURL string, apiKey string, retries int) (T, *ChatCompletionResponse, error) {
	var structuredResult T

	// First, get the standard response using the SendRequest function
	response, err := SendRequest(ctx, request, baseURL, apiKey, retries)
	if err != nil {
		return structuredResult, nil, err
	}

	// Check for a valid response structure
	if len(response.Choices) == 0 {
		return structuredResult, response, fmt.Errorf("API response contained no choices")
	}
	if response.Choices[0].Message.(*ChatCompletionAssistantMessage).Content == "" {
		// This can happen if the model calls a tool instead of providing content
		return structuredResult, response, fmt.Errorf("API response message content is nil, possibly due to a tool call")
	}

	// The content is a JSON string, so we unmarshal it into the generic type T
	contentJSON := response.Choices[0].Message.(*ChatCompletionAssistantMessage).Content
	err = json.Unmarshal([]byte(contentJSON), &structuredResult)
	if err != nil {
		return structuredResult, response, fmt.Errorf("failed to unmarshal content into structured type: %w", err)
	}

	return structuredResult, response, nil
}
