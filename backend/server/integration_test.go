package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullAgentFlowIntegration tests the complete agent execution flow
func TestFullAgentFlowIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full integration test in short mode")
	}

	// Setup test server with real dependencies
	server := setupTestServer(t)
	defer server.Close()

	// Test 1: Execute summary agent
	t.Run("SummaryAgentExecution", func(t *testing.T) {
		testSummaryAgentExecution(t, server)
	})

	// Test 2: Execute tool directly
	t.Run("ToolExecution", func(t *testing.T) {
		testToolExecution(t, server)
	})

	// Test 3: Async task execution
	t.Run("AsyncTaskExecution", func(t *testing.T) {
		testAsyncTaskExecution(t, server)
	})
}

func testSummaryAgentExecution(t *testing.T, server *httptest.Server) {
	// Prepare request
	req := ExecuteAgentRequest{
		Input: map[string]interface{}{
			"contents": []string{
				"Go is a statically typed programming language.",
				"It features garbage collection and concurrent execution.",
			},
			"instructions": "Summarize the key features",
		},
		Timeout: 60 * time.Second,
	}

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	// Make HTTP request
	resp, err := http.Post(
		server.URL+"/api/v1/sub-agents/summary",
		"application/json",
		bytes.NewBuffer(reqBody),
	)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	// Parse response
	var agentResp AgentResponse
	err = json.NewDecoder(resp.Body).Decode(&agentResp)
	require.NoError(t, err)

	assert.True(t, agentResp.Success)
	assert.NotNil(t, agentResp.Result)
	assert.NotNil(t, agentResp.Stats)

	// Verify result structure
	resultMap, ok := agentResp.Result.(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, resultMap, "query")
	assert.Contains(t, resultMap, "summary")
	assert.Contains(t, resultMap, "findings")

	// Verify stats
	assert.Greater(t, agentResp.Stats.TokensUsed, 0)
	assert.Greater(t, agentResp.Stats.Duration, int64(0))
}

func testToolExecution(t *testing.T, server *httptest.Server) {
	// Prepare request
	req := ExecuteToolRequest{
		Input: map[string]interface{}{
			"content":    "Go is a programming language developed by Google. It is statically typed and compiled.",
			"max_length": 30,
		},
	}

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	// Make HTTP request
	resp, err := http.Post(
		server.URL+"/api/v1/tools/document_summarizer",
		"application/json",
		bytes.NewBuffer(reqBody),
	)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	// Parse response
	var toolResp ToolResponse
	err = json.NewDecoder(resp.Body).Decode(&toolResp)
	require.NoError(t, err)

	assert.True(t, toolResp.Success)
	assert.NotNil(t, toolResp.Output)

	// Verify output structure
	outputMap, ok := toolResp.Output.(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, outputMap, "summary")

	summary, ok := outputMap["summary"].(string)
	require.True(t, ok)
	assert.NotEmpty(t, summary)
	assert.LessOrEqual(t, len(summary), 60) // Should respect max_length
}

func testAsyncTaskExecution(t *testing.T, server *httptest.Server) {
	// Create async task
	req := CreateTaskRequest{
		SubAgentName: "summary",
		Input: map[string]interface{}{
			"contents":     []string{"async test content"},
			"instructions": "summarize this content",
		},
	}

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	// Make HTTP request to create task
	resp, err := http.Post(
		server.URL+"/api/v1/tasks",
		"application/json",
		bytes.NewBuffer(reqBody),
	)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusAccepted, resp.StatusCode)

	// Parse task creation response
	var taskResp TaskCreatedResponse
	err = json.NewDecoder(resp.Body).Decode(&taskResp)
	require.NoError(t, err)

	assert.NotEmpty(t, taskResp.TaskID)
	assert.Equal(t, "pending", taskResp.Status)

	// Poll task status
	taskID := taskResp.TaskID
	var taskStatusResp TaskStatusResponse
	maxRetries := 30 // 30 seconds max

	for i := 0; i < maxRetries; i++ {
		time.Sleep(1 * time.Second)

		statusResp, err := http.Get(server.URL + "/api/v1/tasks/" + taskID)
		require.NoError(t, err)

		err = json.NewDecoder(statusResp.Body).Decode(&taskStatusResp)
		statusResp.Body.Close()
		require.NoError(t, err)

		if taskStatusResp.Status == "completed" || taskStatusResp.Status == "failed" {
			break
		}
	}

	assert.Equal(t, "completed", taskStatusResp.Status)
	assert.NotNil(t, taskStatusResp.Result)
	assert.Greater(t, taskStatusResp.Updated.Unix(), taskStatusResp.Created.Unix())
}

// TestAPIEndpoints tests all API endpoints
func TestAPIEndpoints(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping API endpoint tests in short mode")
	}

	server := setupTestServer(t)
	defer server.Close()

	endpoints := []struct {
		name     string
		method   string
		path     string
		body     interface{}
		expected int
	}{
		{"Health Check", "GET", "/health", nil, 200},
		{"List Tools", "GET", "/api/v1/tools", nil, 200},
		{"Agent Execution", "POST", "/api/v1/sub-agents/summary", ExecuteAgentRequest{
			Input: map[string]interface{}{
				"contents":     []string{"test content"},
				"instructions": "summarize",
			},
		}, 200},
		{"Tool Execution", "POST", "/api/v1/tools/document_summarizer", ExecuteToolRequest{
			Input: map[string]interface{}{
				"content":    "test content",
				"max_length": 20,
			},
		}, 200},
	}

	for _, ep := range endpoints {
		t.Run(ep.name, func(t *testing.T) {
			var reqBody bytes.Buffer
			if ep.body != nil {
				err := json.NewEncoder(&reqBody).Encode(ep.body)
				require.NoError(t, err)
			}

			req, err := http.NewRequest(ep.method, server.URL+ep.path, &reqBody)
			require.NoError(t, err)

			if ep.body != nil {
				req.Header.Set("Content-Type", "application/json")
			}

			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			assert.Equal(t, ep.expected, resp.StatusCode)
		})
	}
}

// TestConcurrentRequests tests multiple concurrent API requests
func TestConcurrentRequests(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent test in short mode")
	}

	server := setupTestServer(t)
	defer server.Close()

	const numRequests = 5
	results := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			req := ExecuteAgentRequest{
				Input: map[string]interface{}{
					"contents":     []string{fmt.Sprintf("concurrent content %d", id)},
					"instructions": "summarize",
				},
			}

			reqBody, err := json.Marshal(req)
			if err != nil {
				results <- err
				return
			}

			resp, err := http.Post(
				server.URL+"/api/v1/sub-agents/summary",
				"application/json",
				bytes.NewBuffer(reqBody),
			)
			if err != nil {
				results <- err
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				results <- fmt.Errorf("unexpected status: %d", resp.StatusCode)
				return
			}

			results <- nil
		}(i)
	}

	// Wait for all requests
	for i := 0; i < numRequests; i++ {
		err := <-results
		assert.NoError(t, err)
	}
}

// TestErrorHandling tests API error handling
func TestErrorHandling(t *testing.T) {
	server := setupTestServer(t)
	defer server.Close()

	// Test invalid agent
	resp, err := http.Post(
		server.URL+"/api/v1/agents/nonexistent",
		"application/json",
		bytes.NewBufferString(`{"input": {"query": "test"}}`),
	)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)

	// Test invalid tool
	resp, err = http.Post(
		server.URL+"/api/v1/tools/nonexistent",
		"application/json",
		bytes.NewBufferString(`{"input": {"content": "test"}}`),
	)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)

	// Test invalid JSON
	resp, err = http.Post(
		server.URL+"/api/v1/agents/researcher",
		"application/json",
		bytes.NewBufferString(`invalid json`),
	)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

// BenchmarkAPIEndpoints benchmarks API performance
func BenchmarkAPIEndpoints(b *testing.B) {
	server := setupTestServer(b)
	defer server.Close()

	req := ExecuteAgentRequest{
		Input: map[string]interface{}{
			"contents":     []string{"benchmark content"},
			"instructions": "summarize",
		},
	}

	reqBody, err := json.Marshal(req)
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := http.Post(
			server.URL+"/api/v1/sub-agents/summary",
			"application/json",
			bytes.NewBuffer(reqBody),
		)
		if err != nil {
			b.Fatal(err)
		}
		resp.Body.Close()
	}
}

// setupTestServer creates a test server with real dependencies
func setupTestServer(t testing.TB) *httptest.Server {
	// Setup LLM client
	llmClient := NewOllamaClient(&LLMConfig{
		BaseURL: "http://localhost:11434/v1",
		Model:   "gpt-oss:20b",
		APIKey:  "ollama",
	})

	// Setup tool registry
	toolRegistry := NewToolRegistry()
	toolRegistry.Register("web_search", NewWebSearchTool())
	toolRegistry.Register("document_summarizer", NewDocumentSummarizer(llmClient))

	// Setup agent manager
	agentManager := NewAgentManager(llmClient, toolRegistry)
	summary := NewSummaryAgent(llmClient)
	err := agentManager.RegisterSubAgent(context.Background(), &SubAgentConfig{
		Name:   "summary",
		Agent:  summary,
		Config: map[string]interface{}{},
	})
	require.NoError(t, err)

	// Setup handlers
	agentHandler := NewAgentHandler(agentManager, &RequestValidator{})
	toolHandler := NewToolHandler(toolRegistry, &RequestValidator{})
	taskHandler := NewTaskHandler(agentManager, &TaskStore{})

	// Setup router
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/agents/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		agentHandler.ExecuteAgent(w, r)
	})

	mux.HandleFunc("/api/v1/tools/", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			toolHandler.ListTools(w, r)
		case http.MethodPost:
			toolHandler.ExecuteTool(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/v1/tasks/", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			taskHandler.CreateTask(w, r)
		case http.MethodGet:
			taskHandler.GetTaskStatus(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	return httptest.NewServer(mux)
}
