package openai

import (
	"fmt"

	"internal-perplexity/server/llm/providers/shared"

	"github.com/sashabaranov/go-openai"
)

// ToOpenAIRequest converts a shared CompletionRequest to OpenAI format
func ToOpenAIRequest(req *shared.CompletionRequest) (*openai.ChatCompletionRequest, error) {
	msgs := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+1)

	// Add system message if provided
	if req.System != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: req.System,
		})
	}

	// Convert messages
	for _, m := range req.Messages {
		msg := openai.ChatCompletionMessage{
			Role:    string(m.Role),
			Content: m.Content,
		}

		// Handle tool calls if present
		if len(m.ToolCalls) > 0 {
			// Convert tool calls to OpenAI format
			toolCalls := make([]openai.ToolCall, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				toolCalls[i] = openai.ToolCall{
					ID:   tc.ID,
					Type: "function", // OpenAI uses "function" type
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: fmt.Sprintf("%v", tc.Arguments), // Convert to JSON string
					},
				}
			}
			msg.ToolCalls = toolCalls
		}

		// Handle tool invocation if present
		if m.ToolInvocation != nil {
			// Convert tool invocation to tool message
			msg.Role = openai.ChatMessageRoleTool
			msg.Content = fmt.Sprintf("%v", m.ToolInvocation.Result)
			if m.ToolInvocation.CallID != "" {
				msg.ToolCallID = m.ToolInvocation.CallID
			}
		}

		msgs = append(msgs, msg)
	}

	o := req.Options
	openaiReq := openai.ChatCompletionRequest{
		Model:            o.Model,
		Messages:         msgs,
		MaxTokens:        o.MaxTokens,
		Temperature:      float32(o.Temperature),
		TopP:             float32(o.TopP),
		Stop:             o.Stop,
		PresencePenalty:  o.PresencePenalty,
		FrequencyPenalty: o.FrequencyPenalty,
	}

	// Handle JSON mode
	if o.ResponseFormat == shared.ResponseFormatJSON {
		openaiReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: "json_object",
		}
	}

	// Handle tools
	if len(o.Tools) > 0 {
		tools := make([]openai.Tool, len(o.Tools))
		for i, t := range o.Tools {
			tools[i] = openai.Tool{
				Type: "function",
				Function: &openai.FunctionDefinition{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.JSONSchema,
				},
			}
		}
		openaiReq.Tools = tools
		openaiReq.ToolChoice = "auto" // Default to auto
		if o.ParallelTools {
			openaiReq.ToolChoice = "parallel"
		}
	}

	return &openaiReq, nil
}

// ToOpenAIStreamRequest converts a shared CompletionRequest to OpenAI streaming format
func ToOpenAIStreamRequest(req *shared.CompletionRequest) (*openai.ChatCompletionRequest, error) {
	r, err := ToOpenAIRequest(req)
	if err != nil {
		return nil, err
	}
	r.Stream = true
	return r, nil
}

// FromOpenAIResponse converts an OpenAI response to shared format
func FromOpenAIResponse(resp openai.ChatCompletionResponse) *shared.CompletionResponse {
	var content string
	var messages []shared.Message
	var stopReason string

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		content = choice.Message.Content
		stopReason = string(choice.FinishReason)

		// Convert the assistant message
		msg := shared.Message{
			Role:    shared.RoleAssistant,
			Content: choice.Message.Content,
		}

		// Handle tool calls if present
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls := make([]shared.ToolCall, len(choice.Message.ToolCalls))
			for i, tc := range choice.Message.ToolCalls {
				toolCalls[i] = shared.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					// Note: Arguments would need JSON parsing in a real implementation
				}
			}
			msg.ToolCalls = toolCalls
		}

		messages = append(messages, msg)
	}

	return &shared.CompletionResponse{
		Content:  content,
		Messages: messages,
		Usage: shared.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		StopReason:  stopReason,
		RawProvider: resp,
	}
}

// FromOpenAIStream converts an OpenAI streaming response to shared format
func FromOpenAIStream(resp openai.ChatCompletionStreamResponse) *shared.StreamChunk {
	chunk := &shared.StreamChunk{
		RawProvider: resp,
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if choice.Delta.Content != "" {
			chunk.DeltaText = choice.Delta.Content
		}

		if choice.FinishReason != "" {
			chunk.Done = true
		}

		// Handle tool calls in streaming
		if len(choice.Delta.ToolCalls) > 0 {
			toolDelta := &shared.ToolDelta{}
			toolCalls := make([]shared.ToolCall, len(choice.Delta.ToolCalls))
			for i, tc := range choice.Delta.ToolCalls {
				toolCalls[i] = shared.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					// Arguments would be built up incrementally in a real implementation
				}
			}
			toolDelta.Call = &toolCalls[0] // Simplified - only handle first tool call
			chunk.ToolDelta = toolDelta
		}
	}

	return chunk
}
