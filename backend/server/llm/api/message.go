package api

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Role is the role of a chat message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ChatCompletionMessage is the interface all chat messages implement.
type ChatCompletionMessage interface {
	GetRole() Role
}

// -------------------- Message Types --------------------

// ChatCompletionSystemMessage represents a system message.
type ChatCompletionSystemMessage struct {
	Content string `json:"content"`
	Role    Role   `json:"role"`
	Name    string `json:"name,omitempty"`
}

func (m ChatCompletionSystemMessage) GetRole() Role { return m.Role }

// ChatCompletionUserMessage represents a user message.
// Note: content can be a plain string OR an array of content parts.
type ChatCompletionUserMessage struct {
	Content UserMessageContent `json:"content"`
	Role    Role               `json:"role"`
	Name    string             `json:"name,omitempty"`
}

func (m ChatCompletionUserMessage) GetRole() Role { return m.Role }

// ChatCompletionAssistantMessage represents an assistant message.
type ChatCompletionAssistantMessage struct {
	Content   string     `json:"content,omitempty"`
	Role      Role       `json:"role"`
	Name      string     `json:"name,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

func (m ChatCompletionAssistantMessage) GetRole() Role { return m.Role }

// ChatCompletionToolMessage represents a tool message.
type ChatCompletionToolMessage struct {
	Content    string `json:"content"`
	Role       Role   `json:"role"`
	ToolCallID string `json:"tool_call_id"`
}

func (m ChatCompletionToolMessage) GetRole() Role { return m.Role }

// -------------------- User Content (string or parts[]) --------------------

// UserMessageContent can be either a string or a list of structured parts.
type UserMessageContent struct {
	// If String != nil, the original JSON was a string.
	String *string

	// If Parts != nil, the original JSON was an array of parts.
	Parts []ChatCompletionMessageContentPart
}

func (c *UserMessageContent) UnmarshalJSON(data []byte) error {
	d := bytes.TrimSpace(data)
	if len(d) == 0 {
		c.String = nil
		c.Parts = nil
		return nil
	}

	switch d[0] {
	case '"':
		var s string
		if err := json.Unmarshal(d, &s); err != nil {
			return err
		}
		c.String = &s
		c.Parts = nil
		return nil
	case '[':
		var raws []json.RawMessage
		if err := json.Unmarshal(d, &raws); err != nil {
			return err
		}
		parts := make([]ChatCompletionMessageContentPart, 0, len(raws))
		for _, r := range raws {
			p, err := unmarshalContentPart(r)
			if err != nil {
				return err
			}
			parts = append(parts, p)
		}
		c.String = nil
		c.Parts = parts
		return nil
	default:
		return fmt.Errorf(
			"user content must be string or array of content parts",
		)
	}
}

func (c UserMessageContent) MarshalJSON() ([]byte, error) {
	if c.String != nil && (c.Parts == nil || len(c.Parts) == 0) {
		return json.Marshal(*c.String)
	}
	// Marshal parts array form
	return json.Marshal(c.Parts)
}

// -------------------- Content Parts --------------------

// ChatCompletionMessageContentPart is a discriminated interface for user content parts.
type ChatCompletionMessageContentPart interface {
	GetType() string
}

// ChatCompletionMessageContentPartText represents a text content part.
type ChatCompletionMessageContentPartText struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

func (p ChatCompletionMessageContentPartText) GetType() string { return p.Type }

// ChatCompletionMessageContentPartImage represents an image content part.
type ChatCompletionMessageContentPartImage struct {
	Type     string   `json:"type"` // "image_url"
	ImageURL ImageURL `json:"image_url"`
}

func (p ChatCompletionMessageContentPartImage) GetType() string { return p.Type }

// ImageURL represents the URL and detail of an image.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // e.g., "low", "high", "auto"
}

// unmarshalContentPart inspects the "type" discriminator and decodes into the
// correct concrete content part type.
func unmarshalContentPart(data []byte) (ChatCompletionMessageContentPart, error) {
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Type {
	case "text":
		var p ChatCompletionMessageContentPartText
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case "image_url":
		var p ChatCompletionMessageContentPartImage
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	default:
		return nil, fmt.Errorf("unknown content part type: %q", probe.Type)
	}
}

// -------------------- Helpers to (un)marshal lists of messages --------------------

// UnmarshalChatMessages parses a JSON array of mixed-role messages into
// []ChatCompletionMessage with the correct concrete types.
func UnmarshalChatMessages(data []byte) ([]ChatCompletionMessage, error) {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return nil, err
	}

	out := make([]ChatCompletionMessage, 0, len(raws))
	for _, r := range raws {
		var head struct {
			Role Role `json:"role"`
		}
		if err := json.Unmarshal(r, &head); err != nil {
			return nil, err
		}
		switch head.Role {
		case RoleSystem:
			var m ChatCompletionSystemMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			out = append(out, m)
		case RoleUser:
			var m ChatCompletionUserMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			out = append(out, m)
		case RoleAssistant:
			var m ChatCompletionAssistantMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			out = append(out, m)
		case RoleTool:
			var m ChatCompletionToolMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			out = append(out, m)
		default:
			return nil, fmt.Errorf("unknown role: %q", head.Role)
		}
	}
	return out, nil
}

// MarshalChatMessages serializes []ChatCompletionMessage to a JSON array.
// Since each element is a concrete struct under the interface, the default
// json.Marshal behavior is sufficient.
func MarshalChatMessages(msgs []ChatCompletionMessage) ([]byte, error) {
	// The json package will marshal each element using its concrete type.
	return json.Marshal(msgs)
}
