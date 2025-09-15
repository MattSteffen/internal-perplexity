package api

// Tool represents a tool that the model can call.
type ToolDefinition struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// Function represents a function that the model can call.
type FunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"` // or json.rawmessage
	Strict      *bool          `json:"strict,omitempty"`     // Whether to enable strict schema adherence
}

// ToolCall represents a tool call from the assistant.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ResponseFormat encapsulates the various options for the model's response format.
// It uses an interface to represent the 'oneOf' structure found in the OpenAPI spec.
type ResponseFormat struct {
	Type       string                    `json:"type"`                  // "json_schema" or "text" or "json_object"
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"` // Only for type "json_schema"
}

// ResponseFormatJSONSchema represents the structured output JSON Schema response format.
// This allows you to define a specific JSON schema the model must adhere to.
type ResponseFormatJSONSchema struct {
	Type       string                `json:"type"`        // Should be "json_schema"
	Name       string                `json:"name"`        // The name of the response format
	JSONSchema *JSONSchemaProperties `json:"json_schema"` // Nested object for schema definition
}

// JSONSchemaProperties defines the properties for a JSON Schema response format.
type JSONSchemaProperties struct {
	Description string         `json:"description,omitempty"` // A description for the model
	Name        string         `json:"name"`                  // The name of the response format, must be unique
	Schema      map[string]any `json:"schema"`                // The actual JSON Schema definition or json.rawmessage
	Strict      *bool          `json:"strict,omitempty"`      // Whether to enable strict schema adherence
}
