// Package agents provides the core agent framework and implementations
//
// This package contains the main agent system with the following structure:
// - definition.go: Core types, interfaces, and schemas
// - primary/: Main orchestrator agent for user interactions
// - sub-agents/: Specialized agents (summary, analyst, researcher, synthesis)
// - types.go: Legacy compatibility (will be removed in future)
package agents

// This file exists for backward compatibility.
// All type definitions have been moved to definition.go
// New code should import types from definition.go directly.
