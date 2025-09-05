# LLM Package

This package provides the core LLM orchestration framework following agent system design patterns.

## Architecture Overview

### Design Patterns Implemented
- **Vertical vs Horizontal Task Decomposition**: Sequential pipeline and MapReduce patterns
- **Specialization Patterns**: Capability, domain, and model-based specialization
- **Optimizations**: Caching, parallel execution, stateless design, explicit task segregation, monitoring

### Directory Structure
- `models/` - LLM provider integrations and shared types
- `agents/` - Agent implementations (main and sub-agents)
- `tools/` - Deterministic tools with predictable I/O
- `services/` - Orchestration frameworks and communication layers

## Key Components

### Models
Provider-specific integrations with standardized interfaces for:
- Chat completion
- Streaming responses
- Token counting
- Model capabilities

### Agents
- **Main Agents**: Primary orchestrators with access to sub-agents and general tools
- **Sub-Agents**: Specialized agents with tool access, returning results + stats (success, tokens, timing)

### Tools
Deterministic tools with:
- JSON schema definitions
- Predictable input/output
- ≤1 LLM call per execution
- Strict workflow enforcement

### Services
- **Agent Manager**: Orchestrates agent communication and task distribution
- **Task Orchestration**: Manages task execution and result aggregation

## Design Principles
1. **Stateless Design**: Keep everything but main agent stateless
2. **Caching**: Prompt hash-based caching with TTL
3. **Parallel Execution**: Independent agents run concurrently
4. **Monitoring**: Built-in metrics from day one
5. **Explicit Tasks**: Clear definitions and success criteria
