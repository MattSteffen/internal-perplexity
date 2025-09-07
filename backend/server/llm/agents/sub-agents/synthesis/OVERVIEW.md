# Synthesis Agent

The Synthesis Agent is a specialized sub-agent that combines and aggregates outputs from multiple sub-agents into coherent, comprehensive responses. It serves as the final aggregation layer in complex multi-agent workflows, resolving conflicts and creating unified outputs.

## Purpose

The Synthesis Agent handles the critical task of integrating information from multiple sources:
- **Multi-Source Integration**: Combine outputs from different agents and tools
- **Conflict Resolution**: Identify and resolve contradictions between sources
- **Structured Synthesis**: Create organized, comprehensive responses
- **Confidence Assessment**: Provide confidence scores for synthesized information
- **Report Generation**: Produce unified reports and analyses

## Features

- **Intelligent Integration**: Smart combination of heterogeneous information sources
- **Conflict Detection**: Automated identification of conflicting information
- **Structured Output**: Consistent formatting with clear sections and components
- **Confidence Scoring**: Quantitative assessment of synthesis reliability
- **Metadata Preservation**: Maintain source attribution and processing information
- **Flexible Formatting**: Adaptable output formats for different use cases

## Architecture

### File Structure
- `definition.go`: Agent schema, types, and construction
- `synthesis.go`: Core synthesis logic and LLM integration
- `synthesis_test.go`: Unit tests for deterministic functionality
- `OVERVIEW.md`: Documentation and usage examples

### Processing Flow
1. **Input Validation**: Validate source inputs and synthesis parameters
2. **Source Analysis**: Analyze each input source for content and structure
3. **Conflict Resolution**: Identify and resolve conflicting information
4. **Content Integration**: Combine information into coherent narrative
5. **Structure Generation**: Create organized output with clear sections
6. **Confidence Assessment**: Calculate reliability scores for the synthesis

## Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | object | Yes | Map of input sources (key-value pairs) |
| `instructions` | string | No | Specific synthesis instructions |
| `format` | string | No | Desired output format (default: "comprehensive") |
| `context` | object | No | Additional context information |

### Input Sources
```json
{
  "inputs": {
    "summary_agent": "Document summary content...",
    "analyst_agent": {
      "content": "Analysis results...",
      "metadata": {"confidence": 0.85}
    },
    "researcher_agent": "Research findings..."
  }
}
```

## Output Schema

```json
{
  "synthesis": "Integrated synthesis combining all input sources...",
  "structure": {
    "executive_summary": "High-level overview...",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "sections": {
      "analysis": "Detailed analysis content...",
      "conclusions": "Final conclusions..."
    }
  },
  "confidence": 0.87,
  "sources_used": ["summary_agent", "analyst_agent", "researcher_agent"],
  "metadata": {
    "input_count": 3,
    "synthesis_type": "comprehensive",
    "processing_time": "2024-01-15T10:30:00Z",
    "conflicts_resolved": 1
  }
}
```

## Usage Examples

### Basic Multi-Agent Synthesis

```go
synthesisAgent := synthesis.NewSynthesisAgent()

result, err := synthesisAgent.Execute(ctx, &agents.AgentInput{
    Data: map[string]interface{}{
        "inputs": map[string]interface{}{
            "summary_result": "Document summary from summary agent...",
            "analysis_result": "Data analysis from analyst agent...",
            "research_result": "Research findings from researcher agent...",
        },
        "instructions": "Create a comprehensive report combining all findings",
        "format": "structured_report",
    },
}, llmProvider)

if result.Success {
    synthesis := result.Content.(map[string]interface{})
    fmt.Printf("Synthesis: %s\n", synthesis["synthesis"])
    fmt.Printf("Confidence: %.2f\n", synthesis["confidence"])
}
```

### API Usage with curl

```bash
curl -X POST http://localhost:8080/agents/synthesis \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key-here" \
  -d '{
    "input": {
      "inputs": {
        "summary_agent": "Document summary content...",
        "analyst_agent": "Analysis results...",
        "researcher_agent": "Research findings..."
      },
      "instructions": "Integrate findings into executive summary",
      "format": "executive_report"
    },
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "success": true,
  "content": {
    "synthesis": "Integrated analysis reveals key patterns across all sources...",
    "structure": {
      "executive_summary": "High-level overview of integrated findings",
      "key_findings": ["Finding 1", "Finding 2"],
      "recommendations": ["Recommendation 1"]
    },
    "confidence": 0.89,
    "sources_used": ["summary_agent", "analyst_agent", "researcher_agent"],
    "metadata": {
      "input_count": 3,
      "synthesis_type": "executive_report",
      "conflicts_resolved": 0
    }
  },
  "tokens_used": 450,
  "duration": "2.3s"
}
```

## Input/Output Validation

### Input Validation Rules

**Required Fields:**
- `inputs`: Must be present and non-empty object
- Each input source must have valid content

**Optional Fields:**
- `instructions`: Must be string if provided
- `format`: Must be valid format type if provided
- `context`: Free-form object, no validation

### Validation Examples

**Valid Input:**
```json
{
  "inputs": {
    "source1": "Content 1",
    "source2": {"content": "Content 2"}
  },
  "instructions": "Combine sources",
  "format": "comprehensive"
}
```

**Invalid Input - Empty Inputs:**
```json
{
  "inputs": {}
}
```
Returns: `ValidationError{Field: "inputs", Message: "at least one input source is required"}`

### Output Validation Rules

**Required Fields:**
- `synthesis`: Main synthesis text (string)
- `structure`: Structured components (object)
- `confidence`: Confidence score (0.0-1.0)
- `sources_used`: Array of source names
- `metadata`: Processing metadata (object)

## Synthesis Formats

### Comprehensive Format
- Full synthesis with detailed analysis
- Complete structured breakdown
- Extensive metadata

### Executive Format
- Concise executive summary
- High-level key findings
- Essential recommendations only

### Technical Format
- Detailed technical analysis
- Code examples and implementation details
- Technical specifications

### Research Format
- Academic-style synthesis
- Citation and source attribution
- Methodological details

## Conflict Resolution

The Synthesis Agent automatically handles conflicting information:

### Conflict Detection
- **Direct Contradictions**: Explicitly opposing statements
- **Statistical Conflicts**: Differing numerical results
- **Contextual Conflicts**: Information valid in different contexts

### Resolution Strategies
- **Evidence-Based**: Prioritize sources with stronger evidence
- **Contextual**: Consider situational validity
- **Consensus**: Find common ground between sources
- **Uncertainty**: Flag unresolved conflicts with confidence penalties

### Example Conflict Resolution
```
Input Sources:
- Source A: "Temperature increased by 2°C"
- Source B: "Temperature increased by 1.5°C"

Resolution: "Temperature increased by 1.75-2°C (range reflects source variation)"
Confidence: 0.75 (reduced due to conflict)
```

## Capabilities

The Synthesis Agent provides these capabilities:

```go
capabilities := synthesisAgent.GetCapabilities()
// Returns:
[
    {
        "name": "multi_source_synthesis",
        "description": "Combine and integrate information from multiple sources"
    },
    {
        "name": "conflict_resolution",
        "description": "Resolve conflicts and contradictions between sources"
    },
    {
        "name": "structured_reporting",
        "description": "Generate structured, comprehensive reports"
    },
    {
        "name": "confidence_assessment",
        "description": "Provide confidence scores for synthesized information"
    }
]
```

## Statistics and Monitoring

```go
stats := synthesisAgent.GetStats()
// Returns:
{
    "total_executions": 150,
    "average_duration": "2.1s",
    "success_rate": 0.97,
    "total_tokens": 45200,
    "sub_agents_used": 150
}
```

### Metrics Tracked
- **Total Executions**: Number of synthesis operations performed
- **Average Duration**: Mean processing time per synthesis
- **Success Rate**: Percentage of successful syntheses
- **Token Usage**: Total tokens consumed by LLM calls
- **Sources Processed**: Average number of input sources per synthesis

## Configuration

### System Prompts
The agent uses specialized prompts for different synthesis types:

```go
prompt := synthesisAgent.GetSystemPrompt()
// Returns context-aware synthesis prompt
```

### Processing Constraints
- **Maximum Input Sources**: 10 sources per synthesis
- **Maximum Output Length**: 5000 tokens
- **Processing Timeout**: 120 seconds
- **Minimum Confidence**: 0.0 (no floor, but flagged)

## Error Handling

### Common Errors

**Input Validation Errors:**
```json
{
  "success": false,
  "error": "inputs field is required",
  "duration": "0.001s"
}
```

**Processing Errors:**
```json
{
  "success": false,
  "error": "failed to parse synthesis result: invalid JSON",
  "duration": "2.3s"
}
```

**LLM Errors:**
```json
{
  "success": false,
  "error": "LLM synthesis failed: connection timeout",
  "duration": "30.5s"
}
```

## Best Practices

### Input Preparation
1. **Consistent Naming**: Use descriptive, consistent source names
2. **Content Quality**: Ensure input sources contain substantial content
3. **Format Consistency**: Standardize input formats when possible

### Instructions
1. **Clear Objectives**: Provide specific synthesis goals
2. **Format Specifications**: Specify desired output structure
3. **Priority Setting**: Indicate which sources are most important

### Performance Optimization
1. **Source Limiting**: Don't exceed recommended source limits
2. **Content Filtering**: Remove irrelevant or redundant content
3. **Batch Processing**: Group similar syntheses for efficiency

## Implementation Details

### Core Components
- **Input Validator**: Schema-based input validation
- **Source Analyzer**: Content and structure analysis
- **Conflict Resolver**: Automated conflict detection and resolution
- **Content Integrator**: Intelligent content combination
- **Structure Generator**: Organized output formatting
- **Confidence Calculator**: Quantitative reliability assessment

### Processing Pipeline
1. **Validation Phase**: Input schema validation
2. **Analysis Phase**: Source content analysis
3. **Integration Phase**: Content combination and conflict resolution
4. **Structuring Phase**: Output organization and formatting
5. **Assessment Phase**: Confidence scoring and metadata generation

## Integration with Primary Agent

The Synthesis Agent integrates seamlessly with the Primary Agent:

```go
// Primary agent automatically calls synthesis for complex queries
primaryAgent := primary.NewPrimaryAgent(map[string]agents.Agent{
    "summary": summaryAgent,
    "analyst": analystAgent,
    "synthesis": synthesisAgent,
})

result, err := primaryAgent.Execute(ctx, &agents.AgentInput{
    Query: "Analyze this data and create a comprehensive report",
}, llmProvider)
```

The Primary Agent automatically:
1. Decomposes the query into sub-agent calls
2. Executes summary and analyst agents
3. Calls synthesis agent to combine results
4. Returns unified response

## Future Enhancements

### Planned Features
- **Advanced Conflict Resolution**: Machine learning-based conflict detection
- **Dynamic Formatting**: Context-aware output format selection
- **Source Weighting**: Importance-based source prioritization
- **Interactive Synthesis**: Human-in-the-loop refinement capabilities
- **Multi-Modal Outputs**: Support for charts, graphs, and visualizations

### Integration Points
- **External Validators**: Integration with fact-checking services
- **Citation Systems**: Automated source citation and referencing
- **Quality Metrics**: Advanced quality assessment algorithms
