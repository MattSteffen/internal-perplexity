# ARCHITECTURE.md

## Researcher Agent MVP – System Architecture Overview

### Overview

This document describes the high-level design and component interactions of the Researcher Agent MVP. The system is designed with clear modular boundaries to promote scalability, maintainability, and future extensibility.

### Directory Structure

researcher_agent_mvp/
├── config/
│ ├── agent_config.yaml # Main configuration (LLM settings, Milvus connection, etc.)
│ ├── metadata_rules.yaml # Required metadata fields and validation rules
│ └── prompt_templates.yaml # Query and sub-query prompt templates for LLM interactions
├── docs/
│ └── ARCHITECTURE.md # Overview of system design and component interactions
├── src/
│ ├── **init**.py
│ ├── main.py # Application entry point
│ ├── researcher_agent.py # Main agent orchestrator
│ ├── worker.py # Worker (recursive sub-agent) implementation
│ ├── query_processor.py # Processes and decomposes queries
│ ├── result_aggregator.py # Aggregates and formats worker results
│ ├── vectorstore_connector.py # Milvus (or other vectorstore) interface
│ ├── security_context.py # Security context and filtering logic
│ └── llm_interface.py # Abstraction over LLM/embedding API calls
├── utils/
│ ├── **init**.py
│ ├── logging_util.py # Central logging configuration and utilities
│ ├── template_util.py # Query template processing helpers
│ ├── validation_util.py # Functions to validate metadata and results
│ └── formatting_util.py # Helper routines for formatting responses and citations
└── tests/
├── **init**.py
└── test_researcher_agent.py # Unit/integration tests for core components

### Diagram

```mermaid
flowchart TD
    %% Configuration Files
    CF[Configuration Files<br/>(agent_config.yaml,<br/>prompt_templates.yaml,<br/>metadata_rules.yaml)]

    %% Main Entry Point
    A[main.py]

    %% Core Components
    RA[ResearcherAgent]
    QP[QueryProcessor]
    W[Worker (ResearchWorker)]
    RAgg[ResultAggregator]

    %% External Interfaces
    LLM[LLM Interface]
    VC[Vectorstore Connector]
    SC[Security Context]

    %% Flow Connections
    A -->|Loads Config| CF
    A -->|Initializes| RA
    RA -->|Parses Query| QP
    RA -->|Spawns Workers| W
    W -->|Executes Sub-query| W_Result[Worker Result]
    RA -->|Aggregates Results| RAgg
    RAgg -->|Formats Final Summary| Final[Final Summary]

    %% Dependency Arrows for External Interfaces
    RA -- Uses --> LLM
    RA -- Uses --> VC
    RA -- Uses --> SC
```

### Core Components

#### A. Configuration Files (YAML)

- **agent_config.yaml:**  
  Contains key parameters such as LLM settings, Milvus connection details, worker recursion limits, security tokens, and logging preferences.

- **metadata_rules.yaml:**  
  Defines required metadata fields (e.g., author, publication_date, source) along with validation rules and scoring weights to assess document quality.

- **prompt_templates.yaml:**  
  Provides LLM prompt templates for query decomposition, document relevance scoring, and summary synthesis with citations.

#### B. Core Modules (in `src/`)

- **main.py:**  
  The entry point that loads configurations, initializes logging, creates the main ResearcherAgent instance, and starts the research process.

- **researcher_agent.py:**  
  Orchestrates the research process by managing worker agents, aggregating their results, and enforcing recursion and security rules.

- **worker.py:**  
  Implements the `ResearchWorker` class to process sub-queries, evaluate document relevance, and report findings back to the main agent.

- **query_processor.py:**  
  Parses raw user queries into actionable sub-queries using defined prompt templates.

- **result_aggregator.py:**  
  Merges and formats the findings from all workers into a coherent summary with inline citations.

- **vectorstore_connector.py:**  
  Manages connectivity and querying of the Milvus vector store.

- **security_context.py:**  
  Enforces security policies by validating requests and filtering sensitive data.

- **llm_interface.py:**  
  Provides an abstraction layer for interacting with LLM services and embedding APIs.

#### C. Utility Modules (in `utils/`)

- **logging_util.py:**  
  Centralizes logging configuration for consistent log messaging.

- **template_util.py:**  
  Contains helper functions for loading and processing YAML prompt templates.

- **validation_util.py:**  
  Validates metadata and search results against defined rules.

- **formatting_util.py:**  
  Formats the final output, including the assembly of summaries with citations.

#### D. Testing (in `tests/`)

- **test_researcher_agent.py:**  
  Unit and integration tests to verify that each component (agent orchestration, vectorstore connectivity, security enforcement, etc.) functions as expected.

### Design Considerations

- **Performance:**  
  Efficient connection pooling, caching for repeated queries, and basic asynchronous worker execution.

- **Security:**  
  Role-based access control, request validation, and sensitive data filtering.

- **Maintainability:**  
  Clear module boundaries, dependency injection for testability, and comprehensive inline documentation.

- **Scalability:**  
  Controlled recursion limits and modular design enable future extensions and integration of additional services.

### Final Notes

Each module is documented with detailed docstrings and inline comments. Configuration files include inline examples and comments to guide setup and customization. The overall design emphasizes clear separation of concerns to ease maintenance and future development.
