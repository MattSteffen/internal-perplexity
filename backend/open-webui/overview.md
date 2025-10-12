# Open-WebUI

This directory contains Open-WebUI pipeline functions for integration with the internal perplexity system. These pipelines enable document retrieval and AI-powered responses with citations.

## Agentic Integration

Backend-controlled UI compatible flow for streaming responses and citations:
https://docs.openwebui.com/tutorials/integrations/backend-controlled-ui-compatible-flow

## Files

### radchat/radchat.py

Production-ready RAG pipeline for querying internal R&D documents (IRADs).

**Architecture**:

- **Hybrid Search**: Combines dense (text embeddings) and sparse (BM25-style) embeddings for document retrieval
- **Agentic Loop**: LLM can iteratively call search tools to gather information
- **Streaming Citations**: Citations emitted in real-time as documents are retrieved
- **Role-Based Access Control**: Automatically filters documents based on user security groups

**Key Components**:

- `connect_milvus()`: Establishes connection to vector database
- `get_embedding()` / `embed_texts()`: Generate embeddings via Ollama
- `perform_search()`: Hybrid search with dense + sparse embeddings
- `perform_query()`: Metadata-only filtering without embeddings
- `retrieve_documents()`: Testable wrapper for document retrieval
- `generate_response()`: LLM response generation with tool calling
- `render_document()`: Unified markdown rendering with optional full text
- `consolidate_documents()`: Merges chunks from same source document
- `build_citations()`: Creates citation objects for UI display
- `build_response()`: Final response assembly with citations

**Data Models**:

- `MilvusDocumentMetadata`: Structured metadata (title, author, date, keywords)
- `MilvusDocument`: Complete document representation with embeddings

**System Prompt**: Follows Anthropic's context engineering format with clear role, instructions, constraints, and examples.

### xmchat.py

Legacy version, superseded by radchat.py.

### xmchat_v0.py, xmchat_v1.py

Earlier experimental versions, kept for reference.
