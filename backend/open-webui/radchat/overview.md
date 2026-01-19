# Radchat - Open WebUI Pipe for RAG Document Retrieval

This directory contains the Radchat Open WebUI pipe function that provides RAG-based document retrieval using the crawler package.

## Files

### radchat.py

The main pipe implementation for Open WebUI that integrates with the crawler package for document retrieval and search.

**Key Components:**

- **RadchatConfig**: Simplified Pydantic configuration for Ollama LLM settings (base URL, model names, timeouts, context length, max tool calls) and default Milvus connection parameters.

- **connect_database()**: Connects to Milvus using the crawler's `DatabaseClient`. Creates an embedder via `get_embedder()` and retrieves the `CollectionDescription` which contains the LLM prompt and metadata schema.

- **perform_search()**: Wrapper around `DatabaseClient.search()` that performs hybrid search (dense + sparse vectors) with security group filtering and RRF ranking.

- **render_document()**: Renders a `SearchResult` to markdown format for display in the LLM context or citations. Extracts metadata (title, authors, date, source) and optionally includes full document text.

- **consolidate_results()**: Groups `SearchResult` objects by `document_id`, combines text chunks in order, merges unique keywords/authors, and sorts by relevance score.

- **build_system_prompt()**: Constructs the system prompt using `collection_desc.llm_prompt` (which contains schema info and filtering examples) along with preliminary context and document metadata summaries.

- **Pipe class**: The Open WebUI pipe that:
  - Accepts `UserValves` for collection name and Milvus credentials
  - Connects to the database via `connect_database()`
  - Performs initial search based on user query
  - Emits citations via `__event_emitter__`
  - Runs an agentic loop with tool calling support for additional searches
  - Streams responses using Ollama's chat API

**Dependencies:**
- `crawler` package: Provides `DatabaseClient`, `SearchResult`, `DatabaseDocument`, `CollectionDescription`, `get_db`, `get_embedder`
- `ollama`: For LLM inference and embeddings
- `pydantic`: For configuration validation

**Data Flow:**
1. User query arrives via Open WebUI
2. Pipe connects to Milvus using crawler's `DatabaseClient`
3. Initial search retrieves relevant documents
4. System prompt is built using `collection_desc.llm_prompt`
5. LLM generates response, optionally calling the search tool
6. Citations are emitted in real-time
7. Final response with citations is returned

## Design Decisions

- **Crawler Integration**: Uses the crawler package's `DatabaseClient` for all Milvus operations instead of raw `pymilvus` calls. This ensures consistent security group handling, hybrid search implementation, and collection description parsing.

- **Collection Description**: Leverages `CollectionDescription.llm_prompt` which contains auto-generated instructions for metadata filtering based on the collection's schema. This removes the need for hardcoded filter examples.

- **Simplified Config**: Ollama-specific settings (not part of crawler) are kept in `RadchatConfig`, while database and embedding configs use crawler's `DatabaseClientConfig` and `EmbedderConfig`.

- **SearchResult Type**: Uses crawler's `SearchResult` wrapper which contains `document: DatabaseDocument`, `distance: float`, and `score: float` for consistent result handling.
