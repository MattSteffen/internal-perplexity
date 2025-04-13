## Now

- Crawler

  - logging
  - Demo with 2 sources of documents

- Radchat

  - [x] Implement RAG with function call to get filters and queries
    - [x] Check what metadata should be returned
    - Find a way to get the metadata schema from the collection so it can properly filter on metadata
  - [ ] Clean up function
    - [ ] Remove Groq
    - [ ] Use langchain? and .with_structured_output
  - [ ] Use with bigger document collection
  - [ ] Implement citations
  - [ ] Implement event_emitter

- Implement security features for milvus
  - Add them to config files
  - Set up 2 repositories with different types of data
  - Test them based on user details in python script (not necessarily in ui)
  - upload different directory too
- Using basic_ollama as function in OI.
- Standardize configuration

- use cloudflared tunnel --url http://localhost:<desired-port> to expose the service to the internet
- maybe can do ingresses exposing http://ollama.localhost:5000 and similarly for the other services.

https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/htmx-go-basic-cursorrules-prompt-file/.cursorrules

# Crawler System Improvement TODOs

- [ ] Configuration (`config`)

  - [ ] Rename `collection_template.yaml` to something clearer like `default_collection_config.yaml`.
  - [ ] Restructure directory-specific config (`directories/*.yaml`) for better clarity (e.g., explicit `target_collection`, `collection_overrides`, `processing` sections).
  - [ ] Change `ConfigManager` to load and key directory configs by _name_ (e.g., "conference") instead of the `path` value within the file.
  - [ ] Implement configuration validation (e.g., using Pydantic or JSON Schema) to check for required fields and correct types.
  - [ ] Replace `print` with `logging` for error reporting in `ConfigManager`.
  - [ ] Ensure extractor configuration (`processing.extractors`) in the effective config correctly enables/disables specific readers in the `Extractor`.

- [ ] Architecture & Core Components (`architecture`)

  - [ ] Refactor `Extractor` to separate responsibilities: Coordinate extraction, select Reader based on config, delegate LLM metadata extraction to a separate step/component.
  - [ ] Modify `Extractor` to only initialize document readers (`self.doc_readers`) that are enabled in the effective configuration for the current directory.
  - [ ] Review `DocumentContent`: Separate raw text blocks from descriptive placeholders for images/tables to avoid duplication in `get_text()`.
  - [ ] Ensure `Crawler`, `DocumentProcessor`, `Extractor`, `Embedder`, and `Storage` classes maintain clear and distinct responsibilities after refactoring.

- [ ] Data Flow & Processing (`processing`)

  - [ ] **Critical:** Fix memory issue in `main.py` by implementing streaming/batch processing and insertion into `VectorStorage` instead of collecting all results first.
  - [ ] **Critical:** Implement document chunking logic (e.g., add a `Chunker` component).
  - [ ] Modify the processing pipeline (`DocumentProcessor` or `Crawler`) to iterate over chunks instead of whole documents.
  - [ ] Adapt embedding generation to work on a per-chunk basis.
  - [ ] Design and implement strategy for handling metadata with chunking (document-level vs. chunk-level metadata).
  - [ ] Implement logic to handle `metadata.extra_embeddings` configuration for embedding specific metadata fields.
  - [ ] Enhance file discovery (`_setup_filepaths`) to filter by configured extensions and potentially add exclusion patterns.
  - [ ] Decide and implement strategy for LLM metadata extraction in the context of chunking (once per doc or per chunk).

- [ ] Storage (`storage`)

  - [ ] Fix `VectorStorage.__enter__` to load an existing Milvus collection instead of dropping and recreating it. Dropping should be a separate, explicit operation.
  - [ ] Refine `build_collection_schema` in `vector_db.py`:
    - [ ] Ensure fields (`id`, `embedding`, `text`) are defined consistently and only once, respecting the schema config.
    - [ ] Align `maxLength` definitions for `VARCHAR` fields between the schema config and Milvus schema generation.
    - [ ] Document the JSON `array` to Milvus `VARCHAR` mapping or investigate native `ARRAY` type.
  - [ ] Align `MAX_DOC_LENGTH` constant in `vector_db.py` with schema definitions and actual Milvus capabilities/limits derived from config.
  - [ ] Re-evaluate and potentially implement an efficient duplicate checking mechanism in `insert_data` if needed (consider Milvus UPSERT).
  - [ ] Add robust error handling and logging for Milvus connection, insertion, and search operations.
  - [ ] Update `VectorStorage` schema/insertion logic to correctly store document-level and chunk-level metadata.

- [ ] Code Quality (`quality`)

  - [ ] Replace all `print` statements used for logging/status/errors with the standard `logging` module.
  - [ ] Implement comprehensive `try-except` blocks around I/O, network calls, and parsing. Log errors effectively.
  - [ ] Ensure consistent naming conventions (e.g., resolve `directory_name` vs. `dir_path`).
  - [ ] Standardize override structures in configuration files.
  - [ ] Refactor large methods into smaller, focused functions for readability.
  - [ ] Ensure all dependencies (including optional ones for readers) are listed correctly in `requirements.txt` (consider `extras_require`).
  - [ ] Review and update docstrings and comments to reflect code changes after refactoring.

- [ ] Examples & Documentation (`docs`)
  - [ ] Update `examples/process_documents.py` to reflect the refactored architecture and demonstrate key features (config, batching, chunking).
  - [ ] Update `README.md` and `Crawler.md` to accurately describe the final architecture, configuration, and usage.
  - [ ] Update `processing/processing.md` based on the refactored structure.
  - [ ] Update `storage/db.md` to reflect the final `VectorStorage` implementation and configuration.

# MVP

## General

- [ ] Data sources

  - [ ] What local data do I download
    - [ ] General conference talks
    - [ ] Scriptures in chapters
  - [ ] What search apis do I use
    - [ ] brave search
  - [ ] What crawling do I do?
    - [ ] Levels deep of the link graph
    - [ ] How many links to follow
  - [ ] How to manage citations

- [x] Make repo public
- If using openwebui, include instructions for it's deployment and special config
  - Include pipelines

## Frontend

## Backend

- [x] Decide framework
  - Python then after MVP -> Go
- [ ] Create good async enabled load balancer

## Future

- [ ] After MVP
  - [ ] Refactor backend into Go
  - [ ] Create the small model fine tuning and test time inference, then run locally
