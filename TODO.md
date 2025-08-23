# TODO

## General

- Test the tools

## Deployment

## Other

- use cloudflared tunnel --url http://localhost:<desired-port> to expose the service to the internet
- maybe can do ingresses exposing http://ollama.localhost:5000 and similarly for the other services.

https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/htmx-go-basic-cursorrules-prompt-file/.cursorrules

## Crawler

- Unify config flow and fix converter/LLM wiring

  1. Update `crawler/src/processing/llm.py` `LLMConfig.from_dict` to accept both `model` and `model_name`; set sensible defaults for `base_url`, `ctx_length`, and `default_timeout`.
  2. In `crawler/src/crawler.py`, stop passing the entire `CrawlerConfig` dataclass to converters. Instead, construct a plain dict for the converter with a `type` key (default: `"markitdown"`) and a `vision_llm` block derived from `config.vision_llm`.
  3. Use `create_converter(conv_type, conv_cfg)` to instantiate the converter.
  4. Update `crawler/Crawler.md` example config to reflect keys actually used (`embeddings.model`, `llm.model` or `llm.model_name`, `vision_llm.model`).

- Sanitize and validate metadata before insert

  1. In `crawler/src/crawler.py`, define `RESERVED = {"document_id","chunk_index","source","text","text_embedding","text_sparse_embedding","metadata","metadata_sparse_embedding"}`.
  2. Add `sanitize_metadata(md: dict) -> dict` to drop reserved keys; call it when constructing `DatabaseDocument`.
  3. Optional: add `jsonschema` validation. If present, validate `metadata` against `config.metadata_schema`; on failure, log and fall back to `{}`.

- Token-aware chunking with overlap + batch embeddings

  1. Replace `Extractor.chunk_text` with token-aware chunking using `tiktoken` (fallback to word-based if unavailable). Support `chunk_tokens` and `overlap` args.
  2. Extend `Embedder` interface with `embed_batch(texts: List[str]) -> List[List[float]]`.
  3. Implement `embed_batch` in `OllamaEmbedder` via `OllamaEmbeddings.embed_documents`.
  4. In `Crawler.crawl`, call `embed_batch(chunks)` and zip chunks with vectors to build documents.
  5. Measure throughput vs. the old per-chunk embedding (log timings for N docs).

- Improve duplicate detection and remove duplicate Milvus implementation

  1. Delete dead/duplicate `crawler/src/storage/milvus.py` (keep `milvus_client.py`).
  2. In `MilvusDB` (`crawler/src/storage/milvus_client.py`), add `_existing_chunk_indexes(source) -> set[int]` to fetch all existing `chunk_index` for a source in one query.
  3. In `insert_data`, group items by `source`; prefetch existing indexes per source; skip duplicates without per-chunk queries.
  4. Keep the pre-flight `check_duplicate(filepath, 0)` in `Crawler.crawl` to skip whole documents early.
  5. Add unit tests for duplicate handling (new source vs. existing chunks).

- Robust error handling and provider-agnostic logging

  1. Replace all `print` in `crawler/src/crawler.py` with `logging` (module-level logger). Do not reference provider names in messages.
  2. Wrap per-file processing in `try/except Exception` to continue on failure; include `logger.exception` with the filepath.
  3. Track counters for processed/skipped/failed and log a final summary.

- Versioned cache invalidation for processed documents

  1. Add `_cache_key(filepath)` using: file mtime, converter class, extractor class, `metadata_schema` hash, and chunking params.
  2. Store `{ key, text, metadata }` in the temp JSON; on load, compare keys and invalidate if mismatch.
  3. Document the cache behavior in `crawler/Crawler.md`.

- Embedder dimension determination (avoid probe call when known)

  1. Extend `EmbedderConfig` with optional `dimension: int | None`.
  2. In `OllamaEmbedder.get_dimension`, return the configured dimension if set; otherwise probe once and cache.
  3. Fix `processing/embeddings.py` test to construct `OllamaEmbedder(EmbedderConfig(...))` correctly.

- Align benchmarks with stored schema

  1. In `crawler/src/storage/milvus_benchmarks.py`, remove sparse search requests unless sparse vectors are actually written; otherwise implement writing `text_sparse_embedding` and `metadata_sparse_embedding` (choose one approach).
  2. Match on `source` (or `document_id` if added) instead of `id` when evaluating placement.
  3. Ensure `output_fields` requested actually exist (e.g., `title`, `author` only if present in metadata/schema).
  4. Update any plots/outputs to reference the aligned fields.

- Documentation and samples

  1. Update `crawler/Crawler.md` to reflect the new config schema and cache behavior.
  2. Add a config snippet showing converter `type` and `vision_llm` block; note supported providers.

- Tests

  1. Add unit tests for: `sanitize_metadata`, token chunking overlap behavior, and `MilvusDB.insert_data` duplicate filtering (mock client).
  2. Add an integration test that runs `Crawler.crawl` on 1–2 small files with temp storage (or mocked DB) and asserts inserted chunk counts and metadata.

- Optional benchmark question generation during metadata extraction

  1. Config: add `utils.generate_benchmark_questions: bool = false` and `utils.num_benchmark_questions: int = 3` to `CrawlerConfig.from_dict`.
  2. Schema: in `storage/milvus_utils.py:create_schema`, add a default field `benchmark_questions` (var-length string or JSON string) to the base schema.
  3. Prompt: add a helper in `processing/extractor.py`:
     - `def generate_benchmark_questions(llm: LLM, text: str, n: int) -> list[str]` that prompts: "Return exactly N JSON strings, each a question answerable by the document. Respond with a JSON array of strings only." Parse and return list.
  4. Extraction: in `BasicExtractor.extract_metadata`, if `config.utils.generate_benchmark_questions` is true, call the helper with `n = utils.num_benchmark_questions`, attach as `{"benchmark_questions": [...]} ` to the metadata dict (after sanitization in crawler).
  5. Storage: ensure `MilvusDB.insert_data` copies `benchmark_questions` from `item.metadata` into both flattened fields and `metadata` JSON string; consider only attaching to `chunk_index == 0` to avoid duplication.
  6. Benchmark use: modify `storage/milvus_benchmarks.py` to optionally use stored `benchmark_questions` per source when `generate_queries=False`.

- Architecture to support personalized crawlers and pre-parsed inputs

  1. Profiles: introduce `CrawlerProfile` (new module) that bundles `converter`, `extractor`, `chunker params`, and `post_processors`. Provide built-ins: `DefaultProfile`, `PreParsedProfile` (uses `NoOpConverter` that reads provided Markdown), `PDFRichProfile` (uses `PyMuPDFConverter`).
  2. Routing: add `ProfileRouter` that selects a profile per file based on rules (mimetype, file extension, path regex). Wire into `Crawler.crawl` to pick the profile before processing each file.
  3. Hooks: add optional hooks on `Crawler` or `Profile` (`before_convert`, `after_convert`, `after_metadata`, `before_insert`) for custom behavior without subclassing.
  4. Factories: expose `get_extractor(config, schema, llm)` and `get_converter(config)` (already have `create_converter`) to keep `Crawler` generic; stop hard-coding `BasicExtractor`.
  5. Config: allow `profiles` section in config with named profiles and routing rules; support `converter.type = "none"` to skip conversion for pre-parsed content.
  6. Example: add a `PreParsedCrawler` example that feeds `.md` files directly and uses a schema tailored to that corpus.

- Update READMEs and examples
  1. `crawler/Crawler.md`: update the workflow with profiles, router, cache versioning; include config examples for `markitdown`, `docling`, and `pymupdf`, plus the `generate_benchmark_questions` flag and `num_benchmark_questions`.
  2. `crawler/src/processing/processing.md`: document token-aware chunking, batch embeddings, and the new question-generation helper, with short code snippets.
  3. `crawler/src/storage/db.md`: document the base schema fields, especially `benchmark_questions`, duplicate detection strategy, and bulk duplicate checks; align examples with actual inserted fields.
  4. Top-level `README.md`: add a minimal “Quickstart: crawl a folder” with the new config keys; link to the detailed docs.
