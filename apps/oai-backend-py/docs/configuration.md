# oai-backend-py Configuration

This document lists all supported configuration settings and where they are used. All env-backed settings are centralized in `src/config.py`.

## Environment Variables

| Env Var | Type | Default | Used In |
| --- | --- | --- | --- |
| `OAI_OLLAMA_BASE_URL` | string | `http://localhost:11434/v1` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/config.py` (settings) |
| `OAI_API_KEY` | string | `ollama` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/config.py` (settings) |
| `OAI_HOST` | string | `0.0.0.0` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_PORT` | int | `8000` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_KEYCLOAK_URL` | string | `""` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/auth.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/endpoints/auth.py` |
| `OAI_CLIENT_ID` | string | `""` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/auth.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/endpoints/auth.py` |
| `OAI_CLIENT_SECRET` | string | `""` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/auth.py` |
| `OAI_REDIRECT_URI` | string | `http://localhost:3000/api/auth/callback` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/endpoints/auth.py` |
| `OAI_FRONTEND_REDIRECT_URL` | string | `http://localhost:3000/dashboard` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/endpoints/auth.py` |
| `OLLAMA_BASE_URL` | string | `http://localhost:11434` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/milvuschat.py` |
| `OLLAMA_EMBEDDING_MODEL` | string | `all-minilm:v2` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py` |
| `OLLAMA_LLM_MODEL` | string | `gpt-oss:20b` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/milvuschat.py` |
| `OLLAMA_REQUEST_TIMEOUT` | int | `300` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/milvuschat.py` |
| `OLLAMA_CONTEXT_LENGTH` | int | `32000` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/milvuschat.py` |
| `MILVUS_URI` | string | unset | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/milvus_client.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_HOST` | string | `localhost` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_PORT` | string | `19530` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_USERNAME` | string | `matt` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py` |
| `MILVUS_PASSWORD` | string | `steffen` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py` |
| `IRAD_COLLECTION_NAME` | string | `arxiv3` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_NPROBE` | int | `10` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_SEARCH_LIMIT` | int | `5` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_HYBRID_SEARCH_LIMIT` | int | `10` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_RRF_K` | int | `100` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `MILVUS_DROP_RATIO` | float | `0.2` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |
| `AGENT_MAX_TOOL_CALLS` | int | `5` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py`, `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/milvuschat.py` |
| `AGENT_DEFAULT_ROLE` | string | `system` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py` |
| `AGENT_LOGGING_LEVEL` | string | `INFO` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/clients/radchat.py` |
| `MILVUS_POOL_TTL_SECONDS` | int | `900` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `MILVUS_POOL_MAX_SIZE` | int | `250` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_CORS_ALLOW_ORIGINS` | list[string] | `["*"]` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_CORS_ALLOW_METHODS` | list[string] | `["*"]` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_CORS_ALLOW_HEADERS` | list[string] | `["*"]` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `OAI_CORS_ALLOW_CREDENTIALS` | bool | `true` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/main.py` |
| `MILVUS_SEARCH_MAX_WORKERS` | int | `4` | `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/tools/milvus_search.py` |

## Precedence and Derived Values

- `MILVUS_URI` takes precedence over `MILVUS_HOST` and `MILVUS_PORT`. If `MILVUS_URI` is unset, the application derives `http://{MILVUS_HOST}:{MILVUS_PORT}`.
- CORS list settings accept comma-separated values via env vars (for example `OAI_CORS_ALLOW_ORIGINS=example.com,localhost`).

## Server Tunables

- Milvus client pool size and TTL: `MILVUS_POOL_MAX_SIZE`, `MILVUS_POOL_TTL_SECONDS`.
- CORS allow list settings: `OAI_CORS_ALLOW_ORIGINS`, `OAI_CORS_ALLOW_METHODS`, `OAI_CORS_ALLOW_HEADERS`, `OAI_CORS_ALLOW_CREDENTIALS`.
- Search executor workers: `MILVUS_SEARCH_MAX_WORKERS`.

## Internal Defaults That Remain Hard-Coded

- Pipeline template defaults live in `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/endpoints/pipeline_registry.py`.
- Milvus token parsing fallback (`root:Milvus`) lives in `/Users/mattsteffen/projects/llm/internal-perplexity/apps/oai-backend-py/src/milvus_client.py`.
