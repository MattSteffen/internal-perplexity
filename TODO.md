# TODO

- [ ] Load real data into milvus
  - [ ] Test tool calling or structured output for gpt-oss
  - [ ] Get real data and test queries via script
- [ ] OI
  - [x] Update
  - [x] Create script to update
  - [x] Load real radchat.py into oi
  - [ ] Load query tool into oi
- [ ] Agent-backend

  - [ ] Go types for openai requests/responses
  - [ ] Tool server implementation

- [ ] Docling
  - [ ] Create request with body that uses ollama and matches https://github.com/docling-project/docling-serve/blob/main/docs/usage.md
  - [ ] Decide between docling-serve and docling-serve-cpu
- [ ] Crawler server
  - [ ] Register collections
- [ ] Milvus
  - [ ] Collections descriptions should have the metadata schema, this way when describing to LLM, it'll know what to look for and how to filter.
- Agent backend
  - [ ] Implement OI's agent API types

Launch docling with:

```bash
docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 -e DOCLING_SERVE_ENABLE_REMOTE_SERVICES=1 quay.io/docling-project/docling-serve
```

```bash
curl -X POST 'http://localhost:5001/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "options": {
      "from_formats": ["pdf"],
      "to_formats": ["md"],
      "abort_on_error": true,
      "do_picture_description": true,
      "image_export_mode": "embedded",
      "include_images": true,
      "picture_description_api": {
        "url": "http://host.docker.internal:11434/v1/chat/completions",
        "params": {
          "model": "granite3.2-vision:latest"
        },
        "timeout": 600,
        "prompt": "Describe this image in detail for a technical document."
      }
    },
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }' > test.json
```
