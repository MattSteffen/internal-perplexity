# TODO:

- [ ] openwebui groq manifold fix chat title generation

OpenWebUI

```bash
docker run --hostname=304fe453051c --user=0:0 --env=USE_OLLAMA_DOCKER=False --env=WEBUI_AUTH=False --env=PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=LANG=C.UTF-8 --env=GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D --env=PYTHON_VERSION=3.11.9 --env=PYTHON_PIP_VERSION=24.0 --env=PYTHON_SETUPTOOLS_VERSION=65.5.1 --env=PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/e03e1607ad60522cf34a92e834138eb89f57667c/public/get-pip.py --env=PYTHON_GET_PIP_SHA256=ee09098395e42eb1f82ef4acb231a767a6ae85504a9cf9983223df0a7cbd35d7 --env=ENV=prod --env=PORT=8080 --env=USE_CUDA_DOCKER=false --env=USE_CUDA_DOCKER_VER=cu121 --env=USE_EMBEDDING_MODEL_DOCKER=sentence-transformers/all-MiniLM-L6-v2 --env=USE_RERANKING_MODEL_DOCKER= --env=OLLAMA_BASE_URL=/ollama --env=OPENAI_API_BASE_URL= --env=OPENAI_API_KEY= --env=WEBUI_SECRET_KEY= --env=SCARF_NO_ANALYTICS=true --env=DO_NOT_TRACK=true --env=ANONYMIZED_TELEMETRY=false --env=WHISPER_MODEL=base --env=WHISPER_MODEL_DIR=/app/backend/data/cache/whisper/models --env=RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 --env=RAG_RERANKING_MODEL= --env=SENTENCE_TRANSFORMERS_HOME=/app/backend/data/cache/embedding/models --env=HF_HOME=/app/backend/data/cache/embedding/models --env=HOME=/root --env=WEBUI_BUILD_VERSION=eff736acd2e0bbbdd0eeca4cc209b216a1f23b6a --network=bridge --workdir=/app/backend -p 3030:8080 --restart=no --label='org.opencontainers.image.created=2024-07-10T21:47:28.630Z' --label='org.opencontainers.image.description=User-friendly WebUI for LLMs (Formerly Ollama WebUI)' --label='org.opencontainers.image.licenses=MIT' --label='org.opencontainers.image.revision=eff736acd2e0bbbdd0eeca4cc209b216a1f23b6a' --label='org.opencontainers.image.source=https://github.com/open-webui/open-webui' --label='org.opencontainers.image.title=open-webui' --label='org.opencontainers.image.url=https://github.com/open-webui/open-webui' --label='org.opencontainers.image.version=main' --runtime=runc -d ghcr.io/open-webui/open-webui:main
```

Pipelines

```bash
docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
```

Set api url to http://host.docker.internal:9099, apikey to 0p3n-w3bu!
