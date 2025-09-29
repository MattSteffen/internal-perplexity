# Generic AI Stack Helm Chart

This Helm chart deploys a complete AI stack including:

- Milvus (vector database)
- Open WebUI (frontend interface)
- Ollama (model serving)

## Prerequisites

- Kubernetes cluster
- Helm v3+
- Storage class "microk8s-hostpath" or modify the values.yaml to use your preferred storage class
- NGINX Ingress Controller

## Installation

```bash
# Add required repositories
helm repo add milvus https://milvus-io.github.io/milvus-helm
helm repo add open-webui https://open-webui.github.io/helm-charts
helm repo add ollama https://otwld.github.io/ollama-helm/
helm repo update

# Update dependencies
helm dependency update ./generic-ai-stack

# Install the chart
helm install ai-stack ./generic-ai-stack --set global.domain=YOUR_IP_ADDRESS
```

## Configuration

See values.yaml for configuration options.

### Important Parameters

- `global.domain`: IP address or domain for your services
- `global.storageClass`: Storage class to use for persistence

## Accessing Services

After deployment, you can access the services at:

- Open WebUI: http://webui.YOUR_IP_ADDRESS.nip.io
- Ollama API: http://ollama.YOUR_IP_ADDRESS.nip.io
- Milvus: http://milvus.YOUR_IP_ADDRESS.nip.io

## Uninstallation

```bash
helm uninstall ai-stack
```
