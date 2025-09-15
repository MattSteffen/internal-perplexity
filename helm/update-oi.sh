#!/bin/bash
set -e

# Configurable values
CONTAINER_NAME="open-webui"
IMAGE_NAME="ghcr.io/open-webui/open-webui:latest"

echo "==> Stopping and removing existing container ($CONTAINER_NAME)..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

echo "==> Pulling latest image ($IMAGE_NAME)..."
docker pull $IMAGE_NAME

echo "==> Redeploying container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 3000:8080 \
  $IMAGE_NAME

echo "==> Done! Open WebUI should now be running on http://localhost:3000"
