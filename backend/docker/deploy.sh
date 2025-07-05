#!/bin/bash

# Docker Compose deployment script
# Usage: ./deploy.sh <service_name>

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    print_error "No service specified!"
    echo "Usage: $0 <service_name>"
    echo "Available services: ollama, service2, service3"
    echo "Example: $0 ollama"
    exit 1
fi

SERVICE=$1
COMPOSE_FILE=""

# Map service names to compose files
case $SERVICE in
    "ollama")
        COMPOSE_FILE="docker-compose-ollama.yml"
        ;;
    "service2")
        COMPOSE_FILE="docker-compose-service2.yml"
        ;;
    "service3")
        COMPOSE_FILE="docker-compose-service3.yml"
        ;;
    *)
        print_error "Unknown service: $SERVICE"
        echo "Available services: ollama, service2, service3"
        exit 1
        ;;
esac

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi

print_info "Deploying service: $SERVICE"
print_info "Using compose file: $COMPOSE_FILE"

# Stop any existing containers for this service
print_info "Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# Pull latest images
print_info "Pulling latest images..."
docker-compose -f "$COMPOSE_FILE" pull

# Start the service
print_info "Starting service..."
docker-compose -f "$COMPOSE_FILE" up -d

# Check if containers are running
if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    print_info "Service $SERVICE deployed successfully!"
    echo ""
    print_info "Container status:"
    docker-compose -f "$COMPOSE_FILE" ps
else
    print_error "Failed to deploy service $SERVICE"
    print_info "Checking logs..."
    docker-compose -f "$COMPOSE_FILE" logs --tail=20
    exit 1
fi

print_info "Deployment complete!"