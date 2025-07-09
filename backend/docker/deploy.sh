#!/bin/bash

# Docker Compose deployment script
# Usage: ./deploy.sh [service_name]

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

COMPOSE_FILE="docker-compose.yml"

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Compose file not found: $COMPOSE_FILE"
    exit 1
fi

SERVICE=$1

if [ -z "$SERVICE" ]; then
    print_info "No service specified, deploying all services."
    SERVICES=$(docker-compose -f "$COMPOSE_FILE" config --services)
else
    # Validate service name
    if ! docker-compose -f "$COMPOSE_FILE" config --services | grep -q "^$SERVICE$"; then
        print_error "Unknown service: $SERVICE"
        echo "Available services:"
        docker-compose -f "$COMPOSE_FILE" config --services
        exit 1
    fi
    print_info "Deploying service: $SERVICE"
    SERVICES=$SERVICE
fi

# Stop any existing containers for the specified services
print_info "Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down $SERVICES 2>/dev/null || true

# Pull latest images
print_info "Pulling latest images for $SERVICES..."
docker-compose -f "$COMPOSE_FILE" pull $SERVICES

# Start the services
print_info "Starting services: $SERVICES..."
docker-compose -f "$COMPOSE_FILE" up -d $SERVICES

# Check if containers are running
if docker-compose -f "$COMPOSE_FILE" ps $SERVICES | grep -q "Up"; then
    print_info "Deployment of $SERVICES completed successfully!"
    echo ""
    print_info "Container status:"
    docker-compose -f "$COMPOSE_FILE" ps $SERVICES
else
    print_error "Failed to deploy $SERVICES"
    print_info "Checking logs..."
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 $SERVICES
    exit 1
fi

print_info "Deployment complete!"
