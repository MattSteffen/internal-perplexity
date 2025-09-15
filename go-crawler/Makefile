# Go Crawler Makefile

# Variables
BINARY_NAME=crawler
SERVER_BINARY=crawler-server
VERSION?=1.0.0
BUILD_TIME=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS=-ldflags "-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME) -X main.gitCommit=$(GIT_COMMIT)"

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Build directories
BUILD_DIR=bin
DIST_DIR=dist

.PHONY: all build clean test coverage run deps help

# Default target
all: clean deps test build

# Build the CLI tool
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/crawler

# Build the HTTP server
build-server:
	@echo "Building $(SERVER_BINARY)..."
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/$(SERVER_BINARY) ./cmd/server

# Build both binaries
build-all: build build-server

# Clean build artifacts
clean:
	@echo "Cleaning..."
	$(GOCLEAN)
	@rm -rf $(BUILD_DIR)
	@rm -rf $(DIST_DIR)
	@rm -rf coverage.out coverage.html

# Run tests
test:
	@echo "Running tests..."
	$(GOTEST) -v ./...

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	$(GOTEST) -v -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run tests for specific package
test-pkg:
	@echo "Running tests for package $(PKG)..."
	$(GOTEST) -v ./$(PKG)

# Download dependencies
deps:
	@echo "Downloading dependencies..."
	$(GOMOD) download
	$(GOMOD) tidy

# Update dependencies
deps-update:
	@echo "Updating dependencies..."
	$(GOMOD) tidy
	$(GOCMD) get -u ./...

# Run the CLI tool
run: build
	@echo "Running $(BINARY_NAME)..."
	./$(BUILD_DIR)/$(BINARY_NAME)

# Run the HTTP server
run-server: build-server
	@echo "Running $(SERVER_BINARY)..."
	./$(BUILD_DIR)/$(SERVER_BINARY)

# Format code
fmt:
	@echo "Formatting code..."
	$(GOCMD) fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
	fi

# Vet code
vet:
	@echo "Vetting code..."
	$(GOCMD) vet ./...

# Check code quality
check: fmt vet lint

# Install development tools
install-tools:
	@echo "Installing development tools..."
	$(GOCMD) install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	$(GOCMD) install github.com/cosmtrek/air@latest

# Generate mocks for testing
mocks:
	@echo "Generating mocks..."
	@if command -v mockery >/dev/null 2>&1; then \
		mockery --all --output ./mocks; \
	else \
		echo "mockery not installed. Install with: go install github.com/vektra/mockery/v2@latest"; \
	fi

# Create distribution packages
dist: build-all
	@echo "Creating distribution packages..."
	@mkdir -p $(DIST_DIR)
	@cp $(BUILD_DIR)/$(BINARY_NAME) $(DIST_DIR)/
	@cp $(BUILD_DIR)/$(SERVER_BINARY) $(DIST_DIR)/
	@cp README.md $(DIST_DIR)/
	@cp crawler-config.json $(DIST_DIR)/ 2>/dev/null || echo "No config file found"

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker build -t go-crawler:$(VERSION) .

# Docker run
docker-run:
	@echo "Running Docker container..."
	docker run -p 8080:8080 go-crawler:$(VERSION)

# Development with hot reload
dev:
	@if command -v air >/dev/null 2>&1; then \
		air; \
	else \
		echo "Air not installed. Install with: go install github.com/cosmtrek/air@latest"; \
	fi

# Install the CLI tool system-wide
install: build
	@echo "Installing $(BINARY_NAME) system-wide..."
	sudo cp $(BUILD_DIR)/$(BINARY_NAME) /usr/local/bin/

# Uninstall the CLI tool
uninstall:
	@echo "Uninstalling $(BINARY_NAME)..."
	sudo rm -f /usr/local/bin/$(BINARY_NAME)

# Generate documentation
docs:
	@echo "Generating documentation..."
	@if command -v godoc >/dev/null 2>&1; then \
		godoc -http=:6060; \
	else \
		echo "godoc not available"; \
	fi

# Show help
help:
	@echo "Go Crawler - Available targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  build       - Build the CLI tool"
	@echo "  build-server - Build the HTTP server"
	@echo "  build-all   - Build both binaries"
	@echo ""
	@echo "Development targets:"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run tests"
	@echo "  coverage    - Run tests with coverage"
	@echo "  fmt         - Format code"
	@echo "  lint        - Lint code"
	@echo "  vet         - Vet code"
	@echo "  check       - Run all code quality checks"
	@echo ""
	@echo "Dependency targets:"
	@echo "  deps        - Download dependencies"
	@echo "  deps-update - Update dependencies"
	@echo ""
	@echo "Run targets:"
	@echo "  run         - Run the CLI tool"
	@echo "  run-server  - Run the HTTP server"
	@echo "  dev         - Run with hot reload (requires air)"
	@echo ""
	@echo "Distribution targets:"
	@echo "  dist        - Create distribution packages"
	@echo "  install     - Install CLI tool system-wide"
	@echo "  uninstall   - Uninstall CLI tool"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "Utility targets:"
	@echo "  mocks       - Generate mocks for testing"
	@echo "  docs        - Generate documentation"
	@echo "  help        - Show this help message"

# Default target when no arguments are given
.DEFAULT_GOAL := help

