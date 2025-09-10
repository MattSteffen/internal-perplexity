# Multi-stage build for Go Crawler

# Build stage
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Install git and ca-certificates (needed for HTTPS requests)
RUN apk add --no-cache git ca-certificates

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the CLI tool
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o crawler ./cmd/crawler

# Build the HTTP server
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o crawler-server ./cmd/server

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN adduser -D -s /bin/sh crawler

# Set working directory
WORKDIR /app

# Copy binaries from builder stage
COPY --from=builder /app/crawler .
COPY --from=builder /app/crawler-server .
COPY --from=builder /app/crawler-config.json .

# Create temp directory
RUN mkdir -p tmp && chown crawler:crawler tmp

# Switch to non-root user
USER crawler

# Expose port for HTTP server
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/api/v1/health || exit 1

# Default command runs the HTTP server
CMD ["./crawler-server"]
