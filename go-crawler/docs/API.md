# Go Crawler REST API Documentation

## Overview

The Go Crawler provides a REST API for managing document processing operations. The API follows RESTful conventions and returns JSON responses.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

Currently, no authentication is required. In production deployments, consider adding API key authentication.

## Response Format

All API responses follow this format:

```json
{
  "success": true|false,
  "data": {...},
  "error": "error message"
}
```

## Endpoints

### Health Check

Check if the service is running and healthy.

**GET** `/health`

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0"
  }
}
```

### Start Document Crawl

Start crawling and processing documents from a specified path.

**POST** `/crawl`

**Request Body:**

```json
{
  "path": "/path/to/documents"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "message": "Crawl started successfully",
    "path": "/path/to/documents",
    "status": "running"
  }
}
```

### Run Benchmark

Execute performance benchmarks on the processed documents.

**POST** `/benchmark`

**Response:**

```json
{
  "success": true,
  "data": {
    "results_by_doc": {...},
    "placement_distribution": {...},
    "distance_distribution": [...],
    "percent_in_top_k": {...},
    "search_time_distribution": [...]
  }
}
```

### Get Statistics

Retrieve current crawler statistics.

**GET** `/stats`

**Response:**

```json
{
  "success": true,
  "data": {
    "total_documents": 100,
    "processed_documents": 95,
    "failed_documents": 5,
    "total_chunks": 1500,
    "processing_time": "2m30s",
    "average_chunk_time": "100ms"
  }
}
```

### Get Configuration

Retrieve current crawler configuration (sensitive data masked).

**GET** `/config`

**Response:**

```json
{
  "success": true,
  "data": {
    "embeddings": {...},
    "llm": {...},
    "database": {...},
    "chunk_size": 10000,
    ...
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error

Error responses include an error message:

```json
{
  "success": false,
  "error": "Detailed error message"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider adding rate limiting for production deployments.

## CORS

The API includes CORS headers to allow cross-origin requests from web applications.

## Examples

### Using curl

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Start crawl
curl -X POST http://localhost:8080/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documents"}'

# Get statistics
curl http://localhost:8080/api/v1/stats

# Run benchmark
curl -X POST http://localhost:8080/api/v1/benchmark
```

### Using JavaScript/fetch

```javascript
// Health check
fetch("http://localhost:8080/api/v1/health")
  .then((response) => response.json())
  .then((data) => console.log(data));

// Start crawl
fetch("http://localhost:8080/api/v1/crawl", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    path: "/path/to/documents",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

## Monitoring

The API server logs all requests and responses. Monitor the logs for:

- Request duration
- Error rates
- Crawler performance metrics

## Production Considerations

For production deployment:

1. Add authentication and authorization
2. Implement rate limiting
3. Add request/response compression
4. Configure proper logging and monitoring
5. Set up health checks and alerts
6. Use HTTPS with proper certificates
