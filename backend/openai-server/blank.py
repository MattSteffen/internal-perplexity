"""
A simple and efficient asynchronous proxy server built with FastAPI and HTTPX.

This server is designed to perfectly forward all incoming HTTP requests to a specified
target host. It handles all HTTP methods, paths, headers, query parameters, and
request bodies, making it a versatile tool for redirecting traffic.

The server uses environment variables for configuration:
- TARGET_HOST: The destination URL for proxied requests (default: http://localhost:11434).
- PORT: The port on which this proxy server will run (default: 8002).
"""

import os
import uvicorn
import httpx
import logging
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response

# --- Configuration ---
TARGET_HOST = os.getenv("TARGET_HOST", "http://localhost:11434")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Logging Setup ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- FastAPI Application ---
app = FastAPI(title="Perfect Proxy Server")
client = httpx.AsyncClient(base_url=TARGET_HOST, timeout=300.0)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request):
    """
    Catches all incoming requests and forwards them to the target host.

    This function extracts all relevant information from the incoming request,
    including the method, path, headers, query parameters, and body. It then
    constructs a new request and sends it to the target host using HTTPX.

    The response from the target host, including the status code, headers, and
    body, is streamed back to the original client.
    """
    path = request.url.path
    query_params = request.query_params
    
    logger.info(f"Incoming request: {request.method} {path} from {request.client.host}")
    logger.debug(f"Request headers: {dict(request.headers)}")

    # Create the full URL for the target service
    url = httpx.URL(path=path, query=str(query_params).encode("utf-8"))

    # Copy headers, but update the host to match the target
    headers = dict(request.headers)
    headers["host"] = client.base_url.host

    # Read and log the request body
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            log_data = json.loads(body)
            
            if "messages" in log_data and isinstance(log_data.get("messages"), list):
                for message in log_data["messages"]:
                    if isinstance(message.get("content"), list):
                        # Filter out image URLs, keep text blocks
                        message["content"] = [
                            item if item.get("type") == "text" else f"type: {item.get('type')}"
                            for item in message["content"]
                        ]
            
            # Omit large fields that are likely images
            for key, value in log_data.items():
                if isinstance(value, str) and len(value) > 1000:
                     log_data[key] = f"<omitted field '{key}' of length {len(value)}>"

            logger.info(f"Request body (JSON): {json.dumps(log_data)}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.info(f"Request body is {len(body)} bytes (non-JSON or binary).")
    elif body:
        logger.info(f"Request body is {len(body)} bytes (Content-Type: {content_type}).")


    # Build the proxied request
    proxied_request = client.build_request(
        method=request.method,
        url=url,
        content=body,
        headers=headers,
    )
    
    logger.info(f"Proxying request to: {proxied_request.method} {proxied_request.url}")
    logger.debug(f"Proxy request headers: {dict(proxied_request.headers)}")

    try:
        # Send the request and stream the response
        response = await client.send(proxied_request, stream=True)
        
        logger.info(f"Received response with status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")

        # Stream the response back to the client
        return StreamingResponse(
            content=response.aiter_raw(),
            status_code=response.status_code,
            headers=response.headers,
        )
    except httpx.RequestError as e:
        # Handle cases where the proxy target is unavailable
        error_message = f"An error occurred while proxying the request to {TARGET_HOST}: {e}"
        logger.error(error_message)
        return Response(content=error_message, status_code=502) # 502 Bad Gateway

# --- Main Entry Point ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"Starting perfect proxy on http://0.0.0.0:{port}")
    print(f"Proxying all requests to: {TARGET_HOST}")
    
    uvicorn.run("blank:app", host="0.0.0.0", port=port, reload=reload, log_level=LOG_LEVEL.lower())
