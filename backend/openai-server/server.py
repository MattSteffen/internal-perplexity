import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenAI Compatible Proxy Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    # Destination server configuration
    DESTINATION_BASE_URL = "http://localhost:11434"  # Change this to your target server
    DESTINATION_API_KEY = "your-destination-api-key"  # Set your destination API key
    
    # Server configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Timeout settings
    TIMEOUT = 300  # 5 minutes

config = Config()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Add other OpenAI parameters as needed

async def forward_request_stream(
    endpoint: str, 
    request_data: Dict[str, Any], 
    headers: Dict[str, str]
) -> AsyncGenerator[bytes, None]:
    """Forward streaming request to destination server"""
    
    destination_url = f"{config.DESTINATION_BASE_URL}{endpoint}"
    
    # Prepare headers for destination
    destination_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DESTINATION_API_KEY}",
        "User-Agent": headers.get("User-Agent", "OpenAI-Proxy/1.0")
    }
    
    # Add any custom headers you want to forward
    for key, value in headers.items():
        if key.lower().startswith("x-") or key.lower() in ["user-agent"]:
            destination_headers[key] = value
    
    logger.info(f"Forwarding streaming request to {destination_url}")
    logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
    
    async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
        try:
            async with client.stream(
                "POST",
                destination_url,
                json=request_data,
                headers=destination_headers
            ) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(f"Destination server error: {response.status_code} - {error_content}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Destination server error: {error_content.decode()}"
                    )
                
                async for chunk in response.aiter_bytes():
                    yield chunk
                    
        except httpx.TimeoutException:
            logger.error("Request to destination server timed out")
            raise HTTPException(status_code=504, detail="Request timeout")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=502, detail="Bad Gateway")

async def forward_request(
    endpoint: str, 
    request_data: Dict[str, Any], 
    headers: Dict[str, str]
) -> Dict[str, Any]:
    """Forward non-streaming request to destination server"""
    
    destination_url = f"{config.DESTINATION_BASE_URL}{endpoint}"
    
    # Prepare headers for destination
    destination_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DESTINATION_API_KEY}",
        "User-Agent": headers.get("User-Agent", "OpenAI-Proxy/1.0")
    }
    
    # Add any custom headers you want to forward
    for key, value in headers.items():
        if key.lower().startswith("x-") or key.lower() in ["user-agent"]:
            destination_headers[key] = value
    
    logger.info(f"Forwarding request to {destination_url}")
    logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
    
    async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
        try:
            response = await client.post(
                destination_url,
                json=request_data,
                headers=destination_headers
            )
            
            if response.status_code != 200:
                logger.error(f"Destination server error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Destination server error: {response.text}"
                )
            
            return response.json()
                
        except httpx.TimeoutException:
            logger.error("Request to destination server timed out")
            raise HTTPException(status_code=504, detail="Request timeout")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=502, detail="Bad Gateway")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions requests"""
    try:
        # Parse request body
        request_data = await request.json()
        headers = dict(request.headers)
        
        # Extract stream parameter
        stream = request_data.get("stream", False)
        
        logger.info(f"Received chat completion request - Model: {request_data.get('model')}, Stream: {stream}")
        
        if stream:
            # Handle streaming response
            return StreamingResponse(
                forward_request_stream("/v1/chat/completions", request_data, headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # Handle non-streaming response
            response_data = await forward_request("/v1/chat/completions", request_data, headers)
            return JSONResponse(content=response_data)
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/models")
async def list_models(request: Request):
    """Handle models list requests"""
    try:
        headers = dict(request.headers)
        response_data = await forward_request("/v1/models", {}, headers)
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completions requests (legacy endpoint)"""
    try:
        request_data = await request.json()
        headers = dict(request.headers)
        stream = request_data.get("stream", False)
        
        logger.info(f"Received completion request - Model: {request_data.get('model')}, Stream: {stream}")
        
        if stream:
            return StreamingResponse(
                forward_request_stream("/v1/completions", request_data, headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            response_data = await forward_request("/v1/completions", request_data, headers)
            return JSONResponse(content=response_data)
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "openai-proxy"}

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "OpenAI Compatible Proxy Server",
        "version": "1.0.0",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions", 
            "/v1/models",
            "/health"
        ]
    }

if __name__ == "__main__":
    # Update configuration before starting
    print(f"Starting OpenAI Compatible Proxy Server on {config.HOST}:{config.PORT}")
    print(f"Forwarding requests to: {config.DESTINATION_BASE_URL}")
    print("Remember to update DESTINATION_BASE_URL and DESTINATION_API_KEY in the Config class!")
    
    uvicorn.run(
        "server:app",  # Changed from main:app to server:app
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    )