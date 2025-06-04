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
    # Ollama server configuration
    OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama default
    
    # Server configuration
    HOST = "localhost"
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

def convert_openai_to_ollama(openai_request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI format to Ollama format"""
    ollama_request = {
        "model": openai_request.get("model"),
        "messages": openai_request.get("messages", []),
        "stream": openai_request.get("stream", False)
    }
    
    # Add optional parameters
    if "temperature" in openai_request:
        ollama_request["options"] = ollama_request.get("options", {})
        ollama_request["options"]["temperature"] = openai_request["temperature"]
    
    if "max_tokens" in openai_request and openai_request["max_tokens"]:
        ollama_request["options"] = ollama_request.get("options", {})
        ollama_request["options"]["num_predict"] = openai_request["max_tokens"]
    
    return ollama_request

def convert_ollama_to_openai(ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert Ollama response to OpenAI format"""
    if "message" in ollama_response:
        # Non-streaming response
        return {
            "id": f"chatcmpl-{''.join(['0']*29)}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": ollama_response["message"],
                "finish_reason": "stop" if ollama_response.get("done", False) else None
            }],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
            }
        }
    return ollama_response

async def forward_request_stream(
    request_data: Dict[str, Any], 
    headers: Dict[str, str]
) -> AsyncGenerator[bytes, None]:
    """Forward streaming request to Ollama"""
    
    destination_url = f"{config.OLLAMA_BASE_URL}/api/chat"
    ollama_request = convert_openai_to_ollama(request_data)
    
    # Ollama doesn't need auth headers
    destination_headers = {
        "Content-Type": "application/json",
    }
    
    logger.info(f"Forwarding streaming request to {destination_url}")
    logger.debug(f"Ollama request data: {json.dumps(ollama_request, indent=2)}")
    
    async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
        try:
            async with client.stream(
                "POST",
                destination_url,
                json=ollama_request,
                headers=destination_headers
            ) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(f"Ollama server error: {response.status_code} - {error_content}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama server error: {error_content.decode()}"
                    )
                
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            # Parse Ollama response
                            ollama_data = json.loads(chunk)
                            
                            # Convert to OpenAI streaming format
                            if "message" in ollama_data and "content" in ollama_data["message"]:
                                openai_chunk = {
                                    "id": f"chatcmpl-{''.join(['0']*29)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": request_data.get("model"),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": ollama_data["message"]["content"]
                                        },
                                        "finish_reason": "stop" if ollama_data.get("done", False) else None
                                    }]
                                }
                                
                                formatted_chunk = f"data: {json.dumps(openai_chunk)}\n\n"
                                yield formatted_chunk.encode()
                            
                            if ollama_data.get("done", False):
                                yield b"data: [DONE]\n\n"
                                break
                                
                        except json.JSONDecodeError:
                            continue
                    
        except httpx.TimeoutException:
            logger.error("Request to Ollama server timed out")
            raise HTTPException(status_code=504, detail="Request timeout")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=502, detail="Bad Gateway")

async def forward_request(
    request_data: Dict[str, Any], 
    headers: Dict[str, str]
) -> Dict[str, Any]:
    """Forward non-streaming request to Ollama"""
    
    destination_url = f"{config.OLLAMA_BASE_URL}/api/chat"
    ollama_request = convert_openai_to_ollama(request_data)
    
    # Ollama doesn't need auth headers
    destination_headers = {
        "Content-Type": "application/json",
    }
    
    logger.info(f"Forwarding request to {destination_url}")
    logger.debug(f"Ollama request data: {json.dumps(ollama_request, indent=2)}")
    
    async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
        try:
            response = await client.post(
                destination_url,
                json=ollama_request,
                headers=destination_headers
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama server error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama server error: {response.text}"
                )
            
            ollama_response = response.json()
            openai_response = convert_ollama_to_openai(ollama_response, request_data.get("model"))
            return openai_response
                
        except httpx.TimeoutException:
            logger.error("Request to Ollama server timed out")
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
                forward_request_stream(request_data, headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # Handle non-streaming response
            response_data = await forward_request(request_data, headers)
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
        # Get models from Ollama
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_models = response.json()
                
                # Convert to OpenAI format
                openai_models = {
                    "object": "list",
                    "data": []
                }
                
                for model in ollama_models.get("models", []):
                    openai_models["data"].append({
                        "id": model["name"],
                        "object": "model",
                        "created": 0,
                        "owned_by": "ollama"
                    })
                
                return JSONResponse(content=openai_models)
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch models from Ollama")
                
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is available
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        ollama_status = "unhealthy"
    
    return {
        "status": "healthy", 
        "service": "openai-proxy",
        "ollama_status": ollama_status
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "OpenAI Compatible Proxy Server for Ollama",
        "version": "1.0.0",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/models",
            "/health"
        ],
        "ollama_url": config.OLLAMA_BASE_URL
    }

if __name__ == "__main__":
    print(f"Starting OpenAI Compatible Proxy Server on {config.HOST}:{config.PORT}")
    print(f"Forwarding requests to Ollama at: {config.OLLAMA_BASE_URL}")
    print("Make sure Ollama is running: ollama serve")
    
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    )