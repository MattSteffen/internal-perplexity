import asyncio
import json
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models import radchat

# Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None


class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


# FastAPI app
app = FastAPI(title="OpenAI Compatible API", version="1.0.0")

# Configuration - you can modify these
PROXY_API_URL = "https://localhost:11434"  # Change this to your target API
AVAILABLE_MODELS = ["radchat"]


class ModelHandler:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def proxy_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
        """Proxy the request to another API with the same format"""
        try:
            # Prepare the request for the target API
            proxy_headers = {
                "Content-Type": "application/json",
                **headers  # Forward original headers
            }
            
            request_data = request.model_dump()
            async with self.client.stream(
                "POST",
                f"{PROXY_API_URL}/v1/chat/completions",
                json=request_data,
                headers=proxy_headers
            ) as response:
                print(response)
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Proxy API error: {error_text.decode()}"
                    )
                
                if request.stream:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk.decode()
                else:
                    content = await response.aread()
                    yield content.decode()
                    
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
        print("Proxy request completed")
    
    
    async def generate_mock_response(self, request: ChatCompletionRequest) -> str:
        """Generate a mock response for testing when proxy is not available"""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())
        
        # Simple mock response
        mock_content = f"This is a mock response from {request.model}. Your message was: {request.messages[-1].content if request.messages else 'No message'}"
        
        response = ChatCompletionResponse(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=mock_content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(request.messages) * 10,  # Rough estimate
                completion_tokens=len(mock_content.split()),
                total_tokens=len(request.messages) * 10 + len(mock_content.split())
            )
        )
        
        return response.model_dump_json()


model_handler = ModelHandler()


@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = [
        Model(
            id=model_id,
            created=int(time.time()),
            owned_by="organization-owner"
        )
        for model_id in AVAILABLE_MODELS
    ]
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """Handle chat completions with optional streaming"""
    
    # Validate model
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not found. Available models: {AVAILABLE_MODELS}"
        )
    
    # Extract headers (excluding some internal ones)
    headers = {
        k: v for k, v in http_request.headers.items()
        if k.lower() not in ['host', 'content-length', 'connection']
    }
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        return await non_stream_chat_completion(request, headers)


async def stream_chat_completion(request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
    """Handle streaming chat completion"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    try:
        if request.model == "radchat":
            radchat_response = radchat.pipe(request.model_dump())

            chunk_data = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "test content"},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(radchat_response)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # Mock streaming for other models
            mock_content = f"This is a streaming response from {request.model}."
            words = mock_content.split()
            
            for i, word in enumerate(words):
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": word + " " if i < len(words) - 1 else word},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.1)  # Simulate streaming delay
            
            # Send final chunk
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def non_stream_chat_completion(request: ChatCompletionRequest, headers: Dict[str, str]):
    """Handle non-streaming chat completion"""
    try:
        if request.model == "model-a":
            # Proxy to another API
            response_text = ""
            request.model = "qwen3:1.7b"
            async for chunk in model_handler.proxy_request(request, headers):
                response_text += chunk
            
            # Parse and return the response
            try:
                response_data = json.loads(response_text)
                return response_data
            except json.JSONDecodeError:
                # If parsing fails, create a wrapper response
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }]
                }
        else:
            # Mock response for other models
            mock_response = await model_handler.generate_mock_response(request)
            return json.loads(mock_response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "openai:app",  # Change "main" to your filename if different
        host="localhost",
        port=8001,
        reload=True,
        log_level="info"
    )