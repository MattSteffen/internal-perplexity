import asyncio
import json
import time
import uuid
import os
from typing import List, Optional, Dict, Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Configuration
# -----------------------------------------------------------------------------
# Use environment variables for better configuration management
PROXY_API_URL = os.getenv("PROXY_API_URL", "http://localhost:11434")
AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", "radchat,ollama,milvus_search").split(',')
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 300))
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3")

# Pydantic Models for API Schema
# -----------------------------------------------------------------------------
class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "function"
    function: Dict[str, Any]

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

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
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None

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

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Model Handling Logic
# -----------------------------------------------------------------------------
class Model:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def handle_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> ChatCompletionResponse | AsyncIterator[str]:
        if request.stream:
            return self._handle_streaming_request(request, headers)
        else:
            return await self._handle_non_streaming_request(request, headers)

    async def _handle_non_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> ChatCompletionResponse:
        # Default implementation for non-streaming: collect streaming response
        final_response = None
        async for chunk_str in self._handle_streaming_request(request, headers):
            if chunk_str.startswith("data: "):
                chunk_data = chunk_str[len("data: "):]
                if chunk_data.strip() == "[DONE]":
                    break
                chunk = json.loads(chunk_data)
                if chunk.get("object") == "chat.completion.final":
                    final_response = chunk
                    break
        
        if final_response:
            return ChatCompletionResponse(
                id=final_response['id'],
                created=final_response['created'],
                model=final_response['model'],
                choices=[Choice(**choice) for choice in final_response['choices']],
                usage=None # Usage info not available in this path
            )
        raise NotImplementedError("Non-streaming response is not properly handled.")


    def _handle_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
        raise NotImplementedError

    def _generate_id(self) -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

def _to_openai_format(ollama_response: dict, model_id: str, response_id: str) -> dict:
    message = ollama_response.get("message", {})
    finish_reason = "tool_calls" if message.get("tool_calls") else ("stop" if ollama_response.get("done") else None)
    
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # Ollama's tool_calls are already in the correct format, but we ensure they are ToolCall objects
        parsed_tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls]
    else:
        parsed_tool_calls = []

    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model_id,
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=message.get("role"),
                    content=message.get("content"),
                    tool_calls=parsed_tool_calls
                ),
                finish_reason=finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=ollama_response.get("prompt_eval_count", 0),
            completion_tokens=ollama_response.get("eval_count", 0),
            total_tokens=ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
        )
    ).model_dump()

class RadChatModel(Model):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from models import radchat
        self.radchat = radchat

    async def _handle_non_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> ChatCompletionResponse:
        response_id = self._generate_id()
        created = int(time.time())
        try:
            radchat_response = await self.radchat.pipe(request.model_dump())
            # Assuming radchat returns a single string content
            content = ""
            async for item in radchat_response:
                content += item
            
            return ChatCompletionResponse(
                id=response_id,
                created=created,
                model=self.model_id,
                choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing radchat model: {str(e)}")

    async def _handle_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
        response_id = self._generate_id()
        created = int(time.time())
        try:
            async for content_chunk in self.radchat.pipe(request.model_dump()):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [{"index": 0, "delta": {"content": content_chunk}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk with finish reason
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_chunk = {"error": {"message": f"Error processing radchat model: {str(e)}", "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

class OllamaModel(Model):
    async def _handle_non_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> ChatCompletionResponse:
        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = DEFAULT_OLLAMA_MODEL

        proxy_headers = {"Content-Type": "application/json", **{k: v for k, v in headers.items() if k.lower() not in ['host', 'content-length', 'connection']}}
        
        try:
            response = await self.client.post(f"{PROXY_API_URL}/api/chat", json=request_data, headers=proxy_headers)
            response.raise_for_status()
            ollama_response = response.json()
            
            openai_response = _to_openai_format(ollama_response, self.model_id, self._generate_id())
            return ChatCompletionResponse(**openai_response)

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request to Ollama failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Ollama proxy error: {e.response.text}")

    async def _handle_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = DEFAULT_OLLAMA_MODEL

        proxy_headers = {"Content-Type": "application/json", **{k: v for k, v in headers.items() if k.lower() not in ['host', 'content-length', 'connection']}}

        try:
            async with self.client.stream("POST", f"{PROXY_API_URL}/api/chat", json=request_data, headers=proxy_headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama proxy error: {error_text.decode()}")
                
                async for chunk in response.aiter_bytes():
                    if chunk:
                        line = chunk.decode('utf-8').strip()
                        if line:
                            ollama_chunk = json.loads(line)
                            openai_response = _to_openai_format(ollama_chunk, self.model_id, self._generate_id())
                            
                            # Since we are converting a full response to a chunk, we need to extract the delta
                            delta = openai_response['choices'][0]['message']
                            finish_reason = openai_response['choices'][0]['finish_reason']
                            
                            openai_chunk = {
                                "id": openai_response['id'],
                                "object": "chat.completion.chunk",
                                "created": openai_response['created'],
                                "model": self.model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except httpx.RequestError as e:
            error_chunk = {"error": {"message": f"Request to Ollama failed: {str(e)}", "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

class MilvusSearchModel(Model):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from models import milvus_search
        self.milvus_search = milvus_search

    async def _handle_streaming_request(self, request: ChatCompletionRequest, headers: Dict[str, str]) -> AsyncIterator[str]:
        async for chunk in self.milvus_search.pipe(request.model_dump()):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

# Router for Model Management
# -----------------------------------------------------------------------------
class Router:
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self._register_models()

    def _register_models(self):
        model_map = {
            "radchat": RadChatModel,
            "ollama": OllamaModel,
            "milvus_search": MilvusSearchModel
        }
        for model_id in AVAILABLE_MODELS:
            if model_id in model_map:
                self.models[model_id] = model_map[model_id](model_id)

    def get_model(self, model_id: str) -> Optional[Model]:
        return self.models.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(id=model_id, created=int(time.time()), owned_by="organization-owner") for model_id in self.models.keys()]

# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(title="OpenAI Compatible API", version="1.0.0")
router = Router()

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(data=router.list_models())

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    model = router.get_model(request.model)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found.")

    headers = dict(http_request.headers)
    response = await model.handle_request(request, headers)

    if isinstance(response, AsyncIterator):
        return StreamingResponse(response, media_type="text/event-stream")
    else:
        return JSONResponse(content=response.model_dump())

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run("openai:app", host="0.0.0.0", port=port, reload=reload, log_level="info")
