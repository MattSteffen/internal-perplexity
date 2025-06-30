import asyncio
import httpx
import json

# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:8001/v1"

# Payloads for Test Cases
# -----------------------------------------------------------------------------
# 1. Test case for Ollama model with streaming
ollama_streaming_payload = {
    "model": "ollama",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "stream": True
}

# 2. Test case for Ollama model without streaming
ollama_non_streaming_payload = {
    "model": "ollama",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": False
}

# 3. Test case for RadChat model with a tool call
search_tool = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search for academic papers based on a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "The author to search for."},
                "topic": {"type": "string", "description": "The topic to search for."}
            },
            "required": [],
        },
    },
}
radchat_tool_call_payload = {
    "model": "radchat",
    "messages": [{"role": "user", "content": "Find papers by author 'joshua fixelle'"}],
    "tools": [search_tool],
    "stream": False
}

# 4. Test case for structured JSON output (non-streaming)
ollama_json_non_streaming_payload = {
    "model": "ollama",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides structured JSON responses."
        },
        {
            "role": "user",
            "content": "Provide the steps to solve the equation '2x + 5 = 11' in a JSON format with a \"steps\" key."
        }
    ],
    "response_format": {"type": "json_object"},
    "stream": False
}

# 5. Test case for structured JSON output (streaming)
ollama_json_streaming_payload = {
    "model": "ollama",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides structured JSON responses."
        },
        {
            "role": "user",
            "content": "Provide the steps to solve the equation '3x - 4 = 11' in a JSON format with a \"steps\" key."
        }
    ],
    "response_format": {"type": "json_object"},
    "stream": True
}

# 6. Test case for tool calling (non-streaming)
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

ollama_tool_call_non_streaming_payload = {
    "model": "ollama",
    "messages": [{"role": "user", "content": "What's the weather like in Boston?"}],
    "tools": [weather_tool],
    "stream": False
}

# 7. Test case for tool calling (streaming)
ollama_tool_call_streaming_payload = {
    "model": "ollama",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [weather_tool],
    "stream": True
}

# 8. Test case for Milvus Search model
milvus_search_payload = {
    "model": "milvus_search",
    "messages": [{"role": "user", "content": "Find papers by author 'joshua fixelle'"}],
    "stream": True
}


# Helper Functions
# -----------------------------------------------------------------------------
async def send_request(client: httpx.AsyncClient, payload: dict):
    """
    Sends a request to the chat completions endpoint and prints the response.
    Handles both streaming and non-streaming responses.
    """
    is_streaming = payload.get("stream", False)
    
    async with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload, timeout=30) as response:
        print(f"\n--- Testing model: {payload.get('model')}, Stream: {is_streaming}, Tools: {bool(payload.get('tools'))}, JSON: {bool(payload.get('response_format'))} ---")
        if response.status_code == 200:
            if is_streaming:
                async for chunk in response.aiter_text():
                    print(chunk, end="")
            else:
                response_data = await response.aread()
                try:
                    print(json.dumps(json.loads(response_data), indent=2))
                except json.JSONDecodeError:
                    print(response_data.decode())
        else:
            print(f"Error: {response.status_code}")
            print(await response.aread())
    print("\n-----------------------------------------------------")

# Main Test Execution
# -----------------------------------------------------------------------------
async def main():
    """
    Runs all test cases against the OpenAI-compatible server.
    """
    async with httpx.AsyncClient() as client:
        # Test Ollama with streaming
        # await send_request(client, ollama_streaming_payload)
        
        # Test Ollama without streaming
        # await send_request(client, ollama_non_streaming_payload)
        
        # Test RadChat with a potential tool call
        # await send_request(client, radchat_tool_call_payload)

        # Test structured JSON output
        # await send_request(client, ollama_json_non_streaming_payload)
        # await send_request(client, ollama_json_streaming_payload)

        # Test tool calling
        # await send_request(client, ollama_tool_call_non_streaming_payload)
        # await send_request(client, ollama_tool_call_streaming_payload)

        # Test Milvus Search
        await send_request(client, milvus_search_payload)

if __name__ == "__main__":
    asyncio.run(main())