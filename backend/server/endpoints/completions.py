# endpoints/completions.py
import json
import time
from uuid import uuid4
from flask import Blueprint, request, jsonify, Response, stream_with_context
from models import get_model_handler

completions_bp = Blueprint('completions', __name__)

@completions_bp.route('/v1/chat/completions', methods=['POST'])
def create_completion():
    """
    Example curl command for non-streaming:
    
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "model-a",
              "messages": [{"role": "user", "content": "Hello world"}]
            }'
    
    Example curl command for streaming:
    
    curl -N -X POST http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
              "model": "model-a",
              "messages": [{"role": "user", "content": "Hello world"}],
              "stream": true
         }'
    """
    print("Received request to /v1/chat/completions")
    data = request.get_json()
    print(f"Request data: {data}")

    if not data:
        return jsonify({
            "error": {
                "message": "Invalid JSON body",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400

    model = data.get("model")
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    if not model or not messages:
        return jsonify({
            "error": {
                "message": "Missing required fields: 'model' or 'messages'",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400

    # Extract last user message as the prompt
    prompt = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")

    handler = get_model_handler(model)
    if not handler:
        return jsonify({
            "error": {
                "message": f"Model '{model}' not found",
                "type": "invalid_request_error",
                "code": 404
            }
        }), 404

    # Call the model handler; the handler returns a string result.
    result: str = handler(prompt + "\nAnswer in a markdown table format.")

    # Unique ID and timestamp for the response
    chat_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())
    system_fingerprint = "fp_ollama"

    if stream:
        # Streaming response: yield each chunk as an OpenAI-compatible "data:" event.
        def generate():
            chunk_size = 5  # Adjust chunk size as needed.
            for i in range(0, len(result), chunk_size):
                chunk = result[i:i+chunk_size]
                chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": system_fingerprint,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            # For the first chunk, include the role if desired.
                            "role": "assistant",
                            "content": chunk
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.1)
            # Final chunk: include the citations in the delta.
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": system_fingerprint,
                "choices": [{
                    "index": 0,
                    "delta": {
                        # You may choose to include an empty content or additional information
                        "content": "",
                    },
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        # Non-streaming response: return the full result in one JSON response.
        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result.split()),
            "total_tokens": len(prompt.split()) + len(result.split())
        }
        response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result,
                },
                "finish_reason": "stop"
            }],
            "usage": usage
        }
        return jsonify(response)