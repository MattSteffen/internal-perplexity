# endpoints/completions.py
from flask import Blueprint, request, jsonify
from models import get_model_handler

completions_bp = Blueprint('completions', __name__)

@completions_bp.route('/v1/completions', methods=['POST'])
def create_completion():
    """
    Example curl command to test this endpoint:
    
    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "model-a",
            "prompt": "Hello world"
        }'
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    model = data.get("model")
    prompt = data.get("prompt")
    if not model or not prompt:
        return jsonify({"error": "Missing 'model' or 'prompt' field"}), 400

    handler = get_model_handler(model)
    if not handler:
        return jsonify({"error": "Unsupported model"}), 400

    # Call the model-specific function to get the response
    result = handler(prompt)
    return jsonify({"response": result})
