# endpoints/completions.py
from flask import Blueprint, jsonify

models_bp = Blueprint('models', __name__)

@models_bp.route('/v1/models', methods=['GET'])
def get_models():
    """
    Example curl command to test this endpoint:
    
    curl http://localhost:8000/v1/models
    """
    
    models_list = [
        {
            "id": "model-a",
            "object": "model",
            "created": 1686935002,  # Example timestamp
            "owned_by": "your-organization"
        },
        {
            "id": "model-b",
            "object": "model",
            "created": 1686935003,
            "owned_by": "your-organization"
        },
        {
            "id": "ollamas",
            "object": "model",
            "created": 1686935004,
            "owned_by": "ollamas"
        }
    ]
    
    return jsonify({
        "object": "list",
        "data": models_list
    })