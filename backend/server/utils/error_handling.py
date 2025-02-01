# utils/error_handling.py
from flask import jsonify

def handle_bad_request(error):
    response = jsonify({
        "detail": "Invalid request",
        "error": str(error)
    })
    response.status_code = 400
    return response

def register_error_handlers(app):
    # Register a handler for 400 Bad Request errors
    app.register_error_handler(400, handle_bad_request)
