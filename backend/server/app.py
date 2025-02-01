# app.py
from flask import Flask
from core.router import api_blueprint
from utils.error_handling import register_error_handlers

app = Flask(__name__)

# Register the API blueprint
app.register_blueprint(api_blueprint)

# Register custom error handlers
register_error_handlers(app)

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)

