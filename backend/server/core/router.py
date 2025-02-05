# core/router.py
from flask import Blueprint
from endpoints.completions import completions_bp
from endpoints.models import models_bp

api_blueprint = Blueprint('api', __name__)

# Register endpoints with the API blueprint
api_blueprint.register_blueprint(completions_bp)
api_blueprint.register_blueprint(models_bp)
