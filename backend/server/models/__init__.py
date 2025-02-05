# models/__init__.py
from models.model_a import model_a_handler, ollama_handler
from models.model_b import model_b_handler

# Mapping of model names to their handler functions
model_mapping = {
    "model-a": model_a_handler,
    "model-b": model_b_handler,
    "ollamas": ollama_handler
}

def get_model_handler(model_name: str):
    return model_mapping.get(model_name)
