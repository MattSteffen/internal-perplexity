# models/model_a.py
def model_a_handler(prompt: str) -> str:
    # Implement your logic for model-a here.
    return f"Model A response to: {prompt}"


import requests

def ollama_handler(prompt: str, model: str = "llama3.2:1b") -> str:
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # We handle streaming separately
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()
    
    if "choices" in response_json and len(response_json["choices"]) > 0:
        return response_json["choices"][0]["message"]["content"]
    
    return "Error: No response from model."
