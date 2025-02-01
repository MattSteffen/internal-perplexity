Provides an overview and instructions for setting up and running your server.

# OpenAI-Compatible API Server

This is a minimal viable product (MVP) for an API server that mimics the OpenAI API.
It accepts requests at `/v1/completions`, reads the `model` field, and routes the request
to a model-specific function.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   Run the server:
   python app.py
   Endpoints
   ```

POST /v1/completions: Accepts a JSON body with fields model and prompt, and returns a response.

---

### How It Works

1. **Incoming Request:**  
   A client sends a POST request to `/v1/completions` with a JSON payload containing the `"model"` and `"prompt"` fields.

2. **Request Handling:**  
   The endpoint in `endpoints/completions.py` validates the request and extracts the `model` and `prompt` fields. It then uses `models.get_model_handler()` to select the appropriate function for the given model.

3. **Model-Specific Processing:**  
   The selected model handler (e.g., `model_a_handler` or `model_b_handler`) processes the prompt and returns a response.

4. **Response:**  
   The response is sent back to the client in JSON format.

---

This Flask-based implementation should help you get started with an OpenAI-compatible API server that can redirect requests to different model handlers. Enjoy building and extending your API!

Below is one way to structure a minimal viable product (MVP) that “mimics” the OpenAI API endpoint behavior. In this example, we’ll use Python with [FastAPI](https://fastapi.tiangolo.com/) (though you could use Flask or another framework if you prefer). The idea is that our server accepts requests similar to OpenAI’s API, looks at the `"model"` field in the JSON body, and then calls a function specific to that model.

Below is an outline of the file structure along with a brief description of what each file does. You can adjust names and structure as needed.

---

### File Structure

```
openai-compatible-server/
├── app.py
├── endpoints/
│   └── completions.py
├── models/
│   ├── __init__.py
│   ├── model_a.py
│   └── model_b.py
├── core/
│   ├── __init__.py
│   └── router.py
├── utils/
│   ├── __init__.py
│   └── error_handling.py
├── requirements.txt
└── README.md
```

---

### File Descriptions

1. **app.py**  
   This is the entry point of your application. It creates the FastAPI app instance and includes the router from the core module.

   ```python
   # app.py
   from fastapi import FastAPI
   from core.router import api_router

   app = FastAPI(title="OpenAI Compatible API")

   app.include_router(api_router)

   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

2. **endpoints/completions.py**  
   This file defines the endpoint that clients will call (e.g., `/v1/completions`). It reads the request, extracts the `"model"` field, and routes the request to the correct model function.

   ```python
   # endpoints/completions.py
   from fastapi import APIRouter, HTTPException
   from pydantic import BaseModel
   from models import get_model_handler

   router = APIRouter()

   # Define a request model for validation (you can extend this as needed)
   class CompletionRequest(BaseModel):
       model: str
       prompt: str  # include additional fields as needed

   # Define a basic response model
   class CompletionResponse(BaseModel):
       response: str

   @router.post("/v1/completions", response_model=CompletionResponse)
   async def create_completion(request: CompletionRequest):
       handler = get_model_handler(request.model)
       if not handler:
           raise HTTPException(status_code=400, detail="Unsupported model")
       # Call the model-specific function to get the response
       result = handler(request.prompt)
       return CompletionResponse(response=result)
   ```

3. **models/\_\_init\_\_.py**  
   This file is used to expose the `get_model_handler` function which maps model names to their corresponding implementations.

   ```python
   # models/__init__.py
   from models.model_a import model_a_handler
   from models.model_b import model_b_handler

   # Mapping of model names to their handler functions
   model_mapping = {
       "model-a": model_a_handler,
       "model-b": model_b_handler,
   }

   def get_model_handler(model_name: str):
       return model_mapping.get(model_name)
   ```

4. **models/model_a.py**  
   Contains the implementation for handling requests for `"model-a"`. You can define the actual logic as required.

   ```python
   # models/model_a.py
   def model_a_handler(prompt: str) -> str:
       # Implement your logic for model-a here.
       return f"Model A response to: {prompt}"
   ```

5. **models/model_b.py**  
   Similar to `model_a.py`, this file contains the logic for `"model-b"`.

   ```python
   # models/model_b.py
   def model_b_handler(prompt: str) -> str:
       # Implement your logic for model-b here.
       return f"Model B response to: {prompt}"
   ```

6. **core/router.py**  
   This file sets up the API router. If you have multiple endpoints (e.g., completions, edits, etc.), you can import them here and include them in one unified router.

   ```python
   # core/router.py
   from fastapi import APIRouter
   from endpoints import completions

   api_router = APIRouter()
   api_router.include_router(completions.router)
   ```

7. **utils/error_handling.py**  
   (Optional but useful) A module for custom error handling or utility functions that you might want to share between endpoints.

   ```python
   # utils/error_handling.py
   from fastapi import Request, HTTPException
   from fastapi.responses import JSONResponse

   async def validation_exception_handler(request: Request, exc):
       return JSONResponse(
           status_code=400,
           content={"detail": "Invalid request", "error": str(exc)},
       )
   ```

   You can then register this error handler in `app.py` if needed.

8. **requirements.txt**  
   Lists the Python dependencies.

   ```
   fastapi==0.95.0
   uvicorn==0.22.0
   pydantic==1.10.2
   ```

9. **README.md**  
   Provides documentation on how to install dependencies, run the server, and any other notes about the project.

   ````markdown
   # OpenAI-Compatible API Server

   This is a minimal viable product (MVP) for an API server that mimics the OpenAI API.
   It accepts requests at `/v1/completions`, reads the `model` field, and routes the request
   to a model-specific function.

   ## Setup

   1. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
   ````

   2. Run the server:
      ```bash
      python app.py
      ```

   ## Endpoints

   - **POST /v1/completions**: Accepts a JSON body with fields `model` and `prompt`, and returns a response.

   ```

   ```

---

### How It Works

1. **Incoming Request:**  
   A client sends a POST request to `/v1/completions` with a JSON payload containing the `"model"` and `"prompt"` fields.

2. **Request Handling:**  
   The `endpoints/completions.py` file validates the request using Pydantic models. It then extracts the model name and uses `models.get_model_handler()` to find the appropriate function.

3. **Model-Specific Processing:**  
   The handler function (e.g., `model_a_handler` or `model_b_handler`) processes the prompt and returns a response.

4. **Response:**  
   The response is sent back to the client in the expected format.

---

This structure provides a clean separation between the routing logic, model-specific logic, and any utility functions you might need. As your project grows, you can expand on this structure by adding more endpoints, additional error handling, configuration management, and tests.
