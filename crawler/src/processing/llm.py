import ollama
import json
import threading
import time
from typing import List, Dict, Union, Optional, Any


class LLM:
    """
    A simplified client for interacting with Ollama using the official Python client library.
    Supports text prompts, message history, and structured JSON output with schema validation.
    """
    
    def __init__(self,
                 model_name: str,
                 base_url: str = "http://localhost:11434",
                 system_prompt: Optional[str] = None,
                 ctx_length: int = 32000,
                 default_timeout: float = 300.0):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the Ollama model to use
            host: Ollama server URL
            system_prompt: Optional system prompt to prepend to conversations
            ctx_length: Context length for the model
            default_timeout: Default request timeout in seconds
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.ctx_length = ctx_length
        self.default_timeout = default_timeout
        
        # Initialize Ollama client
        self.client = ollama.Client(host=base_url)
        
    #     # Verify model availability
    #     self._check_model_availability()
    
    # def _check_model_availability(self):
    #     """Check if the specified model is available locally."""
    #     try:
    #         models = self.client.list()
    #         available_models = [model['name'] for model in models['models']]
            
    #         if not any(self.model_name == m or self.model_name == m.split(':')[0] for m in available_models):
    #             print(f"Warning: Model '{self.model_name}' not found locally. Available models: {available_models}")
    #             print("Ollama will attempt to pull the model automatically.")
    #     except Exception as e:
    #         print(f"Warning: Could not verify model availability: {e}")
    
    def invoke(self,
               prompt_or_messages: Union[str, List[Dict[str, Any]]],
               response_format: Optional[Dict[str, Any]] = None,
               timeout: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt or message history to the model and get a response.
        
        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output
            timeout: Request timeout in seconds (uses default if None)
            
        Returns:
            String response or parsed JSON dictionary if response_format is provided
            
        Raises:
            TimeoutError: If the request exceeds the timeout duration
            RuntimeError: If there's an error with the API call
        """
        if timeout is None:
            timeout = self.default_timeout
        
        # Build messages list
        messages = self._build_messages(prompt_or_messages)
        
        # Set up options
        options = {
            'num_ctx': self.ctx_length
        }
        
        # Use threading to enforce timeout
        result = {'response': None, 'error': None}
        
        def api_call():
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    format=response_format,
                    options=options,
                    stream=False
                )
                result['response'] = response
            except Exception as e:
                result['error'] = e
        
        # Start the API call in a separate thread
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        
        # Wait for completion with timeout
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            # Thread is still running, meaning we hit the timeout
            raise TimeoutError(f"Request to Ollama model '{self.model_name}' timed out after {timeout} seconds")
        
        # Check if there was an error in the API call
        if result['error']:
            raise RuntimeError(f"Error calling Ollama model '{self.model_name}': {result['error']}")
        
        if result['response'] is None:
            raise RuntimeError(f"No response received from Ollama model '{self.model_name}'")
        
        # Extract content from response
        content = result['response']['message']['content']
        
        # If we requested JSON format, try to parse it
        if response_format is not None:
            return self._parse_json_response(content)
        
        return content
    
    def _build_messages(self, prompt_or_messages: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Build the messages list for the API call."""
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt
            })
        
        # Add user messages
        if isinstance(prompt_or_messages, str):
            messages.append({
                'role': 'user',
                'content': prompt_or_messages
            })
        elif isinstance(prompt_or_messages, list):
            messages.extend(prompt_or_messages)
        else:
            raise ValueError("prompt_or_messages must be a string or list of message dictionaries")
        
        return messages
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response, handling common formatting issues."""
        try:
            # Clean up common JSON formatting issues
            cleaned_content = content.strip()
            
            # Remove markdown code blocks if present
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:].strip()
            elif cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:].strip()
            
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3].strip()
            
            return json.loads(cleaned_content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Raw content: {content}")


# Example usage
def test():
    # Initialize the LLM client
    llm = LLM(
        model_name="qwen3:latest",  # or whatever model you have
        system_prompt="You are a helpful assistant.",
        ctx_length=16000,
        default_timeout=120.0
    )
    
    # Simple text generation
    response = llm.invoke("What is the capital of France?")
    print("Text response:", response)
    
    # Test with custom timeout
    try:
        response = llm.invoke("Write a very long essay about AI", timeout=5.0)
        print("Quick response:", response)
    except TimeoutError as e:
        print("Timeout caught:", e)
    
    # JSON structured output
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "number"}
        },
        "required": ["capital", "country"]
    }
    
    json_response = llm.invoke(
        "What is the capital of France? Include population if you know it.",
        response_format=schema
    )
    print("JSON response:", json_response)
    
    # Message history
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"}
    ]
    
    history_response = llm.invoke(messages)
    print("History response:", history_response)

if __name__ == "__main__":
    test()