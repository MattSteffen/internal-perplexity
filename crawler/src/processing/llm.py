# Supposed to use rest requests with response formats to ollama
import requests 

# TODO: make sure context length is large enough
import requests
import json
import warnings
import base64
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Generator, Tuple

class LLM:
    """
    A client for interacting with an Ollama API using the /api/chat endpoint,
    supporting text, images, structured JSON output, tools, and timeouts.
    """
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_REQUEST_TIMEOUT = 60  # seconds for the HTTP request
    DEFAULT_KEEP_ALIVE = "5m"     # Default keep_alive for the model

    def __init__(self,
                 model_name: str,
                 ollama_base_url: str = DEFAULT_OLLAMA_URL,
                 system_prompt: Optional[str] = None,
                 default_options: Optional[Dict[str, Any]] = {"num_ctx": 32000},
                 default_request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
                 default_keep_alive: Union[str, int] = DEFAULT_KEEP_ALIVE):
        """
        Initializes the LLM client for the /api/chat endpoint.

        Args:
            model_name (str): The name of the Ollama model to use (e.g., "qwen3:1.7b", "granite3.2-vision:latest").
            ollama_base_url (str, optional): The base URL for the Ollama API.
            system_prompt (str, optional): A default system-level prompt. This will be
                                           added as the first message in the chat list.
            default_options (dict, optional): Default model parameters (e.g., temperature).
            default_request_timeout (int, optional): Default timeout in seconds for HTTP requests.
            default_keep_alive (Union[str, int], optional): Default keep_alive duration.
        """
        if not model_name:
            raise ValueError("model_name must be specified.")

        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.api_url = f"{self.ollama_base_url}/api/chat"
        self.system_prompt = system_prompt
        self.default_options = default_options if default_options is not None else {}
        self.default_request_timeout = default_request_timeout
        self.default_keep_alive = default_keep_alive

        self._check_ollama_availability()
        self._check_model_availability()

    def _image_to_base64(self, image_path: Union[str, Path]) -> str:
        """Converts an image file to a base64 encoded string."""
        try:
            path = Path(image_path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Image file not found or is not a file: {image_path}")
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path} to base64: {e}") from e

    def _check_ollama_availability(self):
        """Checks if the Ollama API is reachable."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5) # /api/tags is a good health check
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                f"Ensure Ollama is running. Error: {e}"
            ) from e

    def _check_model_availability(self):
        """Checks if the specified model appears in Ollama's local model list."""
        try:
            list_api_url = f"{self.ollama_base_url}/api/tags"
            response = requests.get(list_api_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            available_models = [model_info['name'] for model_info in models_data.get('models', [])]
            
            if not any(self.model_name == m or self.model_name == m.split(':')[0] for m in available_models):
                warnings.warn(
                    f"Model '{self.model_name}' not found in Ollama's local list: {available_models}. "
                    "Ollama might attempt to pull it, or the API call could fail if not pulled.",
                    UserWarning
                )
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Could not verify model availability via /api/tags. Error: {e}", UserWarning)
        except json.JSONDecodeError:
            warnings.warn("Could not parse model list from Ollama /api/tags.", UserWarning)

    def invoke(self,
               prompt_or_messages: Union[str, List[Dict[str, Any]]],
               response_format: Optional[Union[str, Dict[str, Any]]] = None,
               images: Optional[List[Union[str, Path]]] = None,
               tools: Optional[List[Dict[str, Any]]] = None,
               options: Optional[Dict[str, Any]] = None,
               request_timeout: Optional[int] = None,
               keep_alive: Optional[Union[str, int]] = None,
               stream: bool = False
               ) -> Union[Any, Generator[Dict[str, Any], None, None]]:
        """
        Sends messages to the Ollama model via /api/chat and gets a response.

        Args:
            prompt_or_messages: If a string, it's a single user prompt. If a list of dicts,
                                it's the message history (each dict with "role", "content",
                                and optional "images" (list of base64 strings) or "tool_calls").
            response_format: "json" for JSON output, or a dict representing a JSON schema.
            images: List of image file paths or base64 strings. Only used if
                    'prompt_or_messages' is a string.
            tools: List of tools the model can use.
            options: Model parameters (e.g., temperature) overriding defaults.
            request_timeout: HTTP request timeout in seconds for this call.
            keep_alive: keep_alive duration for this call.
            stream: If True, returns a generator for streaming responses.
                    If False (default), returns a single aggregated response.

        Returns:
            If stream=False: The processed response (parsed JSON, text, or dict with tool_calls).
            If stream=True: A generator yielding JSON response chunks from Ollama.

        Raises:
            ConnectionError, requests.exceptions.HTTPError, TimeoutError, ValueError.
        """
        if not prompt_or_messages and not isinstance(prompt_or_messages, list):
            raise ValueError("prompt_or_messages cannot be empty if it's a string.")

        # --- 1. Construct Messages ---
        final_messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            final_messages.append({"role": "system", "content": self.system_prompt})

        if isinstance(prompt_or_messages, str):
            user_message: Dict[str, Any] = {"role": "user", "content": prompt_or_messages}
            if images:
                base64_images_list = []
                for img_ref in images:
                    if isinstance(img_ref, Path) or (isinstance(img_ref, str) and (Path(img_ref).exists() and Path(img_ref).is_file())):
                        base64_images_list.append(self._image_to_base64(img_ref))
                    elif isinstance(img_ref, str): # Assume it's already a base64 string
                         # Basic check for base64: length, no path chars, and padding are weak indicators.
                         # A more robust check would involve trying to decode a small part.
                        if len(img_ref) > 100 and not ('/' in img_ref[:10] or '\\' in img_ref[:10]):
                            base64_images_list.append(img_ref)
                        else:
                            warnings.warn(f"Image string '{img_ref[:30]}...' doesn't look like a path or a long base64 string. Adding as is.", UserWarning)
                            base64_images_list.append(img_ref) # Add as is, let Ollama decide
                    else:
                        warnings.warn(f"Invalid image reference type: {type(img_ref)}. Skipping.", UserWarning)

                if base64_images_list:
                    user_message["images"] = base64_images_list
            final_messages.append(user_message)
        elif isinstance(prompt_or_messages, list):
            if images:
                warnings.warn(
                    "'images' argument is ignored when 'prompt_or_messages' is a list. "
                    "Embed base64 images directly within the message objects under the 'images' key.", UserWarning
                )
            final_messages.extend(prompt_or_messages)
        else:
            raise ValueError("'prompt_or_messages' must be a string or a list of message dictionaries.")

        # --- 2. Construct Payload ---
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": final_messages,
            "stream": stream,
        }

        if response_format is not None:
            payload["format"] = response_format
        if tools is not None:
            payload["tools"] = tools

        current_options = self.default_options.copy()
        if options:
            current_options.update(options)
        if current_options:
            payload["options"] = current_options

        current_keep_alive = keep_alive if keep_alive is not None else self.default_keep_alive
        if current_keep_alive is not None:
            payload["keep_alive"] = current_keep_alive
        
        # --- 3. Make API Call ---
        timeout_val = request_timeout if request_timeout is not None else self.default_request_timeout

        try:
            api_response = requests.post(self.api_url, json=payload, timeout=timeout_val, stream=stream)
            api_response.raise_for_status() 

            # --- 4. Handle Response ---
            if stream:
                return self._stream_response_generator(api_response)
            else:
                response_data = api_response.json()
                return self._handle_non_streamed_response(response_data, response_format)

        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama API timed out after {timeout_val}s for model {self.model_name}."
            raise TimeoutError(error_msg) from None
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}. Error: {e}") from e
        except requests.exceptions.RequestException as e: # Includes HTTPError
            error_message = f"Ollama API request failed for model {self.model_name}: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f"\nOllama API Response: {error_details}"
                    if "error" in error_details: # Ollama specific error format
                        raise ValueError(f"Ollama API error: {error_details['error']}") from e
                except json.JSONDecodeError:
                    error_message += f"\nOllama API Response (raw text): {e.response.text}"
            raise type(e)(error_message) from e # Re-raise with more info

    def _stream_response_generator(self, api_response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """Generator for streaming responses."""
        try:
            for line in api_response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    yield chunk
        except requests.exceptions.ChunkedEncodingError as e:
            warnings.warn(f"Stream interrupted: {e}", UserWarning)
        except json.JSONDecodeError as e:
            warnings.warn(f"Error decoding JSON line from stream: {e}", UserWarning)
        finally:
            api_response.close()

    def _handle_non_streamed_response(self,
                                     response_data: Dict[str, Any],
                                     response_format_requested: Optional[Union[str, Dict[str, Any]]]):
        """Processes the non-streamed JSON response from Ollama."""

        if response_data.get("done_reason") in ["load", "unload"]:
            return {"status": response_data.get("done_reason"), "details": response_data}

        message_obj = response_data.get("message")
        if not message_obj: # Should not happen with done=true if not load/unload
            warnings.warn(f"Ollama response missing 'message' object. Full response: {response_data}", UserWarning)
            return response_data # Return full data for debugging

        generated_content_str = message_obj.get("content", "").strip()
        tool_calls = message_obj.get("tool_calls") # List of tool calls or None

        parsed_json_content = None
        is_json_format_requested = (isinstance(response_format_requested, str) and response_format_requested.lower() == "json") or \
                                   isinstance(response_format_requested, dict)

        if is_json_format_requested and generated_content_str:
            try:
                temp_content = generated_content_str
                if temp_content.startswith("```json"): temp_content = temp_content[len("```json"):].strip()
                if temp_content.startswith("```"): temp_content = temp_content[len("```"):].strip()
                if temp_content.endswith("```"): temp_content = temp_content[:-len("```")].strip()
                parsed_json_content = json.loads(temp_content)
            except json.JSONDecodeError as e:
                warnings.warn(
                    f"LLM was asked for JSON (format: {response_format_requested}) but returned "
                    f"non-JSON or malformed JSON. Error: {e}. Raw content: '{generated_content_str}'",
                    UserWarning
                )
        elif is_json_format_requested and not generated_content_str and not tool_calls:
             warnings.warn(
                f"LLM was asked for JSON (format: {response_format_requested}) but returned empty content and no tool_calls.",
                UserWarning
            )

        # --- Determine primary return ---
        # If content was parsed as JSON, it's likely the main desired output.
        if parsed_json_content is not None:
            if tool_calls: # If we got parsed JSON AND tool calls, return both in a structured way
                return {"content": parsed_json_content, "tool_calls": tool_calls, "_full_ollama_response": response_data}
            return parsed_json_content # Just the parsed JSON
        
        # If no parsed JSON, but there are tool calls (especially if content is empty)
        if tool_calls:
            # If content is also present, it might be an explanation for the tool call
            return {"content": generated_content_str, "tool_calls": tool_calls, "_full_ollama_response": response_data}

        # Default: return the raw string content
        return generated_content_str

# --- Example Usage ---
def test():
    # Create a dummy image file for testing if Pillow is installed
    dummy_image_path = Path("dummy_llm_image.png")
    if not dummy_image_path.exists():
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 50), color = 'blue')
            d = ImageDraw.Draw(img)
            d.text((10,10), "Ollama!", fill=(255,255,0))
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow not installed, skipping dummy image creation. Image tests will require a real image or manual base64 string.")
            dummy_image_path = None
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            dummy_image_path = None
    
    # --- LLM Client Instances ---
    # Ensure Ollama is running (e.g., `ollama serve`)
    # Ensure models are pulled (e.g., `ollama pull qwen3:1.7b`, `ollama pull granite3.2-vision:latest`)
    try:
        llm_text = LLM(model_name="qwen3:1.7b") # For text tasks
        # For vision, ensure granite3.2-vision:latest (or other vision model) is pulled and running
        # llm_vision = LLM(model_name="granite3.2-vision:latest") 
    except (ConnectionError, ValueError) as e:
        print(f"Failed to initialize LLM clients: {e}")
        print("Ensure Ollama is running and models are pulled. Exiting example.")
        if dummy_image_path and dummy_image_path.exists(): dummy_image_path.unlink()
        exit()

    try:
        print("\n--- 1. Simple Text Generation (qwen3:1.7b) ---")
        response1 = llm_text.invoke("Why is the sky blue? Be very brief.")
        print(f"Response: {response1}\n")

        print("\n--- 2. Chat History (qwen3:1.7b) ---")
        history = [
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
            {"role": "user", "content": "What is a famous food there?"}
        ]
        response2 = llm_text.invoke(history)
        print(f"Response to history: {response2}\n")

        print("\n--- 3. Structured JSON Output (format='json', qwen3:1.7b) ---")
        json_prompt = "Extract: My friend Alice is 28 years old. Output a JSON with 'name' and 'age'."
        response3 = llm_text.invoke(json_prompt, response_format="json")
        print(f"Raw JSON response type: {type(response3)}")
        if isinstance(response3, (dict, list)):
            print(f"Parsed JSON: {json.dumps(response3, indent=2)}\n")
        else:
            print(f"Non-JSON response: {response3}\n")

        print("\n--- 4. Structured JSON Output (JSON Schema, qwen3:1.7b) ---")
        schema = {
            "type": "object",
            "properties": {
                "person_name": {"type": "string"},
                "person_age": {"type": "integer"},
                "city_of_residence": {"type": "string"}
            },
            "required": ["person_name", "person_age"]
        }
        schema_prompt = "User's data: John Doe, 42, lives in New York."
        response4 = llm_text.invoke(schema_prompt, response_format=schema, options={"temperature": 0})
        print(f"Raw Schema response type: {type(response4)}")
        if isinstance(response4, (dict, list)):
            print(f"Parsed Schema JSON: {json.dumps(response4, indent=2)}\n")
        else:
            print(f"Non-JSON Schema response: {response4}\n")

        if dummy_image_path and dummy_image_path.exists():
            try:
                llm_vision = LLM(model_name="granite3.2-vision:latest") # Initialize vision model here if not done globally
                print("\n--- 5. Image Input (granite3.2-vision:latest) ---")
                img_prompt = "What text do you see in this image?"
                response5 = llm_vision.invoke(img_prompt, images=[dummy_image_path])
                print(f"Vision Response: {response5}\n")
            except (ConnectionError, ValueError) as e: # Catch errors specific to llm_vision init or invoke
                 print(f"Skipping granite3.2-vision:latest test or granite3.2-vision:latest test failed: {e}\n")
        else:
            print("\n--- 5. Image Input (granite3.2-vision:latest) ---")
            print("Skipping image test as dummy image is not available.\n")
        
        print("\n--- 6. Timeout Test (qwen3:1.7b) ---")
        try:
            # This prompt is designed to take some time
            timeout_prompt = "Write a detailed 500-word essay on the future of AI."
            print(f"Attempting call with short timeout (1s) for prompt: '{timeout_prompt[:30]}...'")
            llm_text.invoke(timeout_prompt, request_timeout=1) # 1 second timeout
        except TimeoutError as e:
            print(f"Successfully caught TimeoutError: {e}\n")
        except Exception as e: # Catch other errors if timeout isn't hit
            print(f"Timeout test resulted in other error: {e}\n")

        print("\n--- 7. Tool Use Example (qwen3:1.7b) ---")
        # This is conceptual; model must support tool use well. Llama 3 is getting better.
        weather_tool = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetches the current weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "The city name."}},
                    "required": ["city"]
                }
            }
        }]
        tool_prompt = "What's the weather in London?"
        response7 = llm_text.invoke(tool_prompt, tools=weather_tool, options={"temperature":0.1})
        print(f"Tool Response type: {type(response7)}")
        if isinstance(response7, dict) and response7.get("tool_calls"):
            print(f"Tool calls: {json.dumps(response7['tool_calls'], indent=2)}")
            if response7.get("content"): print(f"Accompanying text: {response7['content']}")
        else:
            print(f"Response (no direct tool call or text): {response7}")
        print("")

        print("\n--- 8. Streaming Test (qwen3:1.7b) ---")
        stream_prompt = "Tell me a very short story about a robot."
        print(f"Streaming prompt: '{stream_prompt}'")
        full_streamed_text = ""
        try:
            for chunk in llm_text.invoke(stream_prompt, stream=True):
                # print(json.dumps(chunk)) # To inspect full chunk structure
                if chunk.get("message") and chunk["message"].get("content"):
                    content_part = chunk["message"]["content"]
                    print(content_part, end="", flush=True)
                    full_streamed_text += content_part
                if chunk.get("done"):
                    print("\n--- Stream finished ---")
                    # print(f"Final chunk details: {json.dumps(chunk, indent=2)}")
                    break # Exit loop once done
            print(f"\nAggregated streamed text: {full_streamed_text}\n")
        except Exception as e:
            print(f"Streaming test error: {e}\n")
            
        print("\n--- 9. Model Loading/Unloading (Conceptual) ---")
        # Load model (or ensure it's loaded), keep alive for 10s after this call
        print("Sending empty messages to 'load' model (keep_alive='10s')...")
        load_resp = llm_text.invoke([], keep_alive="10s") # messages=[] is valid for this
        print(f"Load response: {load_resp}")

        # Request to unload model (keep_alive=0)
        # print("\nSending empty messages to 'unload' model (keep_alive=0)...")
        # unload_resp = llm_text.invoke([], keep_alive=0)
        # print(f"Unload response: {unload_resp}")
        # Note: Actual unloading depends on Ollama's internal management.

    except (ConnectionError, ValueError, TimeoutError, requests.exceptions.RequestException) as e:
        print(f"\nAN EXAMPLE FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED IN EXAMPLES: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dummy_image_path and dummy_image_path.exists():
            dummy_image_path.unlink()
            print(f"\nCleaned up dummy image: {dummy_image_path}")