import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import ollama
import json
import threading
import time
import httpx

# TODO: Make this an interface and an ollama implementation


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    model_name: str
    base_url: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    ctx_length: int = 32000
    default_timeout: float = 300.0
    provider: str = "ollama"
    api_key: str = ""  # For providers that require API keys
    structured_output: str = "response_format"  # 'response_format' or 'tools'

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.model_name:
            raise ValueError("LLM model_name cannot be empty")
        if not self.base_url:
            raise ValueError("LLM base_url cannot be empty")
        if self.ctx_length <= 0:
            raise ValueError("Context length must be positive")
        if self.default_timeout <= 0:
            raise ValueError("Default timeout must be positive")
        if self.structured_output not in ["response_format", "tools"]:
            raise ValueError(
                "structured_output must be either 'response_format' or 'tools'"
            )

    @classmethod
    def ollama(
        cls,
        model_name: str,
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> "LLMConfig":
        """Create Ollama LLM configuration."""
        return cls(
            model_name=model_name,
            base_url=base_url,
            system_prompt=system_prompt,
            ctx_length=ctx_length,
            default_timeout=default_timeout,
            provider="ollama",
            structured_output=structured_output,
        )

    @classmethod
    def openai(
        cls,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: Optional[str] = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> "LLMConfig":
        """Create OpenAI LLM configuration."""
        return cls(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            ctx_length=ctx_length,
            default_timeout=default_timeout,
            provider="openai",
            structured_output=structured_output,
        )

    @classmethod
    def vllm(
        cls,
        model_name: str,
        base_url: str,
        api_key: str = "",
        system_prompt: Optional[str] = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> "LLMConfig":
        """Create vLLM LLM configuration."""
        return cls(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            ctx_length=ctx_length,
            default_timeout=default_timeout,
            provider="vllm",
            structured_output=structured_output,
        )


# TODO: Implement for vllm


def schema_to_openai_tools(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a JSON schema to OpenAI-compatible tools format.

    Args:
        schema: JSON schema dictionary

    Returns:
        List of OpenAI tool definitions
    """
    # Extract description from schema or use default
    description = schema.get(
        "description",
        "Extract metadata from the document corresponding to the parameters here",
    )

    # Create the tool definition
    tool = {
        "type": "function",
        "function": {
            "name": "extract_metadata",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        },
    }

    return [tool]


class LLM(ABC):
    """Abstract interface for Language Model implementations."""

    @abstractmethod
    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM with configuration.

        Args:
            config: Configuration object for the LLM
        """
        pass

    @abstractmethod
    def invoke(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Send a prompt or message history to the model and get a response.

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output (when structured_output='response_format')
            tools: Optional OpenAI-compatible tools for structured output (when structured_output='tools')

        Returns:
            String response or parsed JSON dictionary if structured output is requested
        """
        pass


def get_llm(config: LLMConfig) -> LLM:
    if config.provider == "ollama":
        return OllamaLLM(config)
    elif config.provider == "vllm":
        return VllmLLM(config)
    raise ValueError(f"unsupported model provider: {config.provider}")


class OllamaLLM(LLM):
    """
    A simplified client for interacting with Ollama using the official Python client library.
    Supports text prompts, message history, and structured JSON output with schema validation.
    Includes comprehensive logging and performance monitoring.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config: Configuration object for the LLM
        """
        self.config = config
        self.model_name = config.model_name
        self.system_prompt = config.system_prompt
        self.ctx_length = config.ctx_length

        # Get logger (already configured by main crawler)
        self.logger = logging.getLogger("OllamaLLM")
        self.logger.info(f"Initializing OllamaLLM with model: {config.model_name}")
        self.logger.debug(f"Base URL: {config.base_url}")
        self.logger.debug(f"Context length: {config.ctx_length}")
        self.logger.debug(f"Default timeout: {config.default_timeout}s")

        # Initialize Ollama client with timeout from config
        try:
            self.client = ollama.Client(
                host=config.base_url, timeout=config.default_timeout
            )
            self.logger.info("✅ OllamaLLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OllamaLLM client: {e}")
            raise

    def invoke(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt or message history to the model and get a response with comprehensive logging.

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output (when structured_output='response_format')
            tools: Optional OpenAI-compatible tools for structured output (when structured_output='tools')

        Returns:
            String response or parsed JSON dictionary if structured output is requested

        Raises:
            TimeoutError: If the request exceeds the timeout duration
            RuntimeError: If there's an error with the API call
        """
        invoke_start_time = time.time()

        # Build messages
        messages = self._build_messages(prompt_or_messages)
        input_length = self._calculate_input_length(prompt_or_messages)

        # Log request details
        self.logger.info("🤖 Starting LLM API call...")
        self.logger.debug(f"Model: {self.model_name}")
        self.logger.debug(f"Input length: {input_length} characters")
        self.logger.debug(f"Number of messages: {len(messages)}")
        self.logger.debug(f"Context length: {self.ctx_length}")

        # Handle structured output based on configuration
        use_structured_output = response_format is not None or tools is not None
        structured_output_mode = None

        if use_structured_output:
            if self.config.structured_output == "response_format":
                if response_format:
                    required_fields = response_format.get("required", [])
                    self.logger.info(
                        f"📋 Requesting structured output (response_format) with {len(required_fields)} required fields: {required_fields}"
                    )
                    structured_output_mode = "response_format"
                elif tools:
                    # Convert tools to response_format for backward compatibility
                    if tools and len(tools) > 0:
                        tool_function = tools[0].get("function", {})
                        response_format = tool_function.get("parameters", {})
                        required_fields = response_format.get("required", [])
                        self.logger.info(
                            f"📋 Converting tools to response_format with {len(required_fields)} required fields: {required_fields}"
                        )
                        structured_output_mode = "response_format"
            elif self.config.structured_output == "tools":
                if tools:
                    required_fields = []
                    for tool in tools:
                        if tool.get("type") == "function":
                            params = tool.get("function", {}).get("parameters", {})
                            required_fields.extend(params.get("required", []))
                    self.logger.info(
                        f"🔧 Requesting structured output (tools) with {len(required_fields)} required fields: {required_fields}"
                    )
                    structured_output_mode = "tools"
                elif response_format:
                    # Convert response_format to tools
                    tools = schema_to_openai_tools(response_format)
                    required_fields = response_format.get("required", [])
                    self.logger.info(
                        f"🔧 Converting response_format to tools with {len(required_fields)} required fields: {required_fields}"
                    )
                    structured_output_mode = "tools"

        options = {
            "num_ctx": self.ctx_length
        }  # TODO: Update as min(tokens_needed_for_message, ctx_length)

        try:
            # Make the API call
            api_start_time = time.time()
            self.logger.debug("📡 Sending request to Ollama API...")

            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "options": options,
            }

            if structured_output_mode == "response_format":
                api_params["format"] = "json"
            elif structured_output_mode == "tools":
                api_params["tools"] = tools
                tool_name = tools[0].get("function", {}).get("name", "extract_metadata")
                options["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
                self.logger.debug("Using tools mode")

            api_params["options"] = options

            response: ollama.ChatResponse = self.client.chat(**api_params)
            print(f"Request: {api_params}")
            print(f"Response: {response}")

            api_time = time.time() - api_start_time
            content = response["message"]["content"]
            output_length = len(content)

            self.logger.info("✅ LLM API call completed successfully")
            self.logger.info("📊 LLM API Call Statistics:")
            self.logger.info(f"   • API processing time: {api_time:.2f}s")
            self.logger.info(f"   • Input length: {input_length} characters")
            self.logger.info(f"   • Output length: {output_length} characters")
            self.logger.info(
                f"   • Processing rate: {input_length/api_time:.0f} chars/sec"
            )
            self.logger.info(
                f"   • Generation rate: {output_length/api_time:.0f} chars/sec"
            )

            # Handle structured output
            if use_structured_output and structured_output_mode == "response_format":
                self.logger.debug("🔄 Parsing structured JSON response...")
                try:
                    parsed_response = self._parse_json_response(content)
                    self.logger.info("✅ JSON response parsed successfully")
                    self.logger.debug(
                        f"Parsed fields: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else type(parsed_response)}"
                    )
                    return parsed_response
                except Exception as parse_error:
                    self.logger.error(
                        f"❌ Failed to parse JSON response: {parse_error}"
                    )
                    self.logger.debug(f"Raw content: {content}")
                    raise
            elif use_structured_output and structured_output_mode == "tools":
                self.logger.debug("🔄 Parsing tools response...")
                try:
                    if response.message.tool_calls:
                        content = response.message.tool_calls[0].function.arguments
                    else:
                        content = response.message.content
                    parsed_response = self._parse_json_response(content)
                    self.logger.info("✅ JSON response parsed successfully")
                    self.logger.debug(
                        f"Parsed fields: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else type(parsed_response)}"
                    )
                    return parsed_response
                except Exception as parse_error:
                    self.logger.error(
                        f"❌ Failed to parse JSON response: {parse_error}"
                    )
                    self.logger.debug(f"Raw content: {content}")
                    raise

            self.logger.debug("📝 Returning text response")
            return content

        except ollama.ResponseError as e:
            api_time = time.time() - api_start_time
            if "timeout" in str(e).lower():
                self.logger.error(
                    f"⏰ Request to Ollama model '{self.model_name}' timed out after {api_time:.2f}s"
                )
                raise TimeoutError(
                    f"Request to Ollama model '{self.model_name}' timed out."
                ) from e
            self.logger.error(f"❌ Ollama API error after {api_time:.2f}s: {e}")
            raise RuntimeError(
                f"Error calling Ollama model '{self.model_name}': {e}"
            ) from e
        except httpx.ReadTimeout as e:
            api_time = time.time() - api_start_time
            self.logger.error(
                f"⏰ Request to Ollama model '{self.model_name}' timed out after {api_time:.2f}s"
            )
            raise TimeoutError(
                f"Request to Ollama model '{self.model_name}' timed out."
            ) from e
        except Exception as e:
            api_time = time.time() - api_start_time
            self.logger.error(
                f"❌ Unexpected error calling Ollama model '{self.model_name}' after {api_time:.2f}s: {e}"
            )
            raise RuntimeError(
                f"An unexpected error occurred while calling Ollama model '{self.model_name}': {e}"
            ) from e

    def _calculate_input_length(
        self, prompt_or_messages: Union[str, List[Dict[str, Any]]]
    ) -> int:
        """Calculate the total character length of the input."""
        if isinstance(prompt_or_messages, str):
            return len(prompt_or_messages)
        elif isinstance(prompt_or_messages, list):
            return sum(len(msg.get("content", "")) for msg in prompt_or_messages)
        else:
            return 0

    def _build_messages(
        self, prompt_or_messages: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build the messages list for the API call."""
        messages = []

        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add user messages
        if isinstance(prompt_or_messages, str):
            messages.append({"role": "user", "content": prompt_or_messages})
        elif isinstance(prompt_or_messages, list):
            messages.extend(prompt_or_messages)
        else:
            raise ValueError(
                "prompt_or_messages must be a string or list of message dictionaries"
            )

        return messages

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response, handling common formatting issues."""
        try:
            # Clean up common JSON formatting issues
            cleaned_content = content.strip()

            # Remove markdown code blocks if present
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:].strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:].strip()

            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3].strip()

            return json.loads(cleaned_content)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}. Raw content: {content}"
            )


class VllmLLM(LLM):
    """
    A client for interacting with a vLLM server using its OpenAI-compatible API.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the vLLM client.

        Args:
            config: Configuration object for the LLM
        """
        self.config = config
        # The base_url should point to the vLLM server's OpenAI-compatible endpoint,
        # e.g., "http://localhost:8000/v1"
        self.client = httpx.Client(
            base_url=config.base_url, timeout=config.default_timeout
        )

    def invoke(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt or message history to the model and get a response.
        """
        messages = self._build_messages(prompt_or_messages)

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": 4096,  # A reasonable default
        }

        # Handle structured output based on configuration
        use_structured_output = response_format is not None or tools is not None

        if use_structured_output:
            if self.config.structured_output == "response_format":
                if response_format:
                    payload["response_format"] = {
                        "type": "json_object",
                        "schema": response_format,
                    }
                elif tools:
                    # Convert tools to response_format
                    if tools and len(tools) > 0:
                        tool_function = tools[0].get("function", {})
                        payload["response_format"] = {
                            "type": "json_object",
                            "schema": tool_function.get("parameters", {}),
                        }
            elif self.config.structured_output == "tools":
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = (
                        "auto"  # Let the model decide which tool to use
                    )
                elif response_format:
                    # Convert response_format to tools
                    payload["tools"] = schema_to_openai_tools(response_format)
                    payload["tool_choice"] = "auto"

        try:
            # vLLM's OpenAI-compatible endpoint is typically at /chat/completions
            # and the base_url should be http://host:port or http://host:port/v1
            endpoint = (
                "/chat/completions"
                if "v1" in self.config.base_url
                else "/v1/chat/completions"
            )
            response = self.client.post(
                endpoint,
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content")

            # Handle tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]  # Take the first tool call
                if tool_call.get("type") == "function":
                    function_call = tool_call.get("function", {})
                    arguments_str = function_call.get("arguments", "{}")
                    try:
                        return self._parse_json_response(arguments_str)
                    except Exception as e:
                        self.logger.error(f"Failed to parse tool call arguments: {e}")
                        return {}
            elif use_structured_output:
                # Handle regular structured output
                if content:
                    return self._parse_json_response(content)
                else:
                    return {}
            else:
                return content or ""

        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Request to vLLM model '{self.config.model_name}' timed out."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Error calling vLLM model '{self.config.model_name}': {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while calling vLLM model '{self.config.model_name}': {e}"
            ) from e

    def _build_messages(
        self, prompt_or_messages: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build the messages list for the API call."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        if isinstance(prompt_or_messages, str):
            messages.append({"role": "user", "content": prompt_or_messages})
        elif isinstance(prompt_or_messages, list):
            messages.extend(prompt_or_messages)
        else:
            raise ValueError(
                "prompt_or_messages must be a string or list of message dictionaries"
            )
        return messages

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response, handling common formatting issues."""
        try:
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:].strip()
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:].strip()
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3].strip()
            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}. Raw content: {content}"
            )


# Example usage
def test():
    # Initialize the LLM client with tools mode
    config = LLMConfig.ollama(
        model_name="qwen3:latest",  # or whatever model you have
        system_prompt="You are a helpful assistant.",
        ctx_length=16000,
        default_timeout=120.0,
        structured_output="tools",  # Use tools mode
    )
    llm = get_llm(config)

    # Simple text generation
    response = llm.invoke("What is the capital of France?")
    print("Text response:", response)

    # Test with custom timeout
    try:
        timeout_config = LLMConfig(
            model_name="gpt-oss:20b",
            default_timeout=1.0,  # 1-second timeout
        )
        timeout_llm = get_llm(timeout_config)
        response = timeout_llm.invoke("Write a very long essay about AI")
        print("Quick response:", response)
    except TimeoutError as e:
        print("Timeout caught:", e)

    # JSON structured output using tools
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "number"},
        },
        "required": ["capital", "country"],
    }

    # Test with tools format
    tools = schema_to_openai_tools(schema)
    json_response = llm.invoke(
        "What is the capital of France? Include population if you know it.",
        tools=tools,
    )
    print("JSON response (tools):", json_response)

    # Also test with response_format for comparison
    config_response_format = LLMConfig.ollama(
        model_name="qwen3:latest",
        structured_output="response_format",
    )
    llm_response_format = get_llm(config_response_format)
    json_response_rf = llm_response_format.invoke(
        "What is the capital of France? Include population if you know it.",
        response_format=schema,
    )
    print("JSON response (response_format):", json_response_rf)

    # Message history
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    history_response = llm.invoke(messages)
    print("History response:", history_response)


# if __name__ == "__main__":
#     test()
