import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import ollama
import json
import time
import httpx


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
                api_params["format"] = response_format
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
    Client for models served via vLLM using the OpenAI-compatible /v1/chat/completions API.

    Supports plain text, message history, and structured outputs via either
    response_format json_schema or tools function-calling.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.system_prompt = config.system_prompt
        self.ctx_length = config.ctx_length

        self.logger = logging.getLogger("VllmLLM")
        self.logger.info(f"Initializing VllmLLM with model: {config.model_name}")
        self.logger.debug(f"Base URL: {config.base_url}")
        self.logger.debug(f"Context length: {config.ctx_length}")
        self.logger.debug(f"Default timeout: {config.default_timeout}s")

        try:
            # A single client with base_url reduces per-call overhead
            self.client = httpx.Client(
                base_url=config.base_url, timeout=config.default_timeout
            )
            self.logger.info("✅ VllmLLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize VllmLLM client: {e}")
            raise

    def invoke(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        invoke_start_time = time.time()

        messages = self._build_messages(prompt_or_messages)
        input_length = self._calculate_input_length(prompt_or_messages)

        self.logger.info("🤖 Starting vLLM API call...")
        self.logger.debug(f"Model: {self.model_name}")
        self.logger.debug(f"Input length: {input_length} characters")
        self.logger.debug(f"Number of messages: {len(messages)}")
        self.logger.debug(f"Context length: {self.ctx_length}")

        use_structured_output = response_format is not None or tools is not None
        structured_output_mode: Optional[str] = None

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }

        if use_structured_output:
            if self.config.structured_output == "response_format":
                if response_format:
                    structured_output_mode = "response_format"
                    # OpenAI-compatible json_schema format
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extract_metadata",
                            "schema": response_format,
                        },
                    }
                elif tools:
                    # Convert tools to response_format if needed
                    if tools and len(tools) > 0:
                        tool_function = tools[0].get("function", {})
                        rf = tool_function.get("parameters", {})
                        structured_output_mode = "response_format"
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": tool_function.get("name", "extract_metadata"),
                                "schema": rf,
                            },
                        }
            elif self.config.structured_output == "tools":
                if tools:
                    structured_output_mode = "tools"
                    payload["tools"] = tools
                    tool_name = (
                        tools[0].get("function", {}).get("name", "extract_metadata")
                    )
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_name},
                    }
                elif response_format:
                    # Convert response_format to tools
                    tools = schema_to_openai_tools(response_format)
                    structured_output_mode = "tools"
                    payload["tools"] = tools
                    tool_name = (
                        tools[0].get("function", {}).get("name", "extract_metadata")
                    )
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_name},
                    }

        try:
            api_start_time = time.time()
            self.logger.debug("📡 Sending request to vLLM API...")
            resp = self.client.post("/v1/chat/completions", json=payload)
            api_time = time.time() - api_start_time

            if resp.status_code >= 400:
                self.logger.error(
                    f"❌ vLLM API error after {api_time:.2f}s: HTTP {resp.status_code} {resp.text[:300]}"
                )
                raise RuntimeError(f"vLLM API error: HTTP {resp.status_code}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                self.logger.error("❌ vLLM API returned no choices")
                raise RuntimeError("vLLM API returned no choices")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            self.logger.info("✅ vLLM API call completed successfully")
            self.logger.info("📊 LLM API Call Statistics:")
            self.logger.info(f"   • API processing time: {api_time:.2f}s")
            self.logger.info(f"   • Input length: {input_length} characters")
            self.logger.info(f"   • Output length: {len(content)} characters")

            if use_structured_output and structured_output_mode == "response_format":
                self.logger.debug(
                    "🔄 Parsing structured JSON response (json_schema)..."
                )
                return self._parse_json_response(content)

            if use_structured_output and structured_output_mode == "tools":
                self.logger.debug("🔄 Parsing tools response...")
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    arguments = tool_calls[0].get("function", {}).get("arguments", "")
                    return self._parse_json_response(arguments)
                # Fallback to message content
                return self._parse_json_response(content)

            return content

        except httpx.ReadTimeout as e:
            api_time = time.time() - api_start_time  # type: ignore[name-defined]
            self.logger.error(
                f"⏰ Request to vLLM model '{self.model_name}' timed out after {api_time:.2f}s"
            )
            raise TimeoutError(
                f"Request to vLLM model '{self.model_name}' timed out."
            ) from e
        except Exception as e:
            api_time = time.time() - invoke_start_time
            self.logger.error(
                f"❌ Unexpected error calling vLLM model '{self.model_name}' after {api_time:.2f}s: {e}"
            )
            raise RuntimeError(
                f"An unexpected error occurred while calling vLLM model '{self.model_name}': {e}"
            ) from e

    def _calculate_input_length(
        self, prompt_or_messages: Union[str, List[Dict[str, Any]]]
    ) -> int:
        if isinstance(prompt_or_messages, str):
            return len(prompt_or_messages)
        elif isinstance(prompt_or_messages, list):
            return sum(len(msg.get("content", "")) for msg in prompt_or_messages)
        else:
            return 0

    def _build_messages(
        self, prompt_or_messages: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
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
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}. Raw content: {content}"
            )


# Tests were removed from the module; move tests to a dedicated test file.
