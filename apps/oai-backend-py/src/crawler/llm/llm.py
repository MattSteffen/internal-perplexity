import json
from abc import ABC, abstractmethod
from typing import Any

import httpx
import ollama
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """
    Configuration for Large Language Model providers.

    This model provides type-safe configuration for connecting to various LLM
    services with automatic validation of parameters and connection settings.

    Attributes:
        model_name: Name of the LLM model to use
        base_url: Base URL for the LLM API service
        system_prompt: Optional system prompt for the model
        ctx_length: Context length (token limit) for the model
        default_timeout: Default timeout in seconds for API calls
        provider: Provider name (e.g., 'ollama', 'openai', 'vllm')
        api_key: API key for authentication (if required by provider)
        structured_output: Mode for structured output ('response_format' or 'tools')
    """

    model_name: str = Field(..., min_length=1, description="Name of the LLM model to use")
    base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
        description="Base URL for the LLM API service",
    )
    system_prompt: str | None = Field(default=None, description="Optional system prompt to set model behavior")
    ctx_length: int = Field(default=32000, gt=0, description="Context length (token limit) for the model")
    default_timeout: float = Field(default=300.0, gt=0, description="Default timeout in seconds for API calls")
    provider: str = Field(default="ollama", description="Provider name (e.g., 'ollama', 'openai', 'vllm')")
    api_key: str = Field(default="", description="API key for authentication (if required by provider)")
    structured_output: str = Field(
        default="response_format",
        description="Mode for structured output: 'response_format' or 'tools'",
    )

    model_config = {
        "validate_assignment": True,
    }

    @field_validator("structured_output")
    @classmethod
    def validate_structured_output(cls, v: str) -> str:
        """Validate structured output mode."""
        if v not in ["response_format", "tools"]:
            raise ValueError("structured_output must be either 'response_format' or 'tools'")
        return v

    @classmethod
    def ollama(
        cls,
        model_name: str,
        base_url: str = "http://localhost:11434",
        system_prompt: str | None = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> LLMConfig:
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
        system_prompt: str | None = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> LLMConfig:
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
        system_prompt: str | None = None,
        ctx_length: int = 32000,
        default_timeout: float = 300.0,
        structured_output: str = "response_format",
    ) -> LLMConfig:
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


def schema_to_openai_tools(schema: dict[str, Any]) -> list[dict[str, Any]]:
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
        prompt_or_messages: str | list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
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

        # Initialize Ollama client with timeout from config
        self.client = ollama.Client(host=config.base_url, timeout=config.default_timeout)

    def invoke(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        """
        Send a prompt or message history to the model and get a response.

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
        # Build messages
        messages = self._build_messages(prompt_or_messages)

        # Handle structured output based on configuration
        use_structured_output = response_format is not None or tools is not None
        structured_output_mode = None

        if use_structured_output:
            if self.config.structured_output == "response_format":
                if response_format:
                    structured_output_mode = "response_format"
                elif tools:
                    # Convert tools to response_format for backward compatibility
                    if tools and len(tools) > 0:
                        tool_function = tools[0].get("function", {})
                        response_format = tool_function.get("parameters", {})
                        structured_output_mode = "response_format"
            elif self.config.structured_output == "tools":
                if tools:
                    structured_output_mode = "tools"
                elif response_format:
                    # Convert response_format to tools
                    tools = schema_to_openai_tools(response_format)
                    structured_output_mode = "tools"

        options: dict[str, Any] = {"num_ctx": self.ctx_length}  # TODO: Update as min(tokens_needed_for_message, ctx_length)

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "options": options,
            }

            if structured_output_mode == "response_format":
                if response_format is None:
                    raise ValueError("response_format is required when structured_output_mode is 'response_format'")
                api_params["format"] = response_format
            elif structured_output_mode == "tools":
                if tools is None:
                    raise ValueError("tools is required when structured_output_mode is 'tools'")
                api_params["tools"] = tools
                tool_name = tools[0].get("function", {}).get("name", "extract_metadata")
                options["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            api_params["options"] = options

            response: ollama.ChatResponse = self.client.chat(**api_params)

            content = response["message"]["content"]

            # Handle structured output
            if use_structured_output and structured_output_mode == "response_format":
                parsed_response = self._parse_json_response(content)
                return parsed_response
            elif use_structured_output and structured_output_mode == "tools":
                if response.message.tool_calls:
                    content = response.message.tool_calls[0].function.arguments
                else:
                    content = response.message.content
                # content might already be a dict (parsed JSON) or a string
                parsed_response = self._parse_json_response(content)
                return parsed_response

            return content  # type: ignore[no-any-return,return-value]

        except ollama.ResponseError as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(f"Request to Ollama model '{self.model_name}' timed out.") from e
            raise RuntimeError(f"Error calling Ollama model '{self.model_name}': {e}") from e
        except httpx.ReadTimeout as e:
            raise TimeoutError(f"Request to Ollama model '{self.model_name}' timed out.") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while calling Ollama model '{self.model_name}': {e}") from e

    def _build_messages(self, prompt_or_messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            raise ValueError("prompt_or_messages must be a string or list of message dictionaries")

        return messages

    def _parse_json_response(self, content: str | dict[str, Any]) -> dict[str, Any]:
        """Parse JSON response, handling common formatting issues."""
        # If content is already a dict, return it directly
        if isinstance(content, dict):
            return content

        # Otherwise, parse the string
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

            return json.loads(cleaned_content)  # type: ignore[no-any-return]

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Raw content: {content}")


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

        # A single client with base_url reduces per-call overhead
        self.client = httpx.Client(base_url=config.base_url, timeout=config.default_timeout)

    def invoke(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        messages = self._build_messages(prompt_or_messages)

        use_structured_output = response_format is not None or tools is not None
        structured_output_mode: str | None = None

        payload: dict[str, Any] = {
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
                    tool_name = tools[0].get("function", {}).get("name", "extract_metadata")
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_name},
                    }
                elif response_format:
                    # Convert response_format to tools
                    tools = schema_to_openai_tools(response_format)
                    structured_output_mode = "tools"
                    payload["tools"] = tools
                    tool_name = tools[0].get("function", {}).get("name", "extract_metadata")
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_name},
                    }

        try:
            resp = self.client.post("/v1/chat/completions", json=payload)

            if resp.status_code >= 400:
                raise RuntimeError(f"vLLM API error: HTTP {resp.status_code}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("vLLM API returned no choices")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if use_structured_output and structured_output_mode == "response_format":
                return self._parse_json_response(content)

            if use_structured_output and structured_output_mode == "tools":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    arguments = tool_calls[0].get("function", {}).get("arguments", "")
                    return self._parse_json_response(arguments)
                # Fallback to message content
                return self._parse_json_response(content)

            return content  # type: ignore[no-any-return,return-value]

        except httpx.ReadTimeout as e:
            raise TimeoutError(f"Request to vLLM model '{self.model_name}' timed out.") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while calling vLLM model '{self.model_name}': {e}") from e

    def _build_messages(self, prompt_or_messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if isinstance(prompt_or_messages, str):
            messages.append({"role": "user", "content": prompt_or_messages})
        elif isinstance(prompt_or_messages, list):
            messages.extend(prompt_or_messages)
        else:
            raise ValueError("prompt_or_messages must be a string or list of message dictionaries")
        return messages

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}. Raw content: {content}")


# Tests were removed from the module; move tests to a dedicated test file.
