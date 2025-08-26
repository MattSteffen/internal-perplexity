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
    """Configuration for Ollama LLM."""

    model_name: str
    base_url: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    ctx_length: int = 32000
    default_timeout: float = 300.0
    provider: str = "ollama"

    @classmethod
    def from_dict(cls, config: Dict[str, any]):
        # Support both 'model' and 'model_name' keys for flexibility
        model_name = config.get("model_name") or config.get("model")
        if not model_name:
            raise ValueError("Either 'model_name' or 'model' must be provided")

        base_url = config.get("base_url", "http://localhost:11434")
        if not base_url:
            raise ValueError("LLM base_url cannot be empty")

        return cls(
            model_name=model_name,
            base_url=base_url,
            system_prompt=config.get("system_prompt"),
            ctx_length=config.get("ctx_length", 32000),
            default_timeout=config.get("default_timeout", 300.0),
            provider=config.get("provider", "ollama"),
        )


# TODO: Implement for vllm


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
    ) -> Union[str, Dict[str, Any]]:
        """Send a prompt or message history to the model and get a response.

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output
            timeout: Request timeout in seconds

        Returns:
            String response or parsed JSON dictionary if response_format is provided
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
        self.logger = logging.getLogger('OllamaLLM')
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
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt or message history to the model and get a response with comprehensive logging.

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output

        Returns:
            String response or parsed JSON dictionary if response_format is provided

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

        if response_format:
            required_fields = response_format.get("required", [])
            self.logger.info(f"📋 Requesting structured output with {len(required_fields)} required fields: {required_fields}")

        options = {
            "num_ctx": self.ctx_length
        }  # TODO: Update as min(tokens_needed_for_message, ctx_length)

        try:
            # Make the API call
            api_start_time = time.time()
            self.logger.debug("📡 Sending request to Ollama API...")

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format="json" if response_format else None,
                options=options,
            )

            api_time = time.time() - api_start_time
            content = response["message"]["content"]
            output_length = len(content)

            self.logger.info("✅ LLM API call completed successfully")
            self.logger.info("📊 LLM API Call Statistics:")
            self.logger.info(f"   • API processing time: {api_time:.2f}s")
            self.logger.info(f"   • Input length: {input_length} characters")
            self.logger.info(f"   • Output length: {output_length} characters")
            self.logger.info(f"   • Processing rate: {input_length/api_time:.0f} chars/sec")
            self.logger.info(f"   • Generation rate: {output_length/api_time:.0f} chars/sec")

            # Handle structured output
            if response_format is not None:
                self.logger.debug("🔄 Parsing structured JSON response...")
                try:
                    parsed_response = self._parse_json_response(content)
                    self.logger.info("✅ JSON response parsed successfully")
                    self.logger.debug(f"Parsed fields: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else type(parsed_response)}")
                    return parsed_response
                except Exception as parse_error:
                    self.logger.error(f"❌ Failed to parse JSON response: {parse_error}")
                    self.logger.debug(f"Raw content: {content}")
                    raise

            self.logger.debug("📝 Returning text response")
            return content

        except ollama.ResponseError as e:
            api_time = time.time() - api_start_time
            if "timeout" in str(e).lower():
                self.logger.error(f"⏰ Request to Ollama model '{self.model_name}' timed out after {api_time:.2f}s")
                raise TimeoutError(
                    f"Request to Ollama model '{self.model_name}' timed out."
                ) from e
            self.logger.error(f"❌ Ollama API error after {api_time:.2f}s: {e}")
            raise RuntimeError(
                f"Error calling Ollama model '{self.model_name}': {e}"
            ) from e
        except httpx.ReadTimeout as e:
            api_time = time.time() - api_start_time
            self.logger.error(f"⏰ Request to Ollama model '{self.model_name}' timed out after {api_time:.2f}s")
            raise TimeoutError(
                f"Request to Ollama model '{self.model_name}' timed out."
            ) from e
        except Exception as e:
            api_time = time.time() - api_start_time
            self.logger.error(f"❌ Unexpected error calling Ollama model '{self.model_name}' after {api_time:.2f}s: {e}")
            raise RuntimeError(
                f"An unexpected error occurred while calling Ollama model '{self.model_name}': {e}"
            ) from e

    def _calculate_input_length(self, prompt_or_messages: Union[str, List[Dict[str, Any]]]) -> int:
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

        if response_format:
            payload["response_format"] = {
                "type": "json_object",
                "schema": response_format,
            }

        try:
            # vLLM's OpenAI-compatible endpoint is typically at /chat/completions
            # and the base_url should be http://host:port or http://host:port/v1
            endpoint = "/chat/completions" if "v1" in self.config.base_url else "/v1/chat/completions"
            response = self.client.post(
                endpoint,
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            if response_format:
                return self._parse_json_response(content)
            return content

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
    # Initialize the LLM client
    config = LLMConfig(
        model_name="qwen3:latest",  # or whatever model you have
        system_prompt="You are a helpful assistant.",
        ctx_length=16000,
        default_timeout=120.0,
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

    # JSON structured output
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "number"},
        },
        "required": ["capital", "country"],
    }

    json_response = llm.invoke(
        "What is the capital of France? Include population if you know it.",
        response_format=schema,
    )
    print("JSON response:", json_response)

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
