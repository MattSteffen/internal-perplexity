from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
import ollama
import json
import threading
import time

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
        return cls(
            model_name=config.get("model_name"),
            base_url=config.get("base_url"),
            system_prompt=config.get("system_prompt"),
            ctx_length=config.get("ctx_length"),
            default_timeout=config.get("default_timeout"),
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
        self.model_name = config.model_name
        self.system_prompt = config.system_prompt
        self.ctx_length = config.ctx_length

        # Initialize Ollama client with timeout from config
        self.client = ollama.Client(host=config.base_url, timeout=config.default_timeout)

    def invoke(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt or message history to the model and get a response.

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            response_format: Optional JSON schema for structured output

        Returns:
            String response or parsed JSON dictionary if response_format is provided

        Raises:
            TimeoutError: If the request exceeds the timeout duration
            RuntimeError: If there's an error with the API call
        """
        messages = self._build_messages(prompt_or_messages)
        options = {"num_ctx": self.ctx_length}

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format="json" if response_format else None,
                options=options,
            )
            content = response["message"]["content"]

            if response_format is not None:
                return self._parse_json_response(content)
            return content
        except ollama.ResponseError as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"Request to Ollama model '{self.model_name}' timed out."
                ) from e
            raise RuntimeError(
                f"Error calling Ollama model '{self.model_name}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while calling Ollama model '{self.model_name}': {e}"
            ) from e

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


# Example usage
def test():
    # Initialize the LLM client
    llm = LLM(
        LLMConfig(
            model_name="qwen3:latest",  # or whatever model you have
            system_prompt="You are a helpful assistant.",
            ctx_length=16000,
            default_timeout=120.0,
        )
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
