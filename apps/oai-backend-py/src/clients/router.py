"""Client router that selects the appropriate client based on model name."""

from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.clients.base import ChatCompletionClient
from src.clients.ollama import ollama_client
from src.clients.radchat import radchat_client


class ClientRouter:
    """Routes requests to the appropriate client based on model name."""

    # Map of model names to their clients
    _MODEL_CLIENTS: dict[str, ChatCompletionClient] = {
        "radchat": radchat_client,
    }

    def get_client(self, model: str) -> ChatCompletionClient:
        """Get the appropriate client for the given model.

        Args:
            model: The model identifier.

        Returns:
            The client instance for the model. Defaults to Ollama client.
        """
        # Check if model has a custom client
        if model.lower() in self._MODEL_CLIENTS:
            return self._MODEL_CLIENTS[model.lower()]

        # Default to Ollama for all other models
        return ollama_client

    async def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using the appropriate client.

        Args:
            model: The model identifier.
            messages: List of message dictionaries.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.

        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk].
        """
        client = self.get_client(model)
        return await client.create_completion(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )


# Singleton router instance
client_router = ClientRouter()
