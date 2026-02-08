"""Client router that selects the appropriate client based on model name."""

from collections.abc import AsyncIterator
from typing import Any

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.clients.base import ChatCompletionClient, EmbeddingClient
from src.clients.ollama import ollama_client


class ClientRouter:
    """Routes requests to the appropriate client based on model name."""

    # Map of model names to their clients
    _MODEL_CLIENTS: dict[str, ChatCompletionClient] = {}

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

    def get_embedding_client(self, model: str) -> EmbeddingClient:
        """Get the appropriate embedding client for the given model.

        Args:
            model: The model identifier.

        Returns:
            The embedding client instance for the model. Defaults to Ollama client.
        """
        # Check if model has a custom client that supports embeddings
        if model.lower() in self._MODEL_CLIENTS:
            client = self._MODEL_CLIENTS[model.lower()]
            # Check if client implements EmbeddingClient
            if hasattr(client, "create_embedding"):
                return client  # type: ignore[return-value]

        # Default to Ollama for all embedding models
        return ollama_client  # type: ignore[return-value]

    async def create_embedding(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using the appropriate client.

        Args:
            model: The model identifier.
            input: Input text(s) to embed.
            **kwargs: Additional parameters.

        Returns:
            CreateEmbeddingResponse with embeddings.
        """
        client = self.get_embedding_client(model)
        return await client.create_embedding(
            model=model,
            input=input,
            **kwargs,
        )


# Singleton router instance
client_router = ClientRouter()
